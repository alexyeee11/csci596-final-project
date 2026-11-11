#include "comm/overlap.h"
#include <pthread.h>
#include <stdlib.h>
#include "cuda_check.h"

/* Bucket i covers [start(i), end(i)) with end(0) = n: bucket 0 is the tail of the buffer (last layer). */
struct Overlap {
  float* dgrads;
  size_t n, bucket;
  int nb;
  ar_algo algo;
  MPI_Comm comm;
  int device;
  float* host;           /* pinned, n floats */
  cudaStream_t copy;     /* non-blocking, so it does not serialize with the legacy default stream */
  cudaEvent_t* ready;    /* ready[i]: bucket i's gradient kernels are done (recorded on the default stream) */
  int queued;            /* buckets handed to the comm thread this step; guarded by mu */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  pthread_t thread;
};

static size_t bend(const Overlap* o, int i) { return o->n - (size_t)i * o->bucket; }
static size_t bstart(const Overlap* o, int i) { return bend(o, i) > o->bucket ? bend(o, i) - o->bucket : 0; }

Overlap* overlap_create(float* dgrads, size_t n, size_t bucket_floats, ar_algo algo, MPI_Comm comm, int device) {
  Overlap* o = calloc(1, sizeof *o);
  o->dgrads = dgrads;
  o->n = n;
  o->bucket = bucket_floats < n ? bucket_floats : n;
  o->nb = (int)((n + o->bucket - 1) / o->bucket);
  o->algo = algo;
  o->comm = comm;
  o->device = device;
  CUDA_CHECK(cudaMallocHost((void**)&o->host, n * sizeof(float)));
  CUDA_CHECK(cudaStreamCreateWithFlags(&o->copy, cudaStreamNonBlocking));
  o->ready = malloc(o->nb * sizeof(cudaEvent_t));
  for (int i = 0; i < o->nb; i++) CUDA_CHECK(cudaEventCreateWithFlags(&o->ready[i], cudaEventDisableTiming));
  pthread_mutex_init(&o->mu, NULL);
  pthread_cond_init(&o->cv, NULL);
  return o;
}

/* Buckets come in order 0, 1, ...; every rank queues the same ones in the same order, so MPI calls match. */
static void* comm_main(void* arg) {
  Overlap* o = arg;
  CUDA_CHECK(cudaSetDevice(o->device));  /* the current device is per thread and defaults to 0 */
  for (int i = 0; i < o->nb; i++) {
    pthread_mutex_lock(&o->mu);
    while (o->queued <= i) pthread_cond_wait(&o->cv, &o->mu);
    pthread_mutex_unlock(&o->mu);
    size_t s = bstart(o, i), len = bend(o, i) - s;
    CUDA_CHECK(cudaStreamWaitEvent(o->copy, o->ready[i], 0));
    CUDA_CHECK(cudaMemcpyAsync(o->host + s, o->dgrads + s, len * sizeof(float), cudaMemcpyDeviceToHost, o->copy));
    CUDA_CHECK(cudaStreamSynchronize(o->copy));
    allreduce_sum(o->host + s, len, o->algo, o->comm);
    CUDA_CHECK(cudaMemcpyAsync(o->dgrads + s, o->host + s, len * sizeof(float), cudaMemcpyHostToDevice, o->copy));
  }
  CUDA_CHECK(cudaStreamSynchronize(o->copy));
  return NULL;
}

void overlap_begin(Overlap* o) {
  o->queued = 0;
  if (pthread_create(&o->thread, NULL, comm_main, o)) { perror("pthread_create"); exit(1); }
}

/* Layer `layer`'s [dW | db] starts at offset off and its kernels are enqueued: every bucket starting at or after
   off is complete once the default stream reaches this point. */
void overlap_on_grad(int layer, float* grad, size_t n, void* ctx) {
  (void)layer; (void)n;
  Overlap* o = ctx;
  size_t off = (size_t)(grad - o->dgrads);
  int q = o->queued;  /* only this thread writes queued */
  while (q < o->nb && bstart(o, q) >= off) CUDA_CHECK(cudaEventRecord(o->ready[q++], 0));
  if (q == o->queued) return;
  pthread_mutex_lock(&o->mu);
  o->queued = q;
  pthread_cond_signal(&o->cv);
  pthread_mutex_unlock(&o->mu);
}

void overlap_wait(Overlap* o) {
  pthread_join(o->thread, NULL);
}

void overlap_free(Overlap* o) {
  for (int i = 0; i < o->nb; i++) cudaEventDestroy(o->ready[i]);
  free(o->ready);
  cudaStreamDestroy(o->copy);
  cudaFreeHost(o->host);
  pthread_mutex_destroy(&o->mu);
  pthread_cond_destroy(&o->cv);
  free(o);
}
