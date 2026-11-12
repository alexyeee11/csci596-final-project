# Laptop: `make check-cpu` (MPI only). CARC: `source scripts/env.sh && make gpu`.
# ponytail: no header dependency tracking; run `make clean` after editing a header.
MPICC     ?= mpicc
NVCC      ?= nvcc
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell which nvcc 2>/dev/null))
ARCH      ?= -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89
CFLAGS    = -O2 -Wall -Isrc
NVFLAGS   = -O2 $(ARCH) -Isrc
GPU_INC   = -I$(CUDA_HOME)/include
GPU_LIBS  = -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++ -lm
KERNELS   = build/gemm.o build/layers.o build/mlp.o

.PHONY: cpu gpu check-cpu clean
cpu: bin/test_allreduce bin/test_cifar bin/bench_comm
gpu: cpu bin/smoke bin/test_gemm bin/test_layers bin/test_mlp bin/train bin/bench_gemm

bin build:
	mkdir -p $@

build/%.o: src/kernels/%.cu | build
	$(NVCC) $(NVFLAGS) -c $< -o $@
build/mlp.o: src/model/mlp.cu | build
	$(NVCC) $(NVFLAGS) -c $< -o $@

bin/test_allreduce: tests/test_allreduce.c src/comm/allreduce.c | bin
	$(MPICC) $(CFLAGS) $^ -o $@
bin/bench_comm: bench/bench_comm.c src/comm/allreduce.c | bin
	$(MPICC) $(CFLAGS) $^ -o $@
bin/test_cifar: tests/test_cifar.c src/data/cifar.c | bin
	$(MPICC) $(CFLAGS) $^ -o $@ -lm
bin/smoke: tests/smoke.c | bin
	$(MPICC) $(CFLAGS) $(GPU_INC) $^ -o $@ $(GPU_LIBS)
bin/test_gemm: tests/test_gemm.cu build/gemm.o | bin
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcublas
bin/test_layers: tests/test_layers.cu build/layers.o | bin
	$(NVCC) $(NVFLAGS) $^ -o $@
bin/test_mlp: tests/test_mlp.cu $(KERNELS) | bin
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcublas
bin/bench_gemm: bench/bench_gemm.cu build/gemm.o | bin
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcublas
bin/train: src/train.c src/comm/allreduce.c src/comm/overlap.c src/data/cifar.c $(KERNELS) | bin
	$(MPICC) $(CFLAGS) $(GPU_INC) $^ -o $@ $(GPU_LIBS) -lpthread

check-cpu: cpu
	./bin/test_cifar
	for n in 1 2 3 4; do mpirun --oversubscribe -np $$n ./bin/test_allreduce || exit 1; done
	test "$$(mpirun --oversubscribe -np 2 ./bin/bench_comm 65536 | wc -l | tr -d ' ')" -eq 41

clean:
	rm -rf bin build
