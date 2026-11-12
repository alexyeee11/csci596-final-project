# source this on CARC Discovery before building or running
module purge
module load gcc openmpi cuda
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
# MPI launcher inside Slurm jobs; check `srun --mpi=list` if pmix is not listed
export SRUN_MPI="${SRUN_MPI:-pmix}"
# NCCL: Discovery's only usable module is in the ver/2506 stack (gcc 14), so link its library directly
# instead of switching every module to that stack. Override NCCL_HOME, or set it empty to build without NCCL.
NCCL_2506=/apps/spack/2506/apps/linux-rocky8-x86_64_v3/gcc-14.3.0/nccl-2.27.7-1-rniyhkw
if [ -z "${NCCL_HOME+set}" ] && [ -d "$NCCL_2506" ]; then export NCCL_HOME="$NCCL_2506"; fi
