# source this on CARC Discovery before building or running
module purge
module load gcc openmpi cuda
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
# MPI launcher inside Slurm jobs; check `srun --mpi=list` if pmix is not listed
export SRUN_MPI="${SRUN_MPI:-pmix}"
