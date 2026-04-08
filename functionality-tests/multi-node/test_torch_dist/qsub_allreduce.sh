#!/bin/bash -x
#PBS -l select=2048
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -N SF_2048
#PBS -e /lus/flare/projects/datasets/softwares/testing/huihuo_testing_frameworks/test_frameworks/functionality-tests/multi-node/test_torch_dist/errordir_aurora
#PBS -o /lus/flare/projects/datasets/softwares/testing/huihuo_testing_frameworks/test_frameworks/functionality-tests/multi-node/test_torch_dist/outdir_aurora
#PBS -j oe

#
## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

BENCH_DIR=/lus/flare/projects/datasets/softwares/testing/huihuo_testing_frameworks/test_frameworks/functionality-tests/multi-node/test_torch_dist

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

echo "NUMBER_OF_NODES=${NNODES}"

#N=2
#PPN=2

#let PALS_WORLD_SIZE=${N}*${PPN}

let PALS_WORLD_SIZE=${NNODES}*${NRANKS_PER_NODE}


export PALS_WORLD_SIZE=${PALS_WORLD_SIZE}

echo "PALS_WORLD_SIZE = ${PALS_WORLD_SIZE}"

#module load pti-gpu
#module load hdf5

## To get the base conda activated. This is a Spack based miniforge installation
#source /opt/aurora/25.190.0/spack/unified/0.10.0/install/linux-sles15-x86_64/gcc-13.3.0/miniforge3-24.3.0-0-gfganax/bin/activate

#conda activate /lus/tegu/projects/datasets/software/wheelforge/envs/conda_envs/numba_dpex_0.23.0_dpnp_0.18.1_dpctl_0.20.0_pytorch_2.8.0_nre_oneapi_2025.2.0_numpy_2.0.2_python3p10p14

#module use /home/cchannui/khalid/modulefiles
module add frameworks

#module load py-mpi4py/4.0.1

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_MR_CACHE_MONITOR=userfaultfd

export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
#export CCL_LOG_LEVEL=debug

## For 1024+ nodes, try:
export CCL_KVS_MODE=mpi

#export CCL_ALLGATHERV_SCALEOUT=ring

#export CPU_AFFINITY="list:4-7"
#export CCL_WORKER_AFFINITY="42"
#export ZE_AFFINITY_MASK="0"

#export CPU_AFFINITY="list:4-7:8-11"
#export CCL_WORKER_AFFINITY="42,43"
#export ZE_AFFINITY_MASK="0,1"
#
#export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=2048 #4096
#export FI_CXI_DEFAULT_CQ_SIZE=4096
export FI_CXI_RX_MATCH_MODE=hybrid


export CPU_AFFINITY="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"

#mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} python test_torch_allgather.py

mpiexec -n ${PALS_WORLD_SIZE} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    -env MASTER_ADDR=$(hostname).hsn.cm.aurora.alcf.anl.gov \
    -env MASTER_PORT=2345 python ${BENCH_DIR}/test_allreduce_nathan.py
