#!/bin/bash -x
#
## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module load pti-gpu
module load hdf5

## To get the base conda activated. This is a Spack based miniforge installation
#source /opt/aurora/25.190.0/spack/unified/0.10.0/install/linux-sles15-x86_64/gcc-13.3.0/miniforge3-24.3.0-0-gfganax/bin/activate

#conda activate /lus/tegu/projects/datasets/software/wheelforge/envs/conda_envs/vllm_0.10.1_torchtune_0.6.1_torchdata_0.11.0_torchao_0.12.0_h5py_3.14.0_mpi4py_4.1.0_torchvision_0.23.0_oneapi_2025.2.0_pti_0.12.3_numpy_2.0.2_python3p10p14
#
module add frameworks/2025.2.0


export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_MR_CACHE_MONITOR=userfaultfd

export CCL_PROCESS_LAUNCHER=pmix

export CPU_AFFINITY="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} python test_mpi4py.py

