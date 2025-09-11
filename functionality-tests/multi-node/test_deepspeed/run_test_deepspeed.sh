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
#module load hdf5

## To get the base conda activated. This is a Spack based miniforge installation
#source /opt/aurora/25.190.0/spack/unified/0.10.0/install/linux-sles15-x86_64/gcc-13.3.0/miniforge3-24.3.0-0-gfganax/bin/activate
#
#
## I did some hacks here: I copied "cp -r /lus/tegu/projects/datasets/software/wheelforge/repositories/deepspeed_0.17.5_rel_08_21_2025/DeepSpeed/csrc ."
## to the /lus/tegu/projects/datasets/software/wheelforge/envs/conda_envs/numba_dpex_0.23.0_dpnp_0.18.1_dpctl_0.20.0_pytorch_2.8.0_nre_oneapi_2025.2.0_numpy_2.0.2_python3p10p14/lib/python3.10/site-packages/deepspeed/ops
## directory. For some reason, the installation process did not copy these files over!
#conda activate /lus/tegu/projects/datasets/software/wheelforge/envs/conda_envs/numba_dpex_0.23.0_dpnp_0.18.1_dpctl_0.20.0_pytorch_2.8.0_nre_oneapi_2025.2.0_numpy_2.0.2_python3p10p14

## This one has pip installed DeepSpeed==0.17.5, apparently pip also builds the 
## wheel on the machine and then installs it!!!
#conda activate /lus/tegu/projects/datasets/software/wheelforge/envs/conda_envs/frameworks_2025.2.0_RC3

## This one has torchtune installed last of all, and has functional torchtune
#conda activate /lus/tegu/projects/datasets/software/wheelforge/envs/conda_envs/frameworks_2025.2.0_RC6

#module use /home/cchannui/khalid/frameworks-test14/modulefiles
module add frameworks/2025.2.0

#conda activate /lus/tegu/projects/datasets/software/wheelforge/envs/conda_envs/aurora-frameworks-RC8-2025.2.0

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_MR_CACHE_MONITOR=userfaultfd

export CCL_PROCESS_LAUNCHER=pmix
#export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
#export CCL_OP_SYNC=1

export CPU_AFFINITY="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} python test_miniGPT.py --epochs=1

