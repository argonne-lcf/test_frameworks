import datetime

t1 = datetime.datetime.now()
import torch

t2 = datetime.datetime.now()
from mpi4py import MPI

comm = MPI.COMM_WORLD

import os
import socket

import numpy as np
import torch.distributed as dist
import torch.nn.parallel

from torch_setup import (
    get_device,
    get_device_type,
    get_profiler_activities,
    init_distributed,
)

import_time = (t2 - t1).total_seconds()
t3 = datetime.datetime.now()
dist, rank, world_size = init_distributed()
t4 = datetime.datetime.now()
init_time = (t4 - t3).total_seconds()

dist_my_rank = dist.get_rank()
dist_world_size = dist.get_world_size()

if rank == 0:
    print(f"Torch version: {torch.__version__}")
    print(f"Torch installation: {torch.__file__}")
    print(f"Import time: {import_time}")
    print(f"Init time: {init_time}")

device = get_device()


def print_rank_0(msg):
    if rank == 0:
        print(msg)


#comm.barrier()
#dist.barrier(device_ids=[torch.xpu.current_device()])
dist.barrier()
torch.cuda.synchronize()

niters = 10

time_iters = np.zeros(niters)

print_rank_0("Allreduce")
for i in range(niters):
    x = torch.ones([1024, 1024]).to(device, non_blocking=True)
    # print_rank_0(x)
    t5 = datetime.datetime.now()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)  # Added Extra op
    torch.cuda.synchronize()
    t6 = datetime.datetime.now()
    elapsed = (t6 - t5).total_seconds()
    time_iters[i] = elapsed
    print_rank_0(f"[{dist_my_rank}] Iter-{i}: {elapsed:.8f}")
print_rank_0(
    f"Average time for all_reduce (exclude first iter): {np.mean(time_iters[1:]):.8f} "
)

