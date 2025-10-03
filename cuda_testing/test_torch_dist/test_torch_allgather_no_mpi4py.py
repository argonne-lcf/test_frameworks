import datetime

t1 = datetime.datetime.now()
import torch

t2 = datetime.datetime.now()
#from mpi4py import MPI

#comm = MPI.COMM_WORLD

import os
import socket

import numpy as np
import torch.distributed as dist
import torch.nn.parallel

#from torch_setup import (
#    get_device,
#    get_device_type,
#    get_profiler_activities,
#    init_distributed,
#)

import_time = (t2 - t1).total_seconds()
t3 = datetime.datetime.now()

device = torch.device(f"xpu:{int(os.environ['PALS_RANKID']) % int(os.environ['PALS_LOCAL_SIZE'])}")

#dist, rank, world_size = init_distributed()
torch.distributed.init_process_group(backend="xccl", init_method="env://",
                        world_size=int(os.environ["PALS_WORLD_SIZE"]),
                        rank=int(os.environ["PALS_RANKID"]))
rank = int(os.environ["PALS_RANKID"])
world_size = int(os.environ["PALS_WORLD_SIZE"])
t4 = datetime.datetime.now()
init_time = (t4 - t3).total_seconds()

#dist_my_rank = dist.get_rank()
#dist_world_size = dist.get_world_size()

dist_my_rank = rank
dist_world_size = world_size


if rank == 0:
    print(f"Torch version: {torch.__version__}")
    print(f"Torch installation: {torch.__file__}")
    print(f"Import time: {import_time}")
    print(f"Init time: {init_time}")

#device = get_device()

def print_rank_0(msg):
    if rank == 0:
        print(msg)


#comm.barrier()
dist.barrier()
niters = 10

time_iters = np.zeros(niters)

print_rank_0("Allgather")
for i in range(niters):
    x = torch.ones(4).to(device, non_blocking=True)
    y = [torch.zeros(4).to(device, non_blocking=True) for _ in range(world_size)]
    # print_rank_0(x)
    t5 = datetime.datetime.now()
    dist.all_gather(y, x)
    t6 = datetime.datetime.now()
    elapsed = (t6 - t5).total_seconds()
    print_rank_0(f"[{dist_my_rank}] Iter-{i}: {elapsed:.8f}")
print_rank_0(
    f"Average time for all_gather (exclude first iter): {np.mean(time_iters[1:]):.8f} "
)
