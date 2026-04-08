import datetime

t1 = datetime.datetime.now()
import torch

t2 = datetime.datetime.now()
import os
import socket

import numpy as np
import torch.distributed as dist
import torch.nn.parallel

import_time = (t2 - t1).total_seconds()
t3 = datetime.datetime.now()

device = torch.device(f"xpu:{int(os.environ['PALS_RANKID']) % int(os.environ['PALS_LOCAL_SIZE'])}")
torch.xpu.set_device(device.index or 0)

torch.distributed.init_process_group(backend="xccl", init_method="env://",
                        world_size=int(os.environ["PALS_WORLD_SIZE"]),
                        rank=int(os.environ["PALS_RANKID"]))
rank = int(os.environ["PALS_RANKID"])
world_size = int(os.environ["PALS_WORLD_SIZE"])
t4 = datetime.datetime.now()
init_time = (t4 - t3).total_seconds()

dist_my_rank = rank
dist_world_size = world_size

#dim_size=32768 ## 2 GB per 2D array
dim_size=46341 ## ~4.01 GB per 2D array

if rank == 0:
    print(f"Torch version: {torch.__version__}")
    print(f"Torch installation: {torch.__file__}")
    print(f"Import time: {import_time}")
    print(f"Init time: {init_time}")
    print(f"MSG Size = {(dim_size * dim_size * 2) / 1000 / 1000} MB")

def print_rank_0(msg):
    if rank == 0:
        print(msg)


niters = 10

time_iters = np.zeros(niters)

dist.barrier(device_ids=[torch.xpu.current_device()])
torch.xpu.synchronize()

print_rank_0("Allreduce")
for i in range(niters):
    dist.barrier(device_ids=[torch.xpu.current_device()])
    torch.xpu.synchronize()
    #x = torch.ones(4).to(device, non_blocking=True)
    #x = torch.ones([1024, 1024]).to(device, non_blocking=True)
    x = torch.ones([dim_size, dim_size], dtype=torch.bfloat16).to(device, non_blocking=True)

    dist.barrier(device_ids=[torch.xpu.current_device()])
    torch.xpu.synchronize()

    # print_rank_0(x)
    t5 = datetime.datetime.now()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.xpu.synchronize()
    t6 = datetime.datetime.now()

    elapsed = (t6 - t5).total_seconds()
    time_iters[i] = elapsed
    print_rank_0(f"[{dist_my_rank}] Iter-{i}: {elapsed:.8f}")
print_rank_0(
    f"Average time for all_reduce (exclude first iter): {np.mean(time_iters[1:]):.8f} "
)

dist.barrier(device_ids=[torch.xpu.current_device()])
dist.destroy_process_group()
