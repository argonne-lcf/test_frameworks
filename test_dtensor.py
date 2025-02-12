import os
#import intel_extension_for_pytorch
#import oneccl_bindings_for_pytorch
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import time
import argparse

def init_distributed(backend="nccl"):
    """
    Initialize the default process group.
    """
    dist.init_process_group(
        backend=backend,
        init_method="env://",  # Read MASTER_ADDR, MASTER_PORT, etc. from environment
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"])
    )


def parse_args():
    parser = argparse.ArgumentParser(description="DTensor + torch.profiler example.")
    # Tensor parallel (TP) size (how many ranks in our device mesh)
    parser.add_argument("--tp-size", type=int, default=4, help="Number of ranks/devices to use in the device mesh.")
    parser.add_argument("--dim", type=int, default=256, help="dimension of the matrix")
    args = parser.parse_args()
    return args
    
def main():
    # 1. Initialize the distributed environment with NCCL for GPU usage
    args = parse_args()
    init_distributed(backend="gloo")

    # 2. Set local GPU device
    local_rank = int(os.environ["LOCAL_RANK"])
    def get_default_device():
        return torch.device(f"xpu:{local_rank}")
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 3. Build a 1D DeviceMesh across all GPUs
    #    This maps logical mesh coordinates [0..world_size-1] to CUDA devices [0..world_size-1].
    import numpy as np
    mesh = DeviceMesh("xpu", torch.tensor(np.arange(world_size).reshape(world_size//args.tp_size, args.tp_size).transpose()))
    print(mesh)
    # 4. Create local tensors on each rank (just random data plus an offset for demonstration)
    local_tensor_a = rank*torch.ones(args.dim, args.dim//args.tp_size).to(device)
    local_tensor_b = rank*torch.ones(args.dim//args.tp_size, args.dim).to(device)
    local_tensor_ap = 2*torch.ones(args.dim, args.dim//args.tp_size).to(device)
    local_tensor_bp = 2*torch.ones(args.dim//args.tp_size, args.dim).to(device)
    
    # 5. Wrap them as DTensors with Replicate placement
    dt_a = DTensor.from_local(
        local_tensor_a,
        device_mesh=mesh,
        placements=[Shard(1)]
    )
    dt_b = DTensor.from_local(
        local_tensor_b,
        device_mesh=mesh,
        placements=[Shard(0)]
    )
    print(dt_a.shape, local_tensor_a.shape)
    print(dt_b.shape, local_tensor_b.shape)
    dt_ap = DTensor.from_local(
        local_tensor_ap,
        device_mesh=mesh,
        placements=[Shard(1)]
    )
    dt_bp = DTensor.from_local(
        local_tensor_bp,
        device_mesh=mesh,
        placements=[Shard(0)]
    )

    
        # 7. Run a few steps of matmul in a loop so we can capture multiple profiler events
        
    def run():
        # Use record_function to label operations
        start = time.time()
        with record_function("dtensor_matmul_C"):
            dt_c = dt_a.matmul(dt_b)  # distributed matmul
            #dt_c = dt_c.redistribute(device_mesh = mesh, placements=[Shard(1)], async_op=True)
        with record_function("dtensor_matmul_CP"):                
            dt_cp = dt_ap.matmul(dt_bp)  # distributed matmul
            #dt_cp = dt_cp.redistribute(device_mesh = mesh, placements=[Shard(1)], async_op=True)
            # (Optional) we could do more computations here
        end = time.time()
        if rank == 0:
            print(f"step: {end-start:.6f} sec")
            # Step the profiler each iteration
        dist.barrier()
    run()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        for step in range(6):
            run()
    prof.export_chrome_trace(f"trace-{rank}-of-{world_size}.json")

    # 8. (Optional) convert the last result to local to see what is on this rank
    #local_c = dt_c.to_local()

    #print(f"[Rank {rank}] local A:\n{local_tensor_a}")
    #print(f"[Rank {rank}] local B:\n{local_tensor_b}")
    #print(f"[Rank {rank}] local C = A@B:\n{local_c}")
    print("-" * 70)

if __name__ == "__main__":
    main()
