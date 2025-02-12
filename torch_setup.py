import socket
from mpi4py import MPI
comm = MPI.COMM_WORLD
import torch.distributed as dist
import os
import torch

def get_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_avaiable():
        return "xpu"
    else:
        return cpu

def get_device(gpu=None):
    if gpu == None:
        gpu = get_device_type()
    os.environ['LOCAL_RANK'] = os.environ["PALS_LOCAL_RANKID"]
    local_rank = int(os.environ["LOCAL_RANK"])    
    return torch.device(f"{gpu}:{local_rank}")

def init_distributed(backend="nccl"):
    """
    Initialize the default process group.
    """

    if backend==None:
        gpu = get_device_type()    

        if gpu == "xpu":
            backend = "ccl"
        elif gpu == "cuda":
            backend = "nccl"
        else:
            backend = "mpi"
    if backend == "ccl":
        import intel_extension_for_pytorch
        import oneccl_bindings_for_pytorch
    os.environ['LOCAL_RANK'] = os.environ["PALS_LOCAL_RANKID"]
    os.environ['RANK'] = str(comm.rank)    
    rank = int(os.environ['RANK'])
    os.environ['WORLD_SIZE']= str(comm.size)
    world_size = comm.size
    local_rank = int(os.environ['LOCAL_RANK'])
    from mpi4py import MPI
    import socket
    master_addr = socket.gethostname()
    print(f"I am rank {rank} of {world_size} - {local_rank} on {master_addr}")    
    master_addr = comm.bcast(master_addr, root=0)
    print(master_addr)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "5676"
    dist.init_process_group(
        backend=backend,
        init_method="env://",  # Read MASTER_ADDR, MASTER_PORT, etc. from environment
        world_size=world_size, 
        rank=rank
    )
    return dist, rank, world_size
