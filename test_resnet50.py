import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import enable_wrap, size_based_auto_wrap_policy, wrap
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet50

from torch_setup import get_device, get_profiler_activities, init_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="DTensor + torch.profiler example.")
    # Tensor parallel (TP) size (how many ranks in our device mesh)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of ranks/devices to use in the device mesh.",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="dimension of the matrix"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--trace-dir", type=str, default="resnet50_trace")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()
# Initialize Process Group


# Define ResNet-50 Model
def create_model(device):
    model = resnet50(weights=False)
    model = model.to(device)
    if args.fsdp:
        model = FSDP(model)
    else:
        model = DDP(model)

    return model


# Training Function
def train(rank, world_size):
    device = get_device()
    print(f"Running on {device}")
    # Data Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load CIFAR-10 as an example dataset
    if rank == 0:
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, transform=transform, download=True
        )
    dist.barrier()
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=False
    ) 
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    # Model, Loss, Optimizer
    model = create_model(device)
    # model = ipex.optimize(model)  # Optimize with Intel Extension for PyTorch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    import time

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Ensure proper shuffling
        model.train()
        total_loss = 0
        i = 0
        start = time.time()
        steps = min(len(dataloader), args.steps)
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
            if i == steps - 1:
                continue
            i = i + 1
        dist.all_reduce(total_loss)
        end = time.time()
        if rank == 0:
            print(
                f"Epoch {epoch}, Total steps: {i}, Loss: {total_loss / len(dataloader):.4f}, Throughput: {i*args.batch_size/(end-start):.4f} images/second/gpu"
            )

    # Save model (only on rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), "resnet50_xpu_ddp.pth")

    dist.destroy_process_group()


# Main Execution
if __name__ == "__main__":
    dist, rank, world_size = init_distributed()
    if args.profile:
        with profile(
            activities=get_profiler_activities(),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            train(rank, world_size)
        os.makedirs(args.trace_dir, exist_ok=True)
        prof.export_chrome_trace(f"{args.trace_dir}/trace-{rank}-of-{world_size}.json")
    else:
        train(rank, world_size)
