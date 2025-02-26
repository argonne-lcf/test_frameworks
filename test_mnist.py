import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch_setup import get_device, get_device_type, init_distributed

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Distributed Training Function
def train(rank, world_size):
    """ Train the model using DDP """
    
    device = get_device()
    
    # Load Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root="/tmp/datascience/alcf_training/MNIST", train=True, transform=transform, download=True)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=train_sampler, num_workers=4)

    # Model
    model = SimpleCNN().to(device)
    model = DDP(model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # Number of epochs
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
        dist.all_reduce(total_loss)
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item() / len(train_loader)}")

    # Save model (only on rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), "/tmp/datascience/alcf_training/MNIST/mnist_ddp.pth")

    dist.destroy_process_group()

# Main Entry Point
if __name__ == "__main__":
    dist, rank, world_size = init_distributed()
    train(rank, world_size)
