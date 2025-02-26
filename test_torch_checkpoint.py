#!/usr/bin/env python
import torch
from mpi4py import MPI
comm = MPI.COMM_WORLD
parser = argparse.ArgumentParser(
                    prog='Checkpoint',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--output-folder', type=str, default="/tmp/")  

args = parser.parse_args()
args.
a=torch.Tensor((100))
b=torch.Tensor((1048576))
c=torch.Tensor((1637385))

data = {
    "a": a,
    "b": b,
    "c": c
}
if comm.rank == 0:
    print(f"Saving checkpoint")
torch.save(data, f"{args.output_folder}/data-{comm.rank}-of-{comm.size}.pt")

if comm.rank == 0:
    print(f"Loading checkpoint")
torch.load(data, f"{args.output_folder}/data-{comm.rank}-of-{comm.size}.pt")
