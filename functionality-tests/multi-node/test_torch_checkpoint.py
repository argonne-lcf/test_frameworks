#!/usr/bin/env python
import torch
from mpi4py import MPI
import argparse
comm = MPI.COMM_WORLD
parser = argparse.ArgumentParser(
                    prog='Checkpoint',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--output-folder', type=str, default="/tmp/")
parser.add_argument('--dim', type=int, default=1048576)

args = parser.parse_args()
a=torch.ones((args.dim))
data=dict()
data = {
    "a": a,
}
if comm.rank == 0:
    print(f"Saving checkpoint")
with open(f"{args.output_folder}/data-{comm.rank}-of-{comm.size}.pt", "wb") as fout:
    torch.save(data, fout)

if comm.rank == 0:
    print(f"Loading checkpoint")
data = torch.load(f"{args.output_folder}/data-{comm.rank}-of-{comm.size}.pt")
