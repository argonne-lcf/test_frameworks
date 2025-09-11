#!/usr/bin/env python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
print(f"I am {rank} of {size}")
