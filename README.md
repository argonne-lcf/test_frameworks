# Testing Frameworks

This repo is meant to be used for testing the frameworks software stacks on ALCF systems. It includes a set of simple pytorch examples to test whether the frameworks work well or not. 

The issues will be reported here: https://github.com/argonne-lcf/test_frameworks/issues. This will be used to keep track of all the issues. 

- test_dist: torch dist communication test
- test_dtensor: testing the fundamental unit
- test_resnet50: resnet50 with FSDP or DDP
- test_mnist: MNIST with DDP
- test_mpi4py: testing mpi4py

To run the test

* Torch Dist
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_torch_dist.py
  ```

* DTensor
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_dtensor.py --tp-size 8 --dim 96
  ```

* ResNet50
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_resnet50.py
  ```

* MNIST
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_mnist.py
  ```

* mpi4py
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_mpi4py.py
  ```

## How to contribute

If you have any tests that you think important to include, please send a PR. In the PR, please include 1) the code to run; 2) information about what aspects of the software / hardware the test is evaluating. 
