# Testing Frameworks

This repo is meant to be used for testing the frameworks software stacks on ALCF systems. It includes a set of simple pytorch examples to test whether the frameworks work well or not. 

The issues will be reported here: https://github.com/argonne-lcf/test_frameworks/issues. This will be used to keep track of all the issues. 

* Torch Dist test: Torch dist communication tests, including all the collective communicatino tests. 
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_torch_dist.py
  ```

* DTensor: This is testing the distributed matrix multiplication tests using DTensor. --dim is the total dimension of the global matrix, and --tp-size is the organization of the processor mesh (tp_size, world_size/tp_size)
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_dtensor.py --tp-size 8 --dim 96
  ```

* ResNet50: Resnet50 with FSDP or DDP
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_resnet50.py
  ```

* MNIST: MNIST with DDP
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_mnist.py
  ```

* mpi4py: Testing mpi4py
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_mpi4py.py
  ```

* Checkpoint
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_torch_checkpoint.py --output-folder /tmp/
  ```

## How to contribute

If you have any tests that you think important to include, please send a PR. In the PR, please include 1) the code to run; 2) information about what aspects of the software / hardware the test is evaluating. 
