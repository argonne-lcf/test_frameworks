# Testing Frameworks

These are a set of simple pytorch examples to test whether the frameworks work well or not. This repo will be used to keep track of the issue of the frameworks module on ALCF systems. 

The issues will be reported here: https://github.com/argonne-lcf/test_frameworks/issues

- test_dtensor: testing the fundamental unit
- test_resnet50: resnet50 with FSDP or DDP
- test_mnist: MNIST with DDP
- test_mpi4py: testing mpi4py


To run the test

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



