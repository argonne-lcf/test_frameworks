# Testing Frameworks

These are a set of simple pytorch examples to test whether the frameworks work well or not.

- test_dtensor: testing the fundamental unit
- test_resnet50: resnet50 with FSDP or DDP
- test_mnist: MNIST with DDP
- test_mpi4py: testing mpi4py
- test_torch_checkpoint.py: testing checkpoint write and read

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

* Checkpoint
  ```bash
  mpiexec -np 24 --ppn 12 --cpu-binding $CPU_BIND python3 ./test_torch_checkpoint.py --output-folder /tmp/
  ```

