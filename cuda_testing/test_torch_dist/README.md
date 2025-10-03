## Allreduce benhmarks

It seems there is an issue with the `test_torch_allreduce.py` in using 
`dist.barrier()`

Apparently, `dist.barrier()` called without devices ids etc. results in a clean
execution, but leads to a  UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.

A solution is to use `dist.barrier(device_ids=[torch.xpu.current_device()])`
But it leads to a hang for allreduce in the implementation of `test_torch_allreduce.py`

But `test_allreduce_nathan.py` leads to a correct execution and the warning 
disappears too.
