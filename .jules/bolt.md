## 2024-04-21 - HDF5 Context Manager Performance Bottleneck
**Learning:** Opening HDF5 files (`h5py.File`) repeatedly inside loops is a recognized performance bottleneck in the codebase, leading to unnecessary I/O overhead.
**Action:** Move the file context manager (`with h5py.File(...) as f:`) outside the loop when writing or appending to HDF5 files to optimize I/O performance.
