## 2024-04-23 - HDF5 I/O Optimization in Data Processing Scripts
**Learning:** Opening `h5py.File` in append mode (`"a"`) repeatedly within a loop creates massive I/O bottlenecks when processing huge files (e.g. 220M stars dataset).
**Action:** Always move `h5py.File` context managers outside of loops when appending or modifying datasets in chunks.
