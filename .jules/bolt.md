## 2025-02-18 - Repeatedly Opening HDF5 files inside a loop is a performance bottleneck
**Learning:** Opening HDF5 files (e.g., using `h5py.File("...", "a")`) repeatedly inside loops, combined with pandas DataFrame overhead inside the loop, is a massive performance bottleneck.
**Action:** Move the `h5py.File` context manager outside the iteration. Also, use NumPy structured arrays instead of `pandas.DataFrame` when only creating intermediate structures to dump into an HDF5 dataset, avoiding large memory overheads.
