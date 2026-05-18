## 2025-04-26 - HDF5 I/O in tight loops
**Learning:** Opening HDF5 files (`h5py.File`) inside tight loops is a significant I/O bottleneck in Python, particularly for scripts operating on many files or large datasets. Opening and closing these files repeatedly adds high overhead compared to opening the file once outside the loop and writing datasets as needed.
**Action:** When doing file I/O within a loop over datasets or chunks, verify if the target file (like an `.h5` file) can be opened globally using a context manager `with h5py.File(..., 'a') as f:` before entering the loop. This single-open approach avoids the constant file locking and opening/closing costs.

## 2025-04-26 - PyTorch Masking Overhead
**Learning:** In PyTorch, using `torch.where(mask, a, b)` for masking operations allocates a new tensor for the result and involves reading multiple tensors, adding unnecessary overhead in tight training loops or loss functions.
**Action:** When applying a mask to zero-out elements (or fill with a constant), prefer using the in-place operation `.masked_fill_(~mask, 0.0)`. This operates directly on an existing temporary tensor, reducing memory allocation overhead and significantly improving execution speed.

## 2025-05-19 - PyTorch Diagonal Masking Overhead
**Learning:** In PyTorch custom loss functions (like `RnCLoss`), repeatedly calling `.masked_select()` with a dynamically constructed diagonal mask (e.g., `(1 - torch.eye(n)).bool()`) causes significant memory allocations and processing overhead.
**Action:** When filtering diagonals or specific elements from multiple tensors sharing the same dimension, precompute the boolean mask once (e.g., `mask = ~torch.eye(n, dtype=torch.bool)`) and apply it using direct boolean indexing (`tensor[mask]`) instead of `.masked_select()`. This reduces intermediate tensor creations and improves performance.

## 2026-05-01 - Optimizing HDF5 Dataset Creation
**Learning:** Instantiating `pandas.DataFrame` purely as an intermediate step to construct structured arrays for HDF5 `create_dataset` incurs significant and unnecessary Pandas overhead.
**Action:** When assembling tabular data strictly for writing to HDF5 datasets, always use native NumPy structured arrays (`np.empty(len(data), dtype=[...])`) to drastically improve script execution speed and reduce memory consumption.
## 2024-05-18 - Optimize PyTorch masked loss reductions
**Learning:** PyTorch dynamic boolean indexing (e.g., `input[mask]`) prior to scalar reductions (`.sum()`, `.mean()`) triggers costly device-to-host syncs in PyTorch because the resulting tensor's size is data-dependent and must be communicated back to the CPU to allocate memory.
**Action:** Replace `masked_input = input[mask]` with `.masked_fill(~mask, 0.0)` to zero out non-target elements, maintaining full-shape arrays through the reduction (e.g., `masked_fill_tensor.sum() / mask.sum().clamp_min(1)`). This maintains math equivalency while completely removing the sync point. Note: Ensure NaNs are explicitly filled *before* mathematical operations like subtraction or exponentiation to prevent NaN propagation in gradients.
