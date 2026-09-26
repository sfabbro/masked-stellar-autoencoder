import os
from pathlib import Path

import h5py
import tqdm
from astropy.io import fits

row_limit = 2_000_000  # Maximum number of rows per dataset
input_dir = Path(os.environ.get("MSA_PREPROCESS_DIR", ".")).expanduser()
filelist = sorted(input_dir.glob("partialtable*.fits"))
if not filelist:
    raise FileNotFoundError(
        f"No FITS files found matching {str(input_dir / 'partialtable*.fits')!r}"
    )

output_path = Path(
    os.environ.get(
        "MSA_PRETRAIN_HDF5_OUT", input_dir / "pretrain_dataset_incomplete.h5"
    )
).expanduser()
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = Path(str(output_path) + ".tmp")
progress_bar = tqdm.tqdm(filelist, total=len(filelist))
dataset_count = 0

# Rebuild through a temporary file so a failed conversion leaves the last complete
# HDF5 artifact usable for training.
with h5py.File(temporary_path, "w") as hf:
    for file in progress_bar:
        stem = file.stem
        prefix = "partialtable-"
        table_id = stem.removeprefix(prefix)
        if not stem.startswith(prefix) or not table_id.isdigit():
            raise ValueError(
                f"Expected a FITS filename like 'partialtable-11.fits', got {file!r}"
            )

        with fits.open(file, memmap=True) as hdul:
            if len(hdul) < 2:
                raise ValueError(f"{file} has no binary-table HDU")
            data = hdul[1].data
            if data is None or len(data) == 0:
                print(f"Warning: {file} contains no rows, skipping")
                continue

            dataset_base_name = f"sslset{table_id}"
            total_rows = data.shape[0]
            num_chunks = (total_rows + row_limit - 1) // row_limit

            for i in range(num_chunks):
                start_idx = i * row_limit
                end_idx = min(start_idx + row_limit, total_rows)
                dataset_name = f"{dataset_base_name}_part{i}"
                hf.create_dataset(dataset_name, data=data[start_idx:end_idx])
                dataset_count += 1

if dataset_count == 0:
    raise ValueError("No non-empty FITS tables were converted; keeping prior output")
os.replace(temporary_path, output_path)
