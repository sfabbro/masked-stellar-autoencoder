import os
from pathlib import Path

import h5py
import numpy as np


def main():
    source_dir = Path(os.environ.get("MSA_GAIA_SOURCE_DIR", "gaia/GaiaSource"))
    output_path = Path(
        os.environ.get("MSA_SOURCE_IDS_FILE", "gaia/source_ids_x_file_names.h5")
    )
    source_files = sorted(source_dir.glob("*"))
    if not source_files:
        raise FileNotFoundError(f"No Gaia source files found in {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with h5py.File(temporary_path, "w") as hf_out:
            for source_file in source_files:
                with h5py.File(source_file, "r") as f:
                    ids = f["source_id"][:]
                    xpq = f["has_xp_continuous"][:]
                    filename = source_file.name.split(".")[0]

                    dtype = [("source_id", ids.dtype), ("has_xp_coeffs", xpq.dtype)]
                    dataset_to_save = np.empty(len(ids), dtype=dtype)
                    dataset_to_save["source_id"] = ids
                    dataset_to_save["has_xp_coeffs"] = xpq

                    hf_out.create_dataset(filename, data=dataset_to_save)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
