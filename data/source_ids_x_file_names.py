import glob

import h5py
import numpy as np

source_files = np.sort(glob.glob("gaia/GaiaSource/*"))

with h5py.File("gaia/source_ids_x_file_names.h5", "a") as hf:
    for file in source_files:
        with h5py.File(file, "r") as f:
            ids = f["source_id"][:]
            xpq = f["has_xp_continuous"][:]
            filename = file.split("/")[-1].split(".")[0]

            # create structured array directly to avoid pandas DataFrame overhead
            dt = np.dtype([('source_id', ids.dtype), ('has_xp_coeffs', xpq.dtype)])
            dataset_to_save = np.empty(len(ids), dtype=dt)
            dataset_to_save['source_id'] = ids
            dataset_to_save['has_xp_coeffs'] = xpq

            hf.create_dataset(filename, data=dataset_to_save)
