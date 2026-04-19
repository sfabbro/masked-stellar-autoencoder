import numpy as np
import h5py
import glob
import pandas as pd

source_files = np.sort(glob.glob('gaia/GaiaSource/*'))
with h5py.File('gaia/source_ids_x_file_names.h5', 'a') as hf:
    for file in source_files:
        with h5py.File(file,'r') as f:
            ids = f['source_id'][:]
            xpq = f['has_xp_continuous'][:]
            filename = file.split('/')[-1].split('.')[0]
            data_dict = {'source_id': ids, 'has_xp_coeffs': xpq}
            dataset_to_save = pd.DataFrame(data_dict)
            hf.create_dataset(filename, data=dataset_to_save)
