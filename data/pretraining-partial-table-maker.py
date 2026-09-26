import ast
import os
import shutil
import time

# from gaiaxpy import generate, PhotometricSystem
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.table import Table
from dustmaps.config import config as dustmaps_config
from dustmaps.sfd import SFDQuery
from tqdm import tqdm

dustmaps_data_dir = os.environ.get("DUSTMAPS_DATA_DIR")
if not dustmaps_data_dir:
    raise RuntimeError(
        "Set DUSTMAPS_DATA_DIR to a persistent directory containing the SFD maps"
    )
dustmaps_config["data_dir"] = os.path.expandvars(os.path.expanduser(dustmaps_data_dir))
import gc

import astropy.units as units
from astropy.coordinates import SkyCoord

GAIA_XP_DIR = (
    Path(
        os.environ.get(
            "MSA_GAIA_XP_DIR",
            "/arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_mean_spectrum",
        )
    )
    .expanduser()
    .resolve()
)
GAIA_SOURCE_DIR = (
    Path(os.environ.get("MSA_GAIA_SOURCE_DIR", "/arc/projects/k-pop/gaia/GaiaSource"))
    .expanduser()
    .resolve()
)
ADQL_MATCH_DIR = (
    Path(
        os.environ.get(
            "MSA_ADQL_MATCH_DIR", "/arc/projects/k-pop/catalogues/adql_matches"
        )
    )
    .expanduser()
    .resolve()
)
PREPROCESS_CACHE_DIR = Path(
    os.environ.get("MSA_PREPROCESS_CACHE_DIR", ".crossmatch-cache")
).expanduser()
PREPROCESS_WORKERS = int(os.environ.get("MSA_PREPROCESS_WORKERS", "4"))
if PREPROCESS_WORKERS < 1:
    raise ValueError("MSA_PREPROCESS_WORKERS must be positive")


def flux_error_to_mag(flux, flux_error):
    """Propagate a positive flux and its error to a symmetric magnitude sigma."""
    flux = np.asarray(flux, dtype=np.float64)
    flux_error = np.asarray(flux_error, dtype=np.float64)
    result = np.full(np.broadcast_shapes(flux.shape, flux_error.shape), np.nan)
    valid = np.isfinite(flux) & (flux > 0) & np.isfinite(flux_error) & (flux_error >= 0)
    # ponytail: first-order propagation is appropriate for symmetric training weights;
    # asymmetric low-S/N Gaia magnitude errors need a flux-space likelihood to improve.
    np.divide(2.5 / np.log(10) * flux_error, flux, out=result, where=valid)
    return result


def _xp_scale_labels():
    return [
        f"{prefix}_{i}" for prefix in ("bp", "rp", "bpe", "rpe") for i in range(1, 56)
    ]


def _scale_xp_measurements(df):
    factor = 10 ** ((8.5 - df["G"]) / 2.5)
    for column in _xp_scale_labels():
        df[column] = df[column] / factor
    return df


# Function to process a single file
def process_xp_file(args):
    ids, source_name = args
    if len(ids) == 0:
        return None

    file_path = (
        GAIA_XP_DIR / f"XpContinuousMeanSpectrum_{source_name.split('_')[-1]}.csv.gz"
    )

    # Use chunking to read the file in parts and only process necessary rows
    chunksize = 100000
    filtered_chunks = []

    for chunk in pd.read_csv(
        file_path,
        compression="gzip",
        comment="#",
        usecols=[
            "source_id",
            "bp_coefficients",
            "bp_coefficient_errors",
            "rp_coefficients",
            "rp_coefficient_errors",
        ],
        chunksize=chunksize,
    ):
        filtered_chunk = chunk[chunk["source_id"].isin(ids)]
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)

    if filtered_chunks:
        return pd.concat(filtered_chunks, ignore_index=True)

    else:
        return None


def process_source_file(args):
    ids, source_name = args
    if len(ids) == 0:
        return None

    file_path = GAIA_SOURCE_DIR / f"{source_name}.hdf5"

    # Open the HDF5 file using h5py
    with h5py.File(file_path, "r") as f:
        # Assuming the datasets are stored as individual arrays
        source_id = f["source_id"][:]
        phot_g_mean_mag = f["phot_g_mean_mag"][:]
        phot_bp_mean_mag = f["phot_bp_mean_mag"][:]
        phot_rp_mean_mag = f["phot_rp_mean_mag"][:]
        phot_g_mean_flux = f["phot_g_mean_flux"][:]
        phot_bp_mean_flux = f["phot_bp_mean_flux"][:]
        phot_rp_mean_flux = f["phot_rp_mean_flux"][:]
        phot_g_mean_flux_error = f["phot_g_mean_flux_error"][:]
        phot_bp_mean_flux_error = f["phot_bp_mean_flux_error"][:]
        phot_rp_mean_flux_error = f["phot_rp_mean_flux_error"][:]
        parallax = f["parallax"][:]
        parallax_error = f["parallax_error"][:]
        pmra = f["pmra"][:]
        pmdec = f["pmdec"][:]
        pmra_error = f["pmra_error"][:]
        pmdec_error = f["pmdec_error"][:]
        ra = f["ra"][:]
        dec = f["dec"][:]

    # getting galactic extinction
    coords = SkyCoord(ra * units.deg, dec * units.deg, frame="icrs")
    sfd = SFDQuery()
    ebv = sfd(coords)

    # Convert the datasets to a pandas DataFrame
    df = pd.DataFrame(
        {
            "source_id": source_id,
            "G": phot_g_mean_mag,
            "BP": phot_bp_mean_mag,
            "RP": phot_rp_mean_mag,
            "e_G": flux_error_to_mag(phot_g_mean_flux, phot_g_mean_flux_error),
            "e_BP": flux_error_to_mag(phot_bp_mean_flux, phot_bp_mean_flux_error),
            "e_RP": flux_error_to_mag(phot_rp_mean_flux, phot_rp_mean_flux_error),
            "PARALLAX": parallax,
            "e_parallax": parallax_error,
            "pmra": pmra,
            "e_pmra": pmra_error,
            "pmdec": pmdec,
            "e_pmdec": pmdec_error,
            "RA": ra,
            "DEC": dec,
            "EBV": ebv,
        }
    )

    # Filter the DataFrame based on the IDs
    filtered_df = df[df["source_id"].isin(ids)]

    if not filtered_df.empty:
        return filtered_df
    else:
        return None


def extract_xp_coeffs(val):
    if isinstance(val, str):
        try:
            evaluated_val = ast.literal_eval(val)
            if isinstance(evaluated_val, list):
                return [float(x) for x in evaluated_val]
        except (ValueError, SyntaxError):
            pass
    return None


def process_xp_coeffs(column):
    column = np.array(column, dtype=object)
    processed_column = [extract_xp_coeffs(val) for val in column]
    expected_length = next(
        (len(lst) for lst in processed_column if lst is not None), 55
    )
    processed_column = [
        lst if lst is not None else [np.nan] * expected_length
        for lst in processed_column
    ]
    return processed_column


def _source_ids_with_xp(dataset):
    """Select source IDs with XP data from either supported index layout."""
    values = dataset[:]
    if values.dtype.names:
        try:
            source_ids = values["source_id"]
            has_xp = values["has_xp_coeffs"]
        except ValueError as exc:
            raise ValueError(
                "Source index datasets must have source_id and has_xp_coeffs fields"
            ) from exc
    elif values.ndim == 2 and values.shape[1] >= 2:
        source_ids, has_xp = values[:, 0], values[:, 1]
    else:
        raise ValueError(
            "Source index datasets must be structured records or two-column arrays"
        )
    return source_ids[has_xp == 1].tolist()


def chunk_crossmatch_np(chunk, local_source_ids, on_index, save_path, chunk_index):
    """Crossmatch chunk using NumPy and save results."""

    # Assuming the source_ids are in the second element of each tuple in chunk, which is a list
    chunk_source_ids = np.array([int(entry[1][0]) for entry in chunk], dtype=np.int64)
    # Extract the first element of the source_ids list

    # Create mask to check if the source_id in chunk is in local_source_ids
    mask = np.isin(chunk_source_ids, local_source_ids)

    # Filter matched entries
    matched = [
        entry for i, entry in enumerate(chunk) if mask[i]
    ]  # Filter based on mask

    if matched:
        save_file = os.path.join(save_path, f"chunk_{chunk_index}.npy")
        np.save(save_file, matched)  # Save the matched chunk

    del matched, mask
    gc.collect()
    return chunk_index


def read_hdf_chunked_np(hdf_path, dataset_name, chunk_size):
    """Yield HDF5 chunks as NumPy arrays."""
    with h5py.File(hdf_path, "r") as f:
        dataset = f[dataset_name]["table"]
        total_rows = dataset.shape[0]

        for start in range(0, total_rows, chunk_size):
            stop = min(start + chunk_size, total_rows)
            chunk = dataset[start:stop]
            yield chunk
            del chunk
            gc.collect()


def parallel_crossmatch_np(
    hdf_path,
    dataset_name,
    local_source_ids,
    on_index=0,
    chunk_size=5_000_000,
    workers=PREPROCESS_WORKERS,
    save_path="crossmatch_output",
):
    """Run parallel crossmatch and save intermediate results."""
    os.makedirs(save_path, exist_ok=True)
    chunk_index = 0
    start_time = time.time()
    # ponytail: cap queued HDF chunks at 2 per worker; tune only after measuring
    # session memory and throughput on the mounted catalogues.
    max_in_flight = workers * 2
    pending = set()

    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        tqdm(desc=f"Crossmatching {dataset_name}", unit="chunk") as progress,
    ):
        for chunk in read_hdf_chunked_np(hdf_path, dataset_name, chunk_size):
            pending.add(
                executor.submit(
                    chunk_crossmatch_np,
                    chunk,
                    local_source_ids,
                    on_index,
                    save_path,
                    chunk_index,
                )
            )
            chunk_index += 1
            if len(pending) >= max_in_flight:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()
                progress.update(len(completed))
                gc.collect()

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
            progress.update(len(completed))
            gc.collect()

    elapsed_time = time.time() - start_time
    print(f"{dataset_name} crossmatching completed in {elapsed_time:.2f} seconds")


def merge_results_np(save_path):
    """Merge all saved NumPy arrays into one, ensuring correct handling of tuple structures."""
    files = sorted(os.listdir(save_path))
    all_source_ids = []
    all_floats = []

    starttime = time.time()

    for file in files:
        chunk = np.load(
            os.path.join(save_path, file), allow_pickle=True
        )  # Important: allow_pickle=True for tuples

        # Flatten the data into the desired format

        source_ids = np.array([entry[1][0] for entry in chunk], dtype=np.int64)
        floats = np.array([entry[2] for entry in chunk], dtype=np.float64)

        all_source_ids.append(source_ids)
        all_floats.append(floats)

    # If all chunks are found, concatenate them
    if all_source_ids:
        all_source_ids = np.concatenate(all_source_ids)
        all_floats = np.vstack(all_floats)  # Stacking float arrays along the first axis
        print("Time to merge results:", time.time() - starttime)
        return all_source_ids, all_floats
    else:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)


def crossmatch_all(df_local):
    """Crossmatch all tables and merge results."""
    local_source_ids = df_local["source_id"].values.astype(np.int64)

    tables = {
        "smssdr4_curated.h5": "catalog",
        "sdss_curated.h5": "catalog",
        "ps1_curated.h5": "catalog",
        "tmass_curated.h5": "catalog",
    }

    merged_df = df_local.copy()

    for table, key in tables.items():
        save_path = PREPROCESS_CACHE_DIR / f"{table}_output"
        shutil.rmtree(save_path, ignore_errors=True)
        parallel_crossmatch_np(
            str(ADQL_MATCH_DIR / table),
            key,
            local_source_ids,
            on_index=0,
            chunk_size=5_000_000,
            save_path=str(save_path),
        )

        result_np = merge_results_np(str(save_path))

        if table == "sdss_curated.h5":
            cols = [
                "U_SDSS",
                "G_SDSS",
                "R_SDSS",
                "I_SDSS",
                "Z_SDSS",
                "E_U_SDSS",
                "E_G_SDSS",
                "E_R_SDSS",
                "E_I_SDSS",
                "E_Z_SDSS",
            ]
        elif table == "smssdr4_curated.h5":
            cols = [
                "U_SMSS",
                "E_U_SMSS",
                "V_SMSS",
                "E_V_SMSS",
                "G_SMSS",
                "E_G_SMSS",
                "R_SMSS",
                "E_R_SMSS",
                "I_SMSS",
                "E_I_SMSS",
                "Z_SMSS",
                "E_Z_SMSS",
            ]
        elif table == "tmass_curated.h5":
            cols = ["J", "E_J", "H", "E_H", "KS", "E_KS"]
        elif table == "ps1_curated.h5":
            cols = [
                "G_PS1",
                "E_G_PS1",
                "R_PS1",
                "E_R_PS1",
                "I_PS1",
                "E_I_PS1",
                "Z_PS1",
                "E_Z_PS1",
                "Y_PS1",
                "E_Y_PS1",
            ]

        if result_np[0].size > 0:
            result_df = pd.DataFrame(result_np[1], columns=cols)
            result_df.insert(0, "source_id", result_np[0].astype(np.int64))
        else:
            result_df = pd.DataFrame(columns=["source_id"] + cols)

        merged_df = pd.merge(merged_df, result_df, on="source_id", how="left")
        print(f"Merged {table} with {len(result_df)} matches")

    print(merged_df["source_id"].dtype)

    for table in (
        "sdss_curated.h5",
        "smssdr4_curated.h5",
        "tmass_curated.h5",
        "ps1_curated.h5",
    ):
        shutil.rmtree(PREPROCESS_CACHE_DIR / f"{table}_output", ignore_errors=True)

    return merged_df


def main():
    global PREPROCESS_CACHE_DIR
    catwise_path = (
        Path(os.environ.get("MSA_CATWISE_FILE", "table_1_catwise.fits.gz"))
        .expanduser()
        .resolve()
    )
    source_ids_path = (
        Path(os.environ.get("MSA_SOURCE_IDS_FILE", "source_ids_x_file_names.h5"))
        .expanduser()
        .resolve()
    )
    output_dir = Path(os.environ.get("MSA_PREPROCESS_DIR", ".")).expanduser().resolve()
    PREPROCESS_CACHE_DIR = (
        Path(
            os.environ.get("MSA_PREPROCESS_CACHE_DIR", output_dir / ".crossmatch-cache")
        )
        .expanduser()
        .resolve()
    )
    required = [
        catwise_path,
        source_ids_path,
        GAIA_XP_DIR,
        GAIA_SOURCE_DIR,
        ADQL_MATCH_DIR,
        *(
            ADQL_MATCH_DIR / name
            for name in (
                "smssdr4_curated.h5",
                "sdss_curated.h5",
                "ps1_curated.h5",
                "tmass_curated.h5",
            )
        ),
        Path(dustmaps_data_dir).expanduser().resolve(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Preprocessing inputs or mounts are missing: " + ", ".join(missing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    PREPROCESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    catwisexmatch = Table.read(catwise_path)
    catwisexmatch = catwisexmatch.to_pandas()
    catwisexmatch["W1"] = catwisexmatch["catwise_w1"]
    catwisexmatch["W2"] = catwisexmatch["catwise_w2"]
    catwisexmatch = catwisexmatch.drop(
        columns=[
            "in_training_sample",
            "teff_xgboost",
            "logg_xgboost",
            "mh_xgboost",
            "catwise_w1",
            "catwise_w2",
        ]
    )

    labels = [
        "source_id",
        "G",
        "BP",
        "RP",
        "e_G",
        "e_BP",
        "e_RP",
        "PARALLAX",
        "e_parallax",
        "pmra",
        "e_pmra",
        "pmdec",
        "e_pmdec",
        "RA",
        "DEC",
        "EBV",
        "W1",
        "W2",
        "U_SMSS",
        "E_U_SMSS",
        "V_SMSS",
        "E_V_SMSS",
        "G_SMSS",
        "E_G_SMSS",
        "R_SMSS",
        "E_R_SMSS",
        "I_SMSS",
        "E_I_SMSS",
        "Z_SMSS",
        "E_Z_SMSS",
        "U_SDSS",
        "G_SDSS",
        "R_SDSS",
        "I_SDSS",
        "Z_SDSS",
        "E_U_SDSS",
        "E_G_SDSS",
        "E_R_SDSS",
        "E_I_SDSS",
        "E_Z_SDSS",
        "G_PS1",
        "E_G_PS1",
        "R_PS1",
        "E_R_PS1",
        "I_PS1",
        "E_I_PS1",
        "Z_PS1",
        "E_Z_PS1",
        "Y_PS1",
        "E_Y_PS1",
        "J",
        "E_J",
        "H",
        "E_H",
        "KS",
        "E_KS",
    ]
    labels.extend(["bp_" + str(i) for i in range(1, 56)])
    labels.extend(["rp_" + str(i) for i in range(1, 56)])
    labels.extend(["bpe_" + str(i) for i in range(1, 56)])
    labels.extend(["rpe_" + str(i) for i in range(1, 56)])
    new_order = ["source_id", "W1", "W2", "G", "BP", "RP", "e_G", "e_BP", "e_RP"]
    new_order.extend(["bp_" + str(i) for i in range(1, 56)])
    new_order.extend(["rp_" + str(i) for i in range(1, 56)])
    new_order.extend(["bpe_" + str(i) for i in range(1, 56)])
    new_order.extend(["rpe_" + str(i) for i in range(1, 56)])
    new_order.extend(
        [
            "U_SMSS",
            "RA",
            "DEC",
            "E_U_SMSS",
            "V_SMSS",
            "E_V_SMSS",
            "G_SMSS",
            "E_G_SMSS",
            "R_SMSS",
            "E_R_SMSS",
            "I_SMSS",
            "E_I_SMSS",
            "Z_SMSS",
            "E_Z_SMSS",
            "U_SDSS",
            "E_U_SDSS",
            "G_SDSS",
            "E_G_SDSS",
            "R_SDSS",
            "E_R_SDSS",
            "I_SDSS",
            "E_I_SDSS",
            "Z_SDSS",
            "E_Z_SDSS",
            "G_PS1",
            "E_G_PS1",
            "R_PS1",
            "E_R_PS1",
            "I_PS1",
            "E_I_PS1",
            "Z_PS1",
            "E_Z_PS1",
            "Y_PS1",
            "E_Y_PS1",
            "J",
            "E_J",
            "H",
            "E_H",
            "KS",
            "E_KS",
            "PARALLAX",
            "e_parallax",
            "pmra",
            "e_pmra",
            "pmdec",
            "e_pmdec",
            "EBV",
        ]
    )
    process_labels = ["bp_" + str(i) for i in range(1, 56)]
    process_labels.extend(["rp_" + str(i) for i in range(1, 56)])
    process_labels.extend(["bpe_" + str(i) for i in range(1, 56)])
    process_labels.extend(["rpe_" + str(i) for i in range(1, 56)])

    file = h5py.File(source_ids_path, "r")

    n_parts = 50
    start_part = int(os.environ.get("START_PART", "0"))
    stop_part = int(os.environ.get("STOP_PART", str(n_parts)))
    if not 0 <= start_part <= stop_part <= n_parts:
        raise ValueError(
            f"Require 0 <= START_PART <= STOP_PART <= {n_parts}; "
            f"got {start_part}, {stop_part}"
        )
    # Determine the size of each split
    keys = list(file.keys())
    n = len(keys) // n_parts
    # Split the keys into n_parts
    for i in range(start_part, stop_part):
        startingtime = time.time()
        start_idx = i * n
        if i == n_parts - 1:  # Handle the remainder keys in the last part
            keys_part = keys[start_idx:]
        else:
            keys_part = keys[start_idx : start_idx + n]

        dictionary = {}
        for key in keys_part:
            dictionary[key] = _source_ids_with_xp(file[key])

        print("WORKING ON PORTION " + str(i + 1) + "/" + str(n_parts) + "\n")

        print("getting the coefficients")
        results = []
        with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
            # Submit tasks to the executor
            futures = [
                executor.submit(process_xp_file, (dictionary[key], key))
                for key in dictionary
            ]
            # futures = [executor.submit(process_xp_file, args) for args in zip(idlist, setlist)]
            for future in tqdm(futures, total=len(dictionary), desc="Processing Files"):
                result = future.result()
                if result is not None:
                    results.append(result)

        print("getting the source ids")
        source_results = []
        with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
            # Submit tasks to the executor
            # futures = [executor.submit(process_source_file, args) for args in zip(idlist, setlist)]
            futures = [
                executor.submit(process_source_file, (dictionary[key], key))
                for key in dictionary
            ]
            for future in tqdm(futures, total=len(dictionary), desc="Processing Files"):
                result = future.result()
                if result is not None:
                    source_results.append(result)

        print("concatenating results")
        # Concatenate all the filtered dataframes
        if results and source_results:
            xp_df = pd.concat(results, ignore_index=True)
            si_df = pd.concat(source_results, ignore_index=True)
            xp_df = pd.merge(xp_df, si_df, on="source_id", how="inner")
        else:
            raise ValueError(
                f"No Gaia XP/source matches were found for portion {i}; "
                "check mounted input files and source IDs"
            )
        print("Final DataFrame shape:", xp_df.shape)

        del si_df
        del source_results
        del results
        # Keep partially measured sources; the model masks missing channels.
        xp_df = xp_df.dropna(subset=["source_id"])

        print("merging with catwise")
        catwise_x_xp_df = pd.merge(xp_df, catwisexmatch, on="source_id", how="left")

        print("crossmatching other phot catalogues")
        ssl_df = crossmatch_all(catwise_x_xp_df)
        print(ssl_df["source_id"].dtype)
        print(f"Final dataset shape: {ssl_df.shape}")

        print("extracting coefficients from strings")
        print("getting bps")
        bp_ssl = ssl_df["bp_coefficients"].values
        bp = process_xp_coeffs(bp_ssl)
        del bp_ssl

        print("getting rps")
        rp_ssl = ssl_df["rp_coefficients"].values
        rp = process_xp_coeffs(rp_ssl)
        del rp_ssl

        print("getting bpes")
        bpe_ssl = ssl_df["bp_coefficient_errors"].values
        bp_e = process_xp_coeffs(bpe_ssl)
        del bpe_ssl

        print("getting rpes")
        rpe_ssl = ssl_df["rp_coefficient_errors"].values
        rp_e = process_xp_coeffs(rpe_ssl)
        del rpe_ssl

        # ⚡ Bolt Optimization: Avoid giant intermediate NumPy array memory spike
        # Convert lists of lists to DataFrames individually and concat them.
        # This significantly reduces peak memory usage.

        bp_cols = process_labels[0:55]
        rp_cols = process_labels[55:110]
        bpe_cols = process_labels[110:165]
        rpe_cols = process_labels[165:220]

        # Use np.array to ensure correct reshaping
        bp_df = pd.DataFrame(data=np.array(bp).reshape(-1, 55), columns=bp_cols)
        rp_df = pd.DataFrame(data=np.array(rp).reshape(-1, 55), columns=rp_cols)
        bpe_df = pd.DataFrame(data=np.array(bp_e).reshape(-1, 55), columns=bpe_cols)
        rpe_df = pd.DataFrame(data=np.array(rp_e).reshape(-1, 55), columns=rpe_cols)

        bprp_df = pd.concat([bp_df, rp_df, bpe_df, rpe_df], axis=1)

        print("cleaning rows")
        del bp
        del rp
        del bp_e
        del rp_e
        gc.collect()

        print("creating df")
        ssl_df = pd.concat([ssl_df, bprp_df], axis=1)
        del bprp_df
        ssl_df = ssl_df.drop(
            columns=[
                "bp_coefficients",
                "rp_coefficients",
                "bp_coefficient_errors",
                "rp_coefficient_errors",
            ]
        )
        ssl_df = ssl_df[new_order]

        print("scaling XP coefficients and their uncertainties")
        ssl_df = _scale_xp_measurements(ssl_df)

        print("dropping weird columns and duplicates")
        ssl_df = ssl_df.drop_duplicates(subset="source_id")

        print("writing partial table")
        ssl_to_write = Table.from_pandas(ssl_df)
        ssl_to_write.write(output_dir / f"partialtable-{i}.fits", overwrite=True)

        print("time for part:", time.time() - startingtime, "s")


if __name__ == "__main__":
    main()
