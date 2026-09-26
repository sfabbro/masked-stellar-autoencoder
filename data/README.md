# Data

The data fed to the model has a specific structure which is created using the following dataset making scripts. The pre-training dataset has the column format of source_id, CatWISE bands W1 and W2, *Gaia* G, BP, and RP, followed by the 55 BP coefficients, the 55 RP coefficients, magnitudes from any other photometric surveys included, and finally positional information not desired to be reconstructed by the model. To create the fine-tuning dataset, the labels are included between the source_ids and W1, having the format of label, label_error, repeating. More surveys can be included in the datasets given that the columns are updated in the [config](../configs) files.

---

## Pre-Training Dataset Creation Procedure

- source_ids_x_file_names.py to create the reference file for all the XP continuous and Gaia source files.
- photometric-dataset-handling.ipynb to clean the photometric datasets before combining if needed.
- smssdr4_filtering.ipynb cleaning specifically the Sky-Mapper DR4 dataset as it is in a different format.
- pretraining-partial-table-maker.py to make all the small partial tables for the pre-training dataset in its original form.
- combine-partial-tables.py to crunch all the files together into 1 HDF file.
- add-gaia-features.py example on how to add more features from the gaia source files if desired, like the proper motions.

The table maker processes portions 0–49 by default. Use the half-open range `START_PART=33 STOP_PART=34` to resume just portion 33 after an interrupted run, or to split preprocessing across sessions. It writes `partialtable-<portion>.fits`; the combiner creates HDF5 keys named `sslset<portion>_part<chunk>`, matching the `valid_keys` pattern in the training configs. The combiner rebuilds the output through a temporary file, so rerunning it replaces the prior complete HDF5 artifact only after conversion succeeds. Point `data.datafile` in the pretraining YAML at this output before training.

The current preprocessing joins Gaia source and XP rows, the precomputed CatWISE/XP table, and curated survey tables on Gaia `source_id`. It does not run a coordinate matcher or call an `xmatch` package. SkyMapper, SDSS, Pan-STARRS, and 2MASS inputs are already curated Gaia source-id matches; the CatWISE table is pre-matched in the Andrae+2023 product. To build new positional matches, use a separate catalogue-matching workflow and record its match-radius / ambiguity policy before feeding the resulting source IDs here.

Configured `error_cols` are per-feature measurement uncertainties in the same units as their values. XP coefficient errors are scaled by the same flux factor as the coefficients. Gaia G/BP/RP magnitude errors are propagated from the source flux and flux error with the first-order magnitude derivative; parallax and proper-motion errors use Gaia's published astrometric uncertainties. Survey magnitude errors come from their `E_*` columns. `W1`, `W2`, `EBV`, `RA`, and `DEC` have no uncertainty mapped by this builder: their aligned config entries are `null`, so training uses neutral scaled reconstruction weights and skips error-scaled noise for those channels. Rebuild the partial FITS/HDF5 and fine-tuning FITS after changing this schema.

Input and output paths are configurable with `MSA_SOURCE_IDS_FILE`,
`MSA_CATWISE_FILE`, `MSA_GAIA_XP_DIR`, `MSA_GAIA_SOURCE_DIR`,
`MSA_ADQL_MATCH_DIR`, `MSA_PREPROCESS_DIR`, and `MSA_PRETRAIN_HDF5_OUT`.
`DUSTMAPS_DATA_DIR` must point to a directory that already contains the SFD maps;
fetch them once with `dustmaps.sfd.fetch()` into persistent storage before
starting the builder. `MSA_PREPROCESS_WORKERS` caps local concurrency. On
CANFAR, put caches under `$WORK` and keep partial tables/HDF5 under `/arc/projects`.

- data_validator.py provides scripts to validate the data in the pre-training dataset file.

---

## Pre-Training Data

- CatWISE - table_1_catwise.fits comes from the Andrae+2023 data as it is a pre computed xmatch with XP spectra, work is done for us already.
    - [Documentation](https://irsa.ipac.caltech.edu/data/WISE/CatWISE/gator_docs/catwise_colDescriptions.html)
    - [Data](https://zenodo.org/records/7945154)
- *Gaia* DR3 - *Gaia* DR3 source and XP continuous data were provided as 3386 files sliced identically by source_id, such that some XP continuous files contained no information as no stars within a source_id range contained XP continuous measurements.
    - [Gaia Source Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
    - [Gaia Source Data](https://sdsc-users.flatironinstitute.org/~gaia/dr3/hdf5/GaiaSource/)
    - [XP Continuous Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_spectroscopic_tables/ssec_dm_xp_continuous_mean_spectrum.html)
    - [XP Continuous Data](https://cdn.gea.esac.esa.int/Gaia/gdr3/Spectroscopy/xp_continuous_mean_spectrum/)
- Sky-Mapper DR4 - Data was provided as numerous files containing pre-computed crossmatches with *Gaia* DR3 as a subset of columns within the DR4 files.
    - [Documentation](https://skymapper.anu.edu.au/data-release/)
    - [Data](https://skymapper.anu.edu.au/_data/DR4/)
- Pan-STARRS DR1 - Matched using source_id column via multiple asynchronous ADQL queries ([example](example_adql_match.py)) through the *Gaia* Archive, where pre computed crossmatches have already exist courtesy of the *Gaia* Data Processing and Analysis Consortium (DPAC).
    - [Documentation](https://outerspace.stsci.edu/display/PANSTARRS)
    - [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_crossmatch/sec_crossmatch_externalCat/ssec_crossmatch_panstarrs.html)
- 2MASS - Matched using source_id column via multiple asynchronous ADQL queries through the *Gaia* Archive.
    - [Documentation](https://irsa.ipac.caltech.edu/Missions/2mass.html)
    - [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_crossmatch/sec_crossmatch_externalCat/ssec_crossmatch_2mass.html)
- SDSS DR13 - Matched using source_id column via multiple asynchronous ADQL queries through the *Gaia* Archive.
    - [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_crossmatch/sec_crossmatch_externalCat/ssec_crossmatch_sdss.html)

topcat stilts was used to concatenate the numerous tables matched through adql and astroquery.gaia into singular HDF files for 2MASS, SDSS, and Pan-STARRS.

---

## Fine-Tuning Data

- sdss-apogee-dr17.fits         : Binary table from SDSS APOGEE APSCAP data release
    - [Documentation](https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/allStar.html)
- apogee_astroNN-DR17.fits      : Binary table from Leung, Bovy, & Mackereth last updated in 2021
    - [Accessible here](https://www.sdss.org/dr18/data_access/value-added-catalogs/?vac_id=85)
- nn_latent_age_dr17.csv.gz     : CSV table from Leung, Bovy, Mackereth & Miglio in 2023
    - [Accessible here](https://github.com/henrysky/astroNN_ages)
- galah_dr4_allstar_240705.fits : Binary table from GALAH 4th data release
    - [Documentation](https://www.galah-survey.org/dr4/the_catalogues/#galah-dr4-main-catalogues)
- li_et_al_x_gaiaids.fits       : Binary table of very metal-poor stars from LAMOST cross-matched with Gaia DR3
    - Matched with astroquery.gaia based on RA and Dec
    - [Paper](https://ui.adsabs.harvard.edu/abs/2022ApJ...931..147L/abstract)
