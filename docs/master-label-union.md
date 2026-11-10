# Master spectroscopic label union

Goal: one long-format list of every public spectroscopic stellar label we can
tie to a Gaia DR3 `source_id`, with extra depth in the metal-poor tail. This is
a **source census**. It assigns no FIT/TUNE/CAL/DEV roles and enables no
training channel.

Code:

| Piece | Path |
|---|---|
| Adapters, Gaia identity, census | `src/masked_stellar_autoencoder/training/master_labels.py` |
| Runner (local or CANFAR) | `scripts/run_master_label_census.py` |
| CANFAR CPU launcher | `scripts/launch_master_label_census.py` |
| Tests | `tests/test_master_labels.py` |
| Latest census manifest | `docs/reports/master-label-census-20261110/manifest.json` |

## Data model

`observations.parquet` has one row per native catalogue row:
`source, tier, release, pipeline, dependency_group, native_row, native_id,
source_id, id_method, match_sep_arcsec, match_n_within, ra, dec,
position_epoch, teff, e_teff, logg, e_logg, fe_h, e_fe_h, fe_h_kind, alpha,
e_alpha, alpha_kind, snr, quality, fe_h_grid_edge`.

- `tier`: `survey`, `metal_poor_survey` or `highres_reference`. High-resolution
  literature samples are for calibration/QA reservation, not silent FIT labels.
- `dependency_group`: rows from the same spectra share it (for example LAMOST
  LASP, Li+2018 VMP and Li+2025 CaT are all `lamost_lrs_spectra`). They are not
  independent measurements.
- `fe_h_kind`: `fe_h`, `fe_h_cat` (Ca triplet) or `m_h_model`. It is never
  silently renamed.
- `fe_h_grid_edge`: the value sits on a model-grid floor and is censored, not
  measured. In DR19 Astra ASPCAP, `fe_h` piles up at −2.50.
- `id_method`: `publisher_dr3`, `publisher_edr3` (EDR3 IDs equal DR3 IDs),
  `positional`, `positional_ambiguous` (more than one Gaia source inside the
  radius) or `positional_unmatched`.

`stars.parquet` has one row per Gaia star: the sources present and the
quality, edge-free median [Fe/H] per tier.

## Gaia identity for catalogues without IDs

Candidates come from the ESA Gaia archive (`gaiadr3.gaia_source`) through an
upload join with the q3c index (`POINT(upload)` inside `CIRCLE(gaia)`),
batched and cached per batch. Only public catalogue positions are uploaded.
Gaia positions are propagated with their proper motions to the catalogue
epoch.

- SDSS/SSPP and APOGEE rows must keep the same nearest source at both 2000 and
  2008 within 1.5″.
- High-resolution literature samples use a 30″ candidate search, because halo
  subdwarfs move several arcsec between epochs, and a 3″ match radius.
- WISE, SkyMapper and Pristine positions use their own epochs.

## Sources

| Source | Tier | Identity | Notes |
|---|---|---|---|
| SDSS DR19 Astra ASPCAP 0.6.0 | survey | publisher DR3 | Supersedes DR17 allStar. DR20 has no APOGEE update. |
| SDSS DR12 SSPP (SEGUE + legacy) | survey | positional | Final SSPP; one row per BESTOBJID |
| Gaia-ESO DR5.1 | survey | positional | ESA upload previously approved |
| S5 DR1 | metal-poor survey | publisher EDR3 | rvspecfit percentiles |
| PIGS AAT | metal-poor survey | publisher DR3 | FERRE adopted; ULySS retained on ARC |
| LAMOST VMP (Li+2018) | metal-poor survey | positional | Same spectra as LASP |
| LAMOST CaT VMP (Li+2025) | metal-poor survey | publisher DR3 | CaT [Fe/H] only |
| Pristine INT (Aguado+2019) | metal-poor survey | positional | FERRE |
| SkyMapper EMP (Da Costa+2019 t2/t3) | metal-poor survey | positional | Spectrophotometric FITTER |
| Best & Brightest (Placco+2019) | metal-poor survey | positional | n-SSPP |
| RPA Hansen+2018, Holmbeck+2020 | high-res reference | positional | |
| Yong+2021 SkyMapper, Yong+2013, Roederer+2014 | high-res reference | positional | Roederer gives model [M/H] |
| PASTEL 2020 | high-res reference | positional | Several papers per star |

The DR17 APOGEE, DESI DR1, GALAH DR4, LAMOST DR12, Gaia GSP-Spec, RAVE DR6,
MaStar and BOSSNet products are already staged on ARC. The CANFAR run joins
every `native_observations/<source>` sidecar under the stellar-improvement root
(`--existing-root`) and records which file it used for each source.

Still to add: SAGA database and JINAbase compilations, H3 (if released),
Ezzeddine+2020 RPA (its parameters need a join on 2MASS ID), AMBRE, Gaia DR4
(expected Dec 2026) and 4MOST pDR0.

## Running

Locally (needs outbound access to data.sdss.org, VizieR and gea.esac.esa.int):

```bash
pixi run python scripts/run_master_label_census.py --work /path/cache --output /path/out
```

On CANFAR, with one of these set: `CANFAR_TOKEN`, `CANFAR_CERTIFICATE`, or
`CADC_PROXY_PEM` (PEM text):

```bash
pixi run -e monitor python scripts/launch_master_label_census.py --dry-run
pixi run -e monitor python scripts/launch_master_label_census.py
```
