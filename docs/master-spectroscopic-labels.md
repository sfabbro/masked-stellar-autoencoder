# Master spectroscopic label table

`master_spectroscopic_labels_20261111_v1` on ARC
(`/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/`), built by
`scripts/build_master_spectroscopic_labels.py`
(`training/master_spectroscopic.py`).

It is the union of every spectroscopic record we have staged:
- the eight staged surveys: LAMOST DR12 LRS and MRS, DESI DR1, APOGEE DR17,
  GALAH DR4, Gaia DR3 GSP-Spec, SDSS DR20 BOSSNet and RAVE DR6;
- the 20 master-label census sources: SDSS DR19 Astra ASPCAP, SSPP,
  Gaia-ESO, S5, PIGS, LAMOST VMP and CaT, Pristine, SkyMapper, Best &
  Brightest, SAGA, JINAbase, PASTEL and the high-res papers.

There is one record per native catalogue row, with no quality cut. Records
without a Gaia DR3 identity are kept.

**v1** (`master_spectroscopic_labels_20261111_v1`, complete):

| | Count |
|---|--:|
| Records | 29,307,497 |
| Stars | 19,138,656 |
| Stars with Gaia DR3 identity | 19,093,456 |
| Stars without Gaia identity | 45,200 |
| Records without Gaia identity | 57,983 |

Most records without Gaia identity are LAMOST LRS/MRS rows with no Gaia
source within 1.5″ (35,262 + 5,242). The APOGEE rows (5,564) have no Gaia
source within 3″; most are cool giants toward the Galactic centre. The 1,030
RAVE rows have no staged position. Census sources contribute 10,846 records
without identity.

**v3** (`master_spectroscopic_labels_20261111_v3`; v2 had a wrong Bensby+2014 position epoch) adds:

- **Census v4** (`master_label_census_20261111_v4`):
  - AMBRE FEROS (Worley+2012);
  - RPA: Ezzeddine+2020 and Sakari+2018;
  - Kirby+2010 dSph DEIMOS abundances and its high-res literature table;
  - LAMOST-Subaru (Li+2022);
  - SEGUE EMP follow-up (Aoki+2013);
  - Bensby+2014, with Hipparcos positions joined on HIP;
  - MaStar parameter VAC v2;
  - the DR19 Astra APOGEENet, AstroNN and The Payne summaries;
  - the DR20 Astra SLAM summary.
- **Other pipelines on staged spectra:** DESI SP (FERRE, `SPTAB`) and LAMOST
  MRS CNN.

Not available, so not included:

- H3: no public catalogue; the survey site lists only papers.
- 4MOST: no public release yet.
- Gaia DR4: expected December 2026.
- SAGA Local Group and LAMOST-Subaru retrievals: the LG form returns a
  different layout and LAMSUB returns a server error. Dwarf galaxies are
  covered by Kirby+2010; LAMOST-Subaru is covered by Li+2022 from VizieR.
- AMBRE HARPS and UVES: no table found on the VizieR mirror.
- MaStar's spectral-library parameters are in v2; its raw Gaia identity file
  is not needed.

## Files

| File | Content |
|---|---|
| `records/<survey>.parquet` | One row per native record (schema below). Read all surveys with `pd.read_parquet(dir / "records")`. |
| `stars.parquet` | One row per star: `star_id`, `star_key`, `source_id`, `n_records`, `surveys`, `n_surveys`, `has_highres`, `fe_h_median`, `fe_h_min`, `teff_median`, `logg_median`, `ra`, `dec`, `label_groups`, `v2_role`, `has_xp`. |
| `catalogues.json` | For each `catalogue` key: the native file on ARC, HDU, release, URL, sha256 and native ID definition. |
| `manifest.json` | Per-survey counts, identity methods, metal-poor star counts and output hashes. |
| `work/` | Intermediate files: per-survey staging and cached Gaia positional candidates. |

## Record schema

| Column | Meaning |
|---|---|
| `survey`, `release`, `pipeline`, `tier`, `dependency_group` | Origin and pipeline. `tier` is `survey`, `metal_poor_survey` or `highres_reference`. |
| `catalogue`, `native_hdu`, `native_row`, `native_id` | Trace-back keys. `native_row` is the 0-based row in the catalogue's HDU (or CSV, or census adapter frame). `native_id` is the survey's own key (LAMOST `obsid`/`mobsid`, DESI `TARGETID`, APOGEE `APOGEE_ID:TELESCOPE:FIELD`, GALAH `sobject_id`, BOSSNet `sdss_id:spectrum_pk`, RAVE `rave_obs_id`, ...). |
| `source_id`, `id_method`, `match_sep_arcsec`, `match_n_within` | Gaia DR3 identity, 0 if none. `id_method` is one of the values listed below the table. |
| `star_id` | Equals `source_id` when known. Otherwise a negative id shared by records within 1″ of each other in any survey, or unique to the record when it has no position. |
| `ra`, `dec` | Native position, degrees. |
| `teff`, `logg`, `fe_h`, `alpha` (+ `e_*`) | Native values. `fe_h_kind` is `fe_h` or `m_h` (GSP-Spec). `alpha_kind` is `alpha_m`, `alpha_fe`, `mg_fe` or `alpha_m_atm`. |
| `fe_h_corrected` | DESI only: the published [Fe/H] correction, inside its validity box. |
| `snr` | The survey's S/N column (LAMOST `snrg`, DESI `SN_R`, GALAH `snr_px_ccd3`, ...). |
| `fit_ok` | The survey's own failed-fit flags pass (see the table below). It is not a quality cut. |
| `flag_int`, `flag_text` | The main native flag: APOGEE `ASPCAPFLAG`, BOSSNet `result_flags`, GALAH `flag_sp`, DESI `RVS_WARN`/`RR_SPECTYPE`, GSP-Spec `flags_gspspec`, RAVE `algo_conv_madera`. |
| `v2_role` | Star-level role in the v2 training source (`expanded_six_source_20261111_v7`): `fit`, `tune`, `cal`, `devtest`, `external`, `excluded`, or empty when the star is not in it. |
| `label_groups` | Star-level groups holding this star *for this survey*: `v2:<group>` (training run v2), `loose:<group>` (looser cuts), `chem:<group>` (declared [M/H]/[α/M]/[Mg/Fe]). |
| `has_xp`, `has_xp_source` | XP present in the v2 training inputs; null for stars outside v2. |

`id_method` values:
- `publisher_dr3` / `publisher_edr3`: ID published with the catalogue (EDR3 and DR3 IDs are identical);
- `gaia_dr3`: the catalogue is Gaia itself (GSP-Spec);
- `simbad_dr3`: resolved through SIMBAD;
- `positional`: epoch-stable nearest Gaia DR3 neighbour within the survey's radius;
- `positional_ambiguous`: as `positional`, but with more than one Gaia candidate inside the radius;
- `positional_unmatched`: positional search found no stable match;
- empty: no ID and no position.

| Survey | `fit_ok` definition |
|---|---|
| LAMOST LRS/MRS | finite Teff (LASP converged) |
| DESI DR1 | `SUCCESS` & `PRIMARY` & `RVS_WARN == 0` & `RR_SPECTYPE == STAR` |
| APOGEE DR17 | `ASPCAPFLAG` STAR_BAD unset |
| GALAH DR4 | `flag_sp == 0` & `flag_sp_fit == 0` |
| GSP-Spec | finite Teff (flags kept in `flag_text`) |
| BOSSNet | `flag_bad` false |
| RAVE DR6 | `algo_conv_madera == 0` |
| census sources | census quality policy (`manifest.json` of `master_label_census_20261110_v3`) |

## Trace back to the native row

```python
import pandas as pd
from masked_stellar_autoencoder.training.master_spectroscopic import read_native_rows

root = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/master_spectroscopic_labels_20261111_v1/"
rec = pd.read_parquet(root + "records")
star = rec[rec.star_id == 4185355646116022528]
native = read_native_rows(star, root + "catalogues.json")
native["galah_dr4"][["eu_fe", "e_eu_fe", "flag_eu_fe"]]  # every native column
```

`read_native_rows` returns `{catalogue: DataFrame indexed by native_row}`
with every native column. For DESI it also returns the row-aligned `GAIA`,
`FIBERMAP` and `SPTAB` columns; `SPTAB` holds the FERRE abundances. For
VizieR, SAGA and JINAbase it re-reads the cached downloads with the census
adapters, so `native_row` refers to that parsed table.

## Caveats

- Group membership and `v2_role` are star-level. A star with several records
  in one survey shows the group on each of them, not only on the record that
  training used.
- Staged surveys are matched positionally only when no publisher ID exists:
  LAMOST, APOGEE, GALAH and BOSSNet. RAVE has no positions in the staged
  files, so RAVE records without the EDR3 crossmatch stay without Gaia
  identity.
- Pipelines run on the same spectra (APOGEE ASPCAP/APOGEENet/AstroNN/The
  Payne; DESI RVSpecFit/SP; LAMOST LASP/CNN) are separate surveys in the
  table that share a `dependency_group`. They are not independent
  measurements.
