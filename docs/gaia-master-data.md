# Gaia DR3 master: data inventory and staging

This is the data-side runbook for the Gaia-anchored master
([xmatcher `ray-anchor` engine](https://github.com/astroai/xmatcher/blob/main/docs/ray-anchor.md)). It records the sources, sizes, local CANFAR
products, and the staging/validation order. The engine keeps every Gaia row;
the data plan must therefore budget for one row per Gaia source, not for the
intersection.

## Scope

- **Anchor:** Gaia DR3, one row per `source_id`.
- **Surveys:** SDSS DR16 imaging, Pan-STARRS1 DR2, CatWISE2020, VHS DR5,
  SkyMapper DR4, 2MASS PSC, AllWISE, GALEX GR6/7, S-PLUS (DR4 public or DR5
  local), UHS DR3, Pristine DR2, HSC-SSP PDR3, LSST DP2, and DESI Legacy DR11.
- **Match:** best counterpart per survey at 1.5″ by default, with
  `sep_second`, `n_within`, `crowded`, `pm_warn`, and `quality`.
- **Epoch:** propagate to J2016.0 where epoch and proper motion exist.

## Source inventory

Sizes are the catalogues' `hats_estsize`/published tree sizes; they change when
a provider rebuilds. `Rows` are from `properties`.

| Alias | Product | Rows | Tree size | Access | Notes |
|---|---|---:|---:|---|---|
| `gaia_lsdb_hats` | Gaia DR3 core (LSDB) | 1,812,731,847 | ~712 GiB | HTTPS | Preferred anchor; 153 cols, no XP coefficients |
| `gaia_hats` | Gaia DR3 core (CDS) | 1,811,709,771 | ~1.6 TB | CDS HTTPS | 225 cols; on-the-fly; no XP coefficients |
| `sdss_hats` | SDSS DR16 photometry | 1,231,051,050 | ~810 GB | CDS HTTPS | Static imaging; DR20 adds no imaging |
| `ps1_hats` | Pan-STARRS1 DR2 | 1,868,236,315 | ~460 GB | CDS HTTPS | 67 cols |
| `catwise_hats` | CatWISE2020 | 1,890,715,640 | ~600 GB | CDS HTTPS | Proper motions |
| `vhs_hats` | VHS DR5 | 1,374,207,485 | ~742 GB | CDS HTTPS | Per-band MJD; no PM |
| `skymapper_hats` | SkyMapper DR4 | 724,329,344 | ~96 GB | LSDB HTTPS | `mean_epoch`; no PM |
| `twomass_hats` | 2MASS PSC | 470,992,970 | ~41 GiB | LSDB HTTPS | J/H/K |
| `allwise_hats` | AllWISE | 747,634,026 | ~246 GiB | LSDB HTTPS | W1–W4, PM |
| `galex_hats` | GALEX GR6/7 | 82,992,086 | ~90 GB | CDS HTTPS | FUV/NUV |
| `splus_hats` | S-PLUS DR4 | 27,235,968 | ~8.8 GB | CDS HTTPS | Public DR4 |
| `splus5_hats` | S-PLUS DR5 | — | local | CANFAR | Set `XMATCHER_SPLUS_DR5_HATS` |
| `uhs_hats` | UHS DR3 | ~560 M | local | CANFAR/WSA | Set `XMATCHER_UHS_DR3_HATS` |
| `pristine_hats` | Pristine DR2 | — | local | CANFAR | Set `XMATCHER_PRISTINE_DR2_HATS` |
| `hsc_hats` | HSC-SSP PDR3 combined | — | local | CANFAR | Set `XMATCHER_HSC_PDR3_HATS` |
| `dp2` | LSST DP2 | 786,420,720 | ~3.6 TiB | RSP/NERSC/CANFAR | Set `XMATCHER_DP2_HATS`; stage only if allocation allows |
| `legacy_dr11` | LS DR11 tractor | ~1.5 B | TAP | Data Lab | No HATS; largest staging bottleneck |

The IVOA registry also provides `hats:skymapper_dr4`, `hats:two_mass`,
`hats:allwise`, and `hats:hsc_pdr3_*`; the aliases above pin the HTTP copy and
the intended release so a run does not silently follow a registry change.

## Quota model

Full mirrors of the nine public surveys plus Gaia are roughly **3.5–4 TB**
before the master. `/scratch` on a CANFAR session is typically ~200 GiB and is
session-local; it is not enough. Use a durable `/arc/projects/<group>` tree for
both the cache and the master.

Narrow staging cuts the stored footprint substantially:

| Stage | Approximate stored size |
|---|---:|
| Gaia core projection (33 cols) | ~100–150 GB |
| SDSS photometry projection (~15 cols) | ~200–300 GB |
| PS1, CatWISE, VHS, SkyMapper, 2MASS, AllWISE projections | ~400–600 GB |
| GALEX + S-PLUS | ~15–30 GB |
| Master with ~100 output columns | ~0.5–1.5 TB |

Plan for **3–5 TB free** on `/arc/projects` for a full run with local UHS,
S-PLUS DR5, HSC, Pristine, and DP2. If DP2 is included, the full 3.6 TiB tree
must be staged separately; otherwise use match-at-source on RSP/NERSC.

## Staging order

1. **Anchor first.** Mirror `gaia_lsdb_hats`, or project it directly:

   ```bash
   xmatcher project-hats gaia_lsdb_hats gaia_core.hats \
     --columns source_id,ra,dec,ref_epoch,parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,ruwe,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,bp_rp,radial_velocity,teff_gspphot,logg_gspphot,mh_gspphot,distance_gspphot,azero_gspphot,ebpminrp_gspphot,classprob_dsc_combmod_star,classprob_dsc_combmod_galaxy,classprob_dsc_combmod_quasar,has_xp_continuous,has_xp_sampled,has_rvs,has_epoch_photometry,phot_variable_flag,duplicated_source,l,b
   ```

   `project-hats` reads HTTPS HATS partitions directly and, on CDS, asks for
   at most eight columns per request (`?columns=…`), so it does not mirror the
   full tree first. For a multi-TB projection, add `--resume`: the
   `project.state` fingerprint binds the source, columns, and partition layout,
   and completed partitions are skipped after an interruption.

2. **Public surveys.** Project each survey to the columns the master needs.
   Examples:

   ```bash
   xmatcher project-hats sdss_hats sdss_narrow.hats \
     --columns objID,RA_ICRS,DE_ICRS,mode,class,clean,upmag,gpmag,rpmag,ipmag,zpmag,e_upmag,e_gpmag,e_rpmag,e_ipmag,e_zpmag,uMJD,gMJD,rMJD,iMJD,zMJD,Q

   xmatcher project-hats ps1_hats ps1_narrow.hats \
     --columns objID,RAJ2000,DEJ2000,gmag,rmag,imag,zmag,ymag,e_gmag,e_rmag,e_imag,e_zmag,e_ymag,Qual,Epoch

   xmatcher project-hats catwise_hats catwise_narrow.hats \
     --columns objID,RA_ICRS,DE_ICRS,W1mproPM,W2mproPM,e_W1mproPM,e_W2mproPM,pmRA,pmDE,e_pmRA,e_pmDE,MJD,pmQual
   ```

   The bundled `default_columns` for each alias is already the recommended
   projection; `xmatcher describe <alias>` prints it.

3. **Local CANFAR products.** Point the environment variables at the HATS
   roots and use the aliases unchanged:

   ```bash
   export XMATCHER_UHS_DR3_HATS=/arc/projects/<group>/uhs_dr3/hats
   export XMATCHER_SPLUS_DR5_HATS=/arc/projects/<group>/splus_dr5/hats
   export XMATCHER_PRISTINE_DR2_HATS=/arc/projects/<group>/pristine_dr2/hats
   export XMATCHER_HSC_PDR3_HATS=/arc/projects/<group>/hsc_pdr3/hats
   export XMATCHER_DP2_HATS=/arc/projects/<group>/dp2/hats
   ```

   If a variable is unset, the alias resolution fails with the variable named;
   no public fallback sky is substituted.

4. **Legacy DR11.** There is no HATS for LS DR11. The Data Lab TAP table
   `ls_dr11.tractor` is the photometry source; the precomputed
   `ls_dr11.x1p5__tractor__gaia_dr3__gaia_source` table (532,539,684 rows) is
   the fastest candidate set and validation reference. Staging the full TAP
   table is a separate, large job; do not start it until the HATS surveys have
   passed the pilot.

5. **XP sidecar.** Keep XP out of the master. Project the CANFAR XP copy to
   `source_id` plus the coefficient/error arrays and join on `source_id` later:

   ```bash
   xmatcher project-hats xp_hats gaia_xp_dr3.hats \
     --columns source_id,bp_coefficients,bp_coefficient_errors,rp_coefficients,rp_coefficient_errors
   ```

## Run

The wrapper selects the public survey list, adds local products only when their
environment variables are set, and runs the anchor engine:

```bash
export CACHE=/arc/projects/hats
export OUT=/arc/projects/hats/xmatcher/gaia_master.hats

PREFLIGHT_ONLY=1 bash scripts/run_gaia_master.sh
bash scripts/run_gaia_master.sh
```

Equivalent direct command:

```bash
xmatcher match gaia_lsdb_hats \
  twomass_hats allwise_hats catwise_hats skymapper_hats \
  sdss_hats ps1_hats vhs_hats galex_hats splus_hats \
  --engine ray-anchor -r 1.5 --target-epoch 2016.0 \
  --cache-root /arc/projects/hats \
  -o /arc/projects/hats/xmatcher/gaia_master.hats
```

The first run mirrors each input on demand. `--no-sync` requires the cache to
be complete. Resume is per anchor partition: rerunning the same output
directory skips partitions that already exist, but changing the inputs,
columns, radius, or epoch wipes the dataset first.

## Validate

Stream the output (only anchor ID and ambiguity columns are read):

```bash
python scripts/validate_anchor_output.py \
  /arc/projects/hats/xmatcher/gaia_master.hats \
  --json gaia_master_validation.json
```

The validator reports total rows, null IDs, duplicate IDs within each
partition, and per-survey matched/crowded/PM/quality counts. `hats_nrows` must
equal the partition sum; the validator's `expected-rows` option pins the Gaia
anchor count when supplied.

## Pilot order

Do not start with all surveys. The first pass should be:

1. `gaia_lsdb_hats` + `twomass_hats` + `catwise_hats` + `skymapper_hats` +
   `uhs_hats` (the smallest useful set).
2. `ps1_hats` + `sdss_hats`.
3. `vhs_hats` + `galex_hats` + `splus_hats`.
4. Local HSC/Pristine/DP2 and the LS DR11 candidate set.

After each pass, run the validator and compare a random sky sample against the
Data Lab precomputed 1.5″ Gaia crossmatches where available. A survey with
systematically low coverage or an implausible `n_within` distribution should be
inspected before the next pass; do not "fix" it by widening the radius.

## Known data caveats

- **Gaia XP:** neither Gaia HATS carries coefficients. The CDS `xpcont` tree
  is metadata-only; use the CANFAR XP copy or the ESA table, and keep it as a
  sidecar.
- **SDSS:** DR20 is spectroscopy; the DR16 HATS is the static imaging
  photometry product. The HATS has per-band MJD (`uMJD`…`zMJD`) and both PSF
  and model magnitude families.
- **S-PLUS:** DR5 is not public; use the CANFAR copy if present, otherwise
  public DR4. `splus_hats` is DR4.
- **UHS:** DR3 is public via WSA but has no public HATS; prefer the CANFAR
  copy. Do not run a WSA SQL harvest for the full sky; tile it if needed.
- **HSC:** the registry exposes per-band forced-photometry trees
  (`hsc_pdr3_g/i/r/y/z`), not one combined object row. Use the CANFAR combined
  HATS or stage the bands and combine by object ID.
- **DP2:** the full HATS is 3.6 TiB. Prefer match-at-source on RSP/NERSC and
  transfer only the match table; a full CANFAR sync needs a dedicated
  `/arc/projects` allocation.
- **Gaia duplicates:** the LSDB and CDS row counts differ slightly. Validate
  `source_id` uniqueness/partition routing before treating either as the exact
  official catalogue.
