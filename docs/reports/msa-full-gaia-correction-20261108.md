# Full-Gaia scope correction and age sources — 2026-11-08

The running expanded-six model uses the restricted pilot feature contract. It is a comparison model, not completion of the requested all-Gaia architecture. It is not retroactively changed by the new helpers below.

The intended parent is Gaia DR3 independently of XP availability. Join labels to that parent before joining optional XP or other photometry. Preserve unmatched and ambiguous labels with reasons. Existing training roles must remain grouped by Gaia identity across surveys and publications.

All reliable WISE, Gaia, SkyMapper, SDSS, Pan-STARRS and 2MASS bands should participate in encoding and masked reconstruction with missingness, errors and native quality/zero-point metadata. Include actual no-XP stars and whole-XP modality dropout. Explicit auxiliary head context includes measured signed parallax, EBV, sky position and proper motion with uncertainty/missingness. The new full_gaia_inputs helper expresses eligibility and these channels; it is not yet integrated into a rebuilt trainer or dataset.

The consensus_survey_map prototype fixes an overlap-weighted coordinate convention with no survey fixed as identity. Its tests verify the weighted identity constraint and permutation invariance. This is not an absolute physical calibration or a complete multivariate noise model. Source definitions and overlap-domain audits remain necessary, especially for different alpha quantities. Alpha/Fe, alpha/M and Mg/Fe must not be silently equated.

Ages should be indexed by publication, estimator and release, not only by survey. The machine-readable candidate registry is configs/published-age-source-registry-20261108.json. Verified published options include:

| Association | Published age product | Method / scope |
|---|---|---|
| APOGEE DR17 | [Leung et al. 2023](https://arxiv.org/abs/2302.05479) | Spectral latent-space estimator calibrated with Kepler seismic ages |
| APOGEE DR17 | [Anders et al. 2023](https://doi.org/10.1051/0004-6361/202346666) | XGBoost ages for 178,825 red giants |
| APOGEE DR17 | [2024 CatBoost age catalogue](https://doi.org/10.1051/0004-6361/202348031) | Asteroseismically calibrated chemical-clock ages for 125,445 giants |
| LAMOST + Gaia | [Updated 2024 subgiants](https://doi.org/10.12149/101467) | Separately published age catalogue; native LAMOST DR7 lineage survives a match to DR12 |
| LAMOST DR4 | [Wu et al. 2019](https://arxiv.org/abs/1901.07233) | Ages and masses for 640,986 RGB stars |
| RAVE DR6 | [BDASP](https://www.rave-survey.org/metadata/ravedr6/dr6_bdasp/) | Bayesian isochrone ages; 494,695 table rows are not unique-star or usable-age counts |
| GALAH DR4 | [Main catalogue and separate BSTEP VAC](https://www.galah-survey.org/dr4/the_catalogues/) | Two named age estimators; shared observations mean dependent information |
| Gaia RVS | [Kordopatis et al. 2023](https://arxiv.org/abs/2206.07937) | GSP-Spec, Gaia/2MASS and isochrones |

Additional local candidates include AMBRE, LAMOST–Kepler, APOKASC and APO-K2 tables. Filename alone does not establish which publication, estimator, usable age count or Gaia release they contain. DESI, Gaia-ESO, BOSS/eBOSS/SEGUE and MaSTAR still need product-specific discovery; crossmatching a published age to their stars supplies an association, not a new independent age observation.

Do not make StarHorse the identity reference for age. All these ages have model/training dependencies, and many share seismic calibrators, photometry or parallaxes. Record those dependencies, priors, validity domains, posterior summaries and quality cuts. Do not multiply published posterior PDFs as independent likelihoods. Multiple age products for a star retain one star split, and one age estimate associated with several surveys remains one measurement.

The header-only CANFAR inventory full_gaia_source_inventory_20261108_v1 is inspecting existing Gaia/catalogue storage without an XP membership requirement. Local evidence is saved in msa-insights-20261108/full-gaia-source-inventory.json. Discovery and schema inspection do not constitute admission into training. The corrected full-Gaia dataset and trainer have not yet been trained.

Validation: the new input/map contract tests and documented pipeline tests completed together: 8 passed. They are synthetic logic checks, not evidence of model quality.

Verified local header findings: APOGEE latent-age CSV has APOGEE_ID, Age/Age_Error and LogAge/LogAge_Error; RAVE BDASP has age/error in years; AMBRE has 1,065 rows with Gaia DR2 identifiers and ages in Gyr; LAMOST dr8_age.dat has 1,348,645 rows with obsid, separate RGB/RC age/error columns in Gyr. The latter publication association is pending; a strong published candidate is [Wang et al. 2023](https://arxiv.org/abs/2305.04528). None of these raw row counts is a measured quality-selected Gaia DR3 unique-star count.
