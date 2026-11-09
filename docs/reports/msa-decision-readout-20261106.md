# Decision-wave readout — 6 November 2026, 11:25 Perth

Four Gaussian/mixture arms completed 10,000 updates and native PDF QA. Four
unrestricted joint NSF arms are at 4,500–5,175 updates; no final-head selection
yet. These are controlled six-target pilots on the historical eligible FIT
pools, not the expanded final multi-survey training.

On the identical 6,147-star PDF panel, adding measured parallax changes the
mixture gravity CRPS from 0.08861 to 0.05605 dex (36.7% lower). Its temperature,
iron and age reference CRPS also improve, while distance is nearly unchanged.
The mixture+parallax has the lowest masked standardized energy among completed
arms (0.28051 versus Gaussian+parallax 0.28229), a small observed difference,
not a significance or physical-truth claim. Only 10 panel stars have all six
reference targets; the historical FIT pool has 277. Age/distance remain
posterior-reference targets. Reference-noise and population-specific PDF QA
must be assessed before calling the physical uncertainty calibrated.

The current frozen three-target model's large catalogue QA completed: exact
matches to 2,432,999 Andrae and 2,497,991 Gaia Net rows over all base roles;
devtest overlaps are 239,639 and 245,060 respectively. The properly
survey-mapped model has lower native-reference MAE than Gaia Net for all
three atmospheric targets in APOGEE/DESI/GALAH comparisons. This is catalogue
reference compatibility, with different input/teacher information, not an
independent physical-accuracy ranking.

Andrae remains a serious gravity comparator. On 46,640 GALAH devtest stars
known outside its released training sample, its gravity MAE is 0.07260 dex
versus our frozen three-target model's 0.10447 dex. Our temperature MAE is
53.56 K versus 73.42 K, and native Fe/H versus Andrae M/H is only a proxy
comparison (0.06361 versus 0.07076 dex). Only 184 APOGEE stars in the overlap
are known outside Andrae's released training sample, so the much larger
APOGEE comparison must not be called independent teacher holdout evidence.

Both Gaia-ESO crossmatch routes completed. ESA processed all 114,916 source
rows in 23 bounded batches and returned 621,688 candidate pairs;
112,890 source rows have the same single candidate under epochs
2000/2010/2016. The local route scanned all 218,131,884 raw input rows,
found 99,412 distinct candidate Gaia IDs, and matched about 60,600–60,700
Gaia-ESO source rows within 1 arcsec depending on assumed epoch. Native
source epochs are unknown; finite-radius and proper-motion missingness remain
explicit. These are not promoted training identities. Results remain private.

Decision: keep measured parallax in the shortlist, await the four NSF arms,
compare proper scores and noise-aware coverage, and finish the expanded
protected source/input bundle before final launch. Target remains 7 November
2026 around 19:30 Perth. Final large training has not started.


## Subsequent mapping and full-footprint audit

The full ID census now confirms **all 174,922,161 Andrae stars** are present
in our raw input. The reconstruction cache covers 120,181,886 of them; the
remaining 54,740,275 require the raw-input lookup rather than that cache.
The previous 2,432,999 figure described existing prediction exports only.

A new trained mapping audit and FIT/TUNE-only seismic anchor controls identify
the equal-weight relative gauge and underdispersion as major current logg
problems. The nonlinear-versus-affine propagation difference is much too small
typically to explain the 0.229 dex median offset on 734 seismic devtest stars.
The current six-target mixture+parallax pilot scores 0.073 dex logg MAE and
90.2% quoted-noise coverage on the same seismic sample, a different pooled-label
recipe from the frozen mapped-survey model. NF results remain pending.

The requested deliverable is six flexible marginal distributions conditional
on Gaia inputs, with population corners and genuine within-star corners where
joint draws are available. Full joint fitting is optional. Survey mappings
must cover all supported subsets of the six targets. See the
[mapping/model-choice audit and corner plots](msa-mapping-density-audit-20261106.md).


## Completed-wave update

All eight controlled 10,000-update arms and their selected-state PDF, seismic,
random-panel and corner QAs have now completed. Earlier pending/running
paragraphs describe their original observation times. NSF+parallax is a
promising candidate with 0.06977 dex seismic logg MAE, but its nominal 90%
coverage is 82.43%, versus mixture+parallax 0.07309 dex / 90.19%. No final
family is selected. See the [completed-wave lessons](msa-completed-wave-20261106.md).
