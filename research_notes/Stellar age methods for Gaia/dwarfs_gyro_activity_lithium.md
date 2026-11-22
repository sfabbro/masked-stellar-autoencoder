# Non-isochrone age-dating of main-sequence (dwarf) stars: gyrochronology, magnetic activity, lithium (state of the art to Nov 2026)

Scope: methods that date FGKM dwarfs without isochrone fitting, how well they work, how they are calibrated, their systematics, which large catalogues of the observables they need exist, and how they could feed or validate a data-driven age catalogue of about 2 billion Gaia stars, including stars with only UV-to-IR photometry. Dates are given for each result. Anything before 2023 is marked "(older)".

---

## Q1. Gyrochronology: current calibrations, stalled spin-down and weakened braking, dependence on mass/[Fe/H]/binarity, precision, calibrators, validity beyond solar age

### Takeaway
Empirical gyrochronology works well for FGK dwarfs from about 0.1 to 2.5–4 Gyr. Statistical precision is about 10–40%; for solar-type stars it improves from about ±38% at 0.2 Gyr to about ±12% at 2 Gyr. Below that range it is degraded by the young-star spread in rotation rates, and above it by stalled spin-down (K dwarfs, about 0.7–2.5 Gyr) and weakened magnetic braking (above about solar Rossby number). Calibrations are still bottlenecked by a handful of benchmark clusters with uncertain absolute ages. A 2026 analysis shows that models with no well-sampled cluster between 1 and 2.5 Gyr bias G and early-K ages high by 20–50%. Asteroseismic comparisons put gyro ages about 13% low below 6 Gyr and about 47% low above it.

### Cited Findings
**Calibration frameworks**
- (older) Barnes (2003) formalized gyrochronology as an empirical rotation–colour–age mapping, building on Skumanich (1972) P ∝ t^1/2. Cited as background in the 2026 NGC 752 calibration paper — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- (older) Angus et al. (2015) calibrated P = A^n (B−V−c)^b with Kepler asteroseismic stars, Hyades, Coma Berenices and field stars including the Sun. They concluded that no single relation fits all three subsets (asteroseismic, cluster, local field stars) — [arXiv:1502.06965](https://arxiv.org/pdf/1502.06965)
- **gyro-interp (Bouma, Palumbo & Hillenbrand 2023):** interpolation between open-cluster sequences, with open-source code. For ≈5800 K stars, statistical age uncertainty falls monotonically from ±38% at 0.2 Gyr to ±12% at 2 Gyr. For ≈4200 K K dwarfs the posteriors are highly asymmetric because of stalled spin-down. The model is not applicable beyond 2.6 Gyr because calibration data are sparse there. The main systematics are possible changes in spin-down rate and the absolute age scale — [arXiv:2303.08830](https://ar5iv.labs.arxiv.org/html/2303.08830); [ADS-indexed record](https://authors.library.caltech.edu/records/saj66-4tn57)
- **gyro-interp applied to Kepler (Bouma et al. 2024, AJ):** rotational ages for 23,813 Kepler FGK stars hosting 795 planets. It recovers open-cluster ages from 0.04 to 2.5 Gyr and agrees with more than 90% of independent Keck/HIRES lithium ages. With an age-dependent population-level scatter, age uncertainties for 2–4 Gyr stars with about 5% period precision grow by up to about 50% — [arXiv:2410.06246](https://arxiv.org/pdf/2410.06246); [abstract](https://arxiv.org/abs/2410.06246)
- **ChronoFlow (Van-Lane et al. 2025):** a conditional normalizing flow trained on a standardized catalogue of about 8,000 rotators in 30 open clusters and associations aged 1.5 Myr to 4 Gyr. Cluster ages: 0.06 dex statistical (about 15%), with 0.06 dex added systematics from extinction, membership and calibration ages, for 0.08 dex total. **Individual-star ages: 0.7 dex statistical** — [arXiv:2412.12244](https://arxiv.org/abs/2412.12244). Its training set has only 8 NGC 752 periods, so it lacks a defined slow sequence at 1.3 Gyr — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- **Lu, Angus, Foreman-Mackey & Hattori 2024 (AJ), "In this Day and Age":** a Gaussian-process gyro relation for partially and fully convective single field stars. It is conditioned on gyro-kinematic ages of field stars (about 1–14 Gyr) and clusters (0.67–3.8 Gyr). Cross-validation recovers cluster and asteroseismic ages with an average uncertainty of just over 1 Gyr. It uses ZTF periods with Gaia cuts G = 13–18 and BP−RP < 1. ZTF and Kepler periods agree within 10% for 81% of 1,270 overlapping stars at P > 4 d, and ZTF periods under 4 d were rejected. Reported sample sizes differ by source: "gyrochronology ages for ∼100,000 stars within 1.5 kpc" versus "gyro-kinematic ages for ∼50,000 stars" — [arXiv:2310.14990](https://arxiv.org/pdf/2310.14990); [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.14990)
- **Gyro-kinematic ages** (older for the first two): Angus et al. (2020) and Lu et al. (2021, about 30,000 Kepler stars) date stars from the velocity dispersion of groups matched in rotation, temperature and absolute magnitude. This addresses the flattening of rotation at old ages. Lu et al. (2024) report single-star ages up to about 14 Gyr — summarized in the HAges catalogue, [arXiv:2605.12647](https://arxiv.org/html/2605.12647)
- Other calibrations still in use (HAges 2026 compilation): Barnes (2007), Mamajek & Hillenbrand (2008), Angus et al. (2015), Engle & Guinan (2023) and Lu et al. (2024). Typical statistical uncertainties are about 10–40%. Gyro ages are "generally reliable to about 4 Gyr" — [arXiv:2605.12647](https://arxiv.org/html/2605.12647)

**Benchmark cluster calibrators and the absolute age scale (2026)**
- Clusters used in current calibrations, with adopted ages: α Per 85 Myr; Pleiades, Pisces–Eridanus and Blanco 1 at 120 Myr; NGC 3532 and Group X at 300 Myr; Praesepe and Hyades at 670 and 700 Myr; NGC 6811 at 1 Gyr; NGC 752 at 1.34 Gyr (adopted 1.3 Gyr); NGC 6819 at 2.5 Gyr; Ruprecht 147 at 2.7 Gyr; M67 at 4 Gyr — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- **NGC 752 (2026):** the benchmark grew from 8 to 25 single, secure members with periods (19 from TESS/TARS, 7 from Agüeros+2018, 1 in common). The slow sequence sits at P ≈ 15 d for 4000–6000 K. gyro-interp and ChronoFlow overestimate ages of many G and early-K dwarfs by **20–50%**. For stars hotter than about 5000 K both overpredict by more than 1σ, often by more than 400 Myr. A well-measured period could give ≲200 Myr (<15%) precision at about 1.3 Gyr, against 20–50% from existing tools at 1–2.7 Gyr. Spin-down has resumed for 4500–5000 K by 1.3 Gyr. There are no slow-sequence data at 3700–4400 K — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- **The absolute age scale is a dominant systematic.** Literature ages for NGC 752 range over 1176 Myr (Wang+2025), 1420 Myr (Hunt & Reffert), 1610 Myr (Sandquist+2023) and 1750 Myr (Malhotra+2026). The authors note that part of the 20–50% offset may come from this uncertainty — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- 20,000 rotation periods in nearby open clusters and moving groups from ZTF (18,614 signals) and ASAS-SN (1,018). For the ASAS-SN part, 98% agree with literature periods or their harmonics. This is a new benchmark resource (2026) — [arXiv:2610.27113](https://arxiv.org/pdf/2610.27113)

**Stalled spin-down (K dwarfs)**
- (older) Curtis et al. (2020) measured periods for 58 members of Ruprecht 147 (2.7 Gyr) with K2 and PTF, 39 after binary cuts. Combined with NGC 6819, the sample reaches about 0.55 M☉. Stars of 0.55 M☉ remain stalled for at least 1.3 Gyr, and the stall lasts longer at lower mass. A core–envelope coupling model tuned to clusters explains most of the stall, but a temporary drop in braking torque cannot be excluded — [arXiv:2010.02272](https://arxiv.org/pdf/2010.02272)
- (older) Spada & Lanzafame (2020): a two-zone model in which the wind-braking torque is balanced by angular momentum transferred from the faster-rotating radiative core. The coupling is strongly mass-dependent, so K stars stall longer. It overshoots mid-K dwarfs in NGC 752 by 1.6 d and in Ruprecht 147 by 2.7 d — [Caltech record for Curtis+2020](https://resolver.caltech.edu/CaltechAUTHORS:20201204-110357459); [arXiv:2010.02272](https://arxiv.org/pdf/2010.02272)
- Ultraviolet evidence for K-dwarf rotational stalling in the first Gyr (HAZMAT VIII, older, 2022) — [arXiv:2203.15237](https://arxiv.org/pdf/2203.15237)

**Weakened magnetic braking (WMB) beyond about solar age**
- (older) van Saders et al. (2016) proposed WMB to explain anomalously rapid rotation in old Kepler field stars — [arXiv:1601.02631](https://arxiv.org/pdf/1601.02631)
- Metcalfe et al. (2025, ApJL 991, L17): WMB "signals the collapse of the global stellar dynamo". With WMB, a solar model reaches the main-sequence turnoff at Ro ≈ 1.1 Ro☉, against Ro ≈ 1.8 Ro☉ for standard spin-down. The truncated period distribution has been confirmed with 94 stars whose rotation comes from asteroseismology — [ADS](https://ui.adsabs.harvard.edu/abs/2025ApJ...991L..17M); [IOP](https://iopscience.iop.org/article/10.3847/2041-8213/ae03bc)
- Metcalfe et al. (2025, ApJ), "Testing the Rossby paradigm": WMB in six early-K stars, all with clear activity cycles. The asteroseismic age of HD 219134 agrees with the WMB age and is slightly older than the standard spin-down age — [NSF PAR](https://par.nsf.gov/servlets/purl/10613353)
- A new asteroseismic Kepler benchmark constrains the onset of WMB in mature Sun-like stars (2024) — [arXiv:2405.12362](https://arxiv.org/pdf/2405.12362). Details were not extracted.
- All the 2025 WMB papers found come from one group. The search found no independent rebuttal — [ADS](https://ui.adsabs.harvard.edu/abs/2025ApJ...991L..17M)
- (older, 2022) Agreement of coeval wide pairs with gyro relations gets worse at older ages — [arXiv:2210.01137](https://arxiv.org/abs/2210.01137v1)

**Gyro vs asteroseismology (accuracy test)**
- HAges (2026), 14 stars with both: median gyro offset ≈ **−13%** relative to asteroseismology for ages ≲6 Gyr, and ≈ **−47%** for older stars. In the full compilation, median gyro uncertainties are about 16% statistical and 18.3% systematic. Asteroseismic ages for LEGACY dwarfs have median uncertainties of about 8% statistical and 12% systematic — [arXiv:2605.12647](https://arxiv.org/html/2605.12647)
- Failure mode: spurious long periods. 51 Peg's 37 d period, used in earlier gyro ages, is probably wrong; a reanalysis gives 21.9 d — [arXiv:2605.12647](https://arxiv.org/html/2605.12647)

**Metallicity dependence**
- (older) Amard, Roquette & Matt (2020): metal-rich Kepler field stars rotate more slowly on average, matching models in direction and amplitude — [arXiv:2009.11785](https://arxiv.org/pdf/2009.11785); [ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3481A)
- "The Maunder" (Kamai, Somers & Perets, Aug 2026): rotation depends on metallicity at fixed mass. Above about 0.85 M☉ the median period increases monotonically with [Fe/H], which the age–metallicity relation cannot explain. This confirms See et al. (2024) on a larger sample — [arXiv:2608.06604](https://arxiv.org/pdf/2608.06604)
- Testing whether metal-rich stars spin down faster with gyro-kinematic ages (2024) — [arXiv:2405.00779](https://arxiv.org/pdf/2405.00779). Result not extracted.
- The metallicity dependence of gyrochronology "remains poorly constrained" (Gaidos et al. 2023, via HAges) — [arXiv:2605.12647](https://arxiv.org/html/2605.12647)
- Ye et al. (2024): among F dwarfs, P_rot, age and [Fe/H] are strongly correlated. For G dwarfs P_rot is a reliable age indicator at about 2–11 Gyr. For K dwarfs the P_rot–age relation is not valid at 3–13 Gyr — [arXiv:2401.15438](https://arxiv.org/abs/2401.15438)

**Binarity and contamination**
- Close or unresolved binaries produce anomalously fast rotators. Example: three NGC 752 stars at 4000–4500 K with P < 2 d are attributed to undetected companions. Standard cuts are RUWE > 1.4, a non-zero Gaia non_single_star flag, CMD over-luminosity and excess RV scatter. RUWE misses some companions, at certain separations and when they are about 5 mag fainter — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- Wide binaries are used to validate TESS periods through gyrochrone consistency: 88% recovery for P < 12 d (2025) — [arXiv:2507.08266](https://arxiv.org/html/2507.08266)
- [C/N] and gyro ages were cross-calibrated through the age–velocity-dispersion relation (2025). Gyrochronology applies to partially convective stars before the onset of WMB — [arXiv:2506.24010](https://arxiv.org/abs/2506.24010)

### Inferences
- A practical validity window for single-star gyro ages is about 0.3–3 Gyr for G and early-K dwarfs, with about 15–30% total error once absolute-age-scale systematics (about 10–20% from NGC 752-type disagreements) are included. For mid-K dwarfs the window is narrower and multimodal at 0.7–2.5 Gyr. Above about 4 Gyr, or Ro ≳ Ro☉, gyro ages are biased young (about −47% versus asteroseismology) unless WMB is modelled.
- The calibration sample is about 10 benchmark clusters at ≤4 Gyr. Any data-driven model that inherits gyro labels inherits their discrete age grid and the uncertainty in cluster ages. Cluster ages should be free parameters with priors, not fixed labels.
- Metallicity enters at the level of a monotonic period shift at fixed mass. A Gaia-scale model should condition on [Fe/H] (from XP or spectroscopy) when it uses rotation, or it will confuse metallicity with age.

### Gaps
- No current calibration paper by a "Mazzola et al." on gyrochronology turned up in two searches, so its content could not be verified. It may be misattributed or very recent.
- Spada & Lanzafame (2020) was covered only through secondary descriptions. The primary paper was not retrieved.
- Soderblom (2010, ARA&A) and later age reviews were not retrieved in this session.
- Gyro precision broken down by spectral type and age, beyond the gyro-interp ≈5800 K numbers and the K-dwarf asymmetry, was not tabulated in the sources retrieved.
- The independence of the WMB evidence (one group) could not be assessed further.

---

## Q2. Rotation-period catalogues: sizes, period ranges, magnitude and sky coverage, future yields

### Takeaway
By late 2026 about 1.5–2 million unique stars have published rotation periods. The largest catalogues are TESS TARS (1.05 M stars within 500 pc, T < 16), Gaia DR3 (0.47 M), the Kepler field (0.15 M in the 2026 "Maunder" ML catalogue; 55 k in Santos+2021), ZTF (about 40 k late-K/M) and ASAS-SN (about 50–200 k variability periods). That is about 0.1% of Gaia. TESS and other ground or space surveys lose completeness above about 10–13 d, so field samples are biased toward young or fast rotators. Gaia DR4, PLATO and Rubin are expected to raise counts to "tens of millions".

### Cited Findings
- **Kepler, McQuillan et al. (2014)** (older): 34,030 periods out of 133,030 stars — as cited in [TARS, arXiv:2603.05586](https://arxiv.org/html/2603.05586v2)
- **Kepler, Santos et al. (2019, 2021)** (older): 55,232 main-sequence and subgiant FGKM stars with periods out of 159,442 targets, 24,182 of them new relative to McQuillan. Paper II (F/G and cool subgiants) gives 39,592. Periods come from wavelet plus ACF analysis with ML selection, and come with the photometric activity proxy S_ph — [ADS 2021ApJS..255...17S](https://ui.adsabs.harvard.edu/abs/2021ApJS..255...17S/abstract); [arXiv:2107.02217](https://arxiv.org/abs/2107.02217v1); VizieR J/ApJS/255/17 [GAVO record](https://dc.g-vo.org/rr/q/lp/custom/CDS.VizieR/J/ApJS/255/17)
- **Kepler, "The Maunder" (2026):** an ML catalogue of periods for 148,746 main-sequence stars, trained self-supervised plus supervised on cross-catalogue consensus labels. It gives conformalized per-star uncertainties, and 119,428 stars form the high-reliability subset. 31,953 stars (21.5%) show bimodal signals, and APOGEE v sin i shows the longer period is the true one. Single-pass periodograms lock onto short aliases, a systematic failure — [arXiv:2608.06604](https://arxiv.org/abs/2608.06604)
- **TESS All-Sky Rotation Survey (TARS; Boyle, Bouma & Mann 2026, ApJS):** periods for 1,046,317 stars (v2; 944,056 in v1) within 500 pc and T < 16. The input was 7,481,412 targets and 39,044,569 FFI sector light curves through Sector 96 (Sep 2025). Method: Lomb–Scargle with two random-forest classifiers, for systematics and half-period harmonics. Periods up to about 25 d are recoverable from a single sector, but completeness declines above about 10–13 d. About 93% of Gaia-matched periods are rotation; estimated purity is about 85% for the default catalogue and 92% for the strict subset. 314,650 stars are "clean" and 216,007 form the highest-quality, multi-sector-consistent subset. Rotation counts for bright stars increase ×2.3 within 100 pc and ×4.0 within 500 pc. Match rates with all flags: ZTF 98.8% (18,894 stars), K2 96.1% (16,823), clusters 95.6% (2,632), Kepler 68.8% (27,711). Data are on Zenodo and MAST — [arXiv:2603.05586](https://arxiv.org/html/2603.05586v2); [IOP ApJS](https://iopscience.iop.org/article/10.3847/1538-4365/ae6657); [MAST HLSP](https://archive.stsci.edu/hlsp/tars); [Zenodo](https://zenodo.org/records/19917941)
- Earlier TESS catalogues (counts limited to T < 16, d < 100 pc): Fetherolf 12,632; Holcomb 4,670; Colman 3,454; Kounkel 1,782; Distefano (Gaia) 1,867; Claytor 685 — [arXiv:2603.05586](https://arxiv.org/html/2603.05586v2)
- TESS periods are reliable below about 10 d, and completeness and accuracy drop above about 12 d (Boyle et al. 2025, via HAges). Only 30 of 659 HWO priority stars are in TARS, so bright nearby stars are poorly covered — [arXiv:2605.12647](https://arxiv.org/html/2605.12647)
- Late-M single dwarfs in TESS: a dedicated catalogue (2025) — [arXiv:2507.10847](https://arxiv.org/html/2507.10847)
- **Gaia DR3 rotational modulation (Distefano et al. 2023, A&A 674, A20):** 474,026 magnetically active stars, about 430,000 of them newly discovered variables. About 70 parameters per star, including P, amplitude A and the magnitude–colour correlation r0 — [arXiv:2206.05500](https://arxiv.org/abs/2206.05500); [A&A](https://www.aanda.org/10.1051/0004-6361/202244178); [Gaia DR3 docs](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu7var/sec_cu7var_rotmod/)
- **ZTF (Lu et al. 2022):** 40,553 periods (9,535 above 10 d) for late-K to M dwarfs. A custom pipeline reduces photometric scatter by 26% — [arXiv:2210.06604](https://arxiv.org/pdf/2210.06604)
- **ASAS-SN:** about 50,000 spotted stars in seven classes (Phillips et al. 2024) — [arXiv:2305.09715](https://ar5iv.labs.arxiv.org/html/2305.09715). 200,000+ deep-learning periods of stellar variability, not all rotation (2025) — [arXiv:2509.14423](https://arxiv.org/pdf/2509.14423). A 2026 CNN study states that ASAS-SN, ZTF and others had measured P_rot for about 100,000 stars — [Schochet 2026 ApJS 282, 10](https://m-schochet.github.io/files/Schochet_2026_ApJS_282_10.pdf)
- **eROSITA × Gaia DR3 rotation:** 3,700 coronal X-ray sources with known periods, the largest such sample — [Freund et al. 2024, arXiv:2401.17282](https://arxiv.org/html/2401.17282v1)
- **PLATO (Breton et al. 2024, A&A 689, A229):** periods recoverable for more than 70% of solar-type stars with 6 months of data and more than 90% with 4 years, based on simulations with spot evolution, cycles, granulation and instrument noise — [arXiv:2407.03709](https://www.arxiv.org/abs/2407.03709); percentages via [arXiv:2605.12647](https://arxiv.org/html/2605.12647)
- **Rubin/LSST** (older forecast, 2016): period sensitivity peaks near 20 d and falls at short periods because of the roughly 3-day revisit interval. The authors call the estimates optimistic because of systematics — [arXiv:1607.04302](https://arxiv.org/pdf/1607.04302). A 2026 paper says Rubin "could recover hundreds of thousands" of periods — [Schochet 2026](https://m-schochet.github.io/files/Schochet_2026_ApJS_282_10.pdf)
- Gaia (DR4), Roman and PLATO are expected to yield periods for "tens of millions of stars" — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)

### Inferences
- Total unique rotators today number about 1.5–2 million, probably dominated by TARS plus Gaia DR3. The overlap between the two was not computed here (TARS cross-matched 293,619 Gaia DR3 variability-class sources). Either way it is about 0.1% of a 2-billion-star catalogue, so gyrochronology can supply training and validation labels, not direct ages at Gaia scale.
- The selection is strongly biased to P ≲ 12 d for TESS and ground surveys, which means young stars or K/M dwarfs. A solar-age G dwarf has P ≈ 25 d, and old K and M dwarfs have P > 30 d. Gyro labels drawn from these catalogues therefore under-represent old stars. A model trained on them would learn an age prior skewed young unless the selection function is modelled.
- For the Kepler field, conformal per-star period uncertainties ("Maunder") and multi-sector consistency flags (TARS) are the best quality weights for label propagation.

### Gaps
- Reinhold & Hekker K2 catalogue size: only the 16,823-star overlap with TARS was found.
- Claytor et al. (2024) and Colman et al. (2024) TESS catalogue full sizes and period ranges were not retrieved; only their counts within 100 pc are given above.
- Gaia DR3 rotational-modulation period range, magnitude range and DR4 rotation expectations were not extracted from the primary paper.
- No Rubin-data rotation catalogue exists yet.

---

## Q3. Activity–age relations (Ca II H&K, Hα, X-ray, UV/GALEX): calibration, scatter, saturation, metallicity; large activity datasets and their use at Gaia scale

### Takeaway
Chromospheric (R′HK) ages give about 0.2 dex (older Mamajek & Hillenbrand 2008) to about 15% (solar-twin calibrations) precision for solar-type stars between Hyades age and the Sun. They are degraded by activity cycles, saturation at young ages and a contested flattening after a few Gyr. 2024–2026 LAMOST work finds R′HK still declining to about 13 Gyr in G and K dwarfs, and that metal-poor FGK stars are more active at fixed age. X-ray (eROSITA eRASS1: about 139 k coronal sources) and UV (GALEX: 83 M sources, NUV ≈ 20.8) are the only activity proxies at the scale of tens of millions of stars or more. They are useful mainly as young-versus-old discriminators (about 0.3–0.5 dex scatter) rather than as precise clocks.

### Cited Findings
**Chromospheric Ca II H&K**
- (older) Mamajek & Hillenbrand (2008): activity–rotation–age calibration with typical age precision of about 0.2 dex for normal solar-type dwarfs between the Hyades age and the Sun — [arXiv:0807.1686](https://arxiv.org/pdf/0807.1686)
- (older) "The age–chromospheric activity relation" (2018, solar-type stars): a chromospheric age error of about 15%. Ca II H&K remain a useful chromospheric evolution tracer until at least 6–7 Gyr — [arXiv:1806.08014](https://arxiv.org/pdf/1806.08014)
- **Ye et al. (2024):** 52,321 FGK dwarfs from LAMOST DR7 plus Kepler plus Gaia DR3, with masses and ages; 47,489 with log R′HK; 6,077 with P_rot and S_ph. For G dwarfs log R′HK is a reliable age indicator over about 2–13 Gyr. For K dwarfs log R′HK drops markedly over about 3–13 Gyr. For F dwarfs log R′HK does not decline significantly, except for [Fe/H] < −0.1 — [arXiv:2401.15438](https://arxiv.org/abs/2401.15438)
- **Han et al. (Jan 2026, ApJS):** LAMOST R′HK for open-cluster members and field stars. A Skumanich-type (linear) relation best describes the decline, with a slope that depends on spectral type. **Lower-metallicity F, G and K stars are more active**, and M dwarfs show no clear metallicity dependence. The work is framed against claims that the activity–age relation flattens — [arXiv:2601.17715](https://arxiv.org/abs/2601.17715); [HTML](https://arxiv.org/html/2601.17715)
- A LAMOST/GALEX study of evolved stars uses Ca II H&K, Hα and GALEX GR6+7 NUV (3″ cross-match) — [arXiv:2411.12994](https://arxiv.org/html/2411.12994)
- Hα: Fine structure of the age–chromospheric activity relation in solar-type stars, II, Hα — [ResearchGate](https://www.researchgate.net/publication/381517938_Fine_Structure_of_the_Age-Chromospheric_Activity_Relation_in_Solar-Type_Stars_II_Halpha_Line). Quantitative content was not retrieved.
- (older) Kepler photometric activity proxy S_ph–age relation derived from Kepler (2023) — [arXiv:2306.11657](https://arxiv.org/pdf/2306.11657)

**X-ray (eROSITA)**
- **Freund et al. (2024), HamStar:** 138,800 coronal eRASS1 sources from a Bayesian match to Gaia DR3, with about 91.5% completeness and reliability. 6,700 are known unresolved binaries, and about 10% have a correlated secondary counterpart. 3,700 have Gaia DR3 rotation periods — [arXiv:2401.17282](https://arxiv.org/abs/2401.17282); [eROSITA DR1 catalogue page](https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/FreundS_DR1/HamStar_eRASS1_Main_Likely_Identifications_v1.1.html)
- **Stassun & Kounkel (2024, ApJL):** about 8,200 F5V–M5V stars with eROSITA L_X, TESS periods and ages of 3–500 Myr from Gaia-identified clusters and groups. Empirical forms predict L_X from basic observables to within 0.3 dex, and angular momentum without a measured period to within 0.5 dex. The activity–rotation relation is simpler in angular momentum than in period — [arXiv:2405.08607](https://arxiv.org/abs/2405.08607)
- In the eROSITA DR1 variability catalogue, most strongly variable stars are M dwarfs, and 10–20% are very young or pre-main-sequence — [arXiv:2401.17280](https://arxiv.org/html/2401.17280)
- eROSITA stellar X-ray luminosity function (2025) uses L_X/L_bol for saturation and age trends, with F/G/K/M subsamples defined by Gaia colours — [A&A 2025](https://www.aanda.org/articles/aa/full_html/2025/10/aa56420-25/aa56420-25.html)
- Asteroseismic anchor: of 66 Kepler LEGACY stars, 13 are detected by eROSITA. The revised activity–rotation–age relations improve L_X agreement for 7 of the 13 (Dec 2025) — [arXiv:2512.14517](https://arxiv.org/abs/2512.14517)

**UV (GALEX)**
- **GUVcat_AIS (Bianchi et al. 2017):** 82,992,086 unique sources over 24,790 deg² (fov055), or 69.8 M over 22,125 deg² (fov050). Typical depth FUV = 19.9 and NUV = 20.8 AB — [arXiv:1704.05903](https://web3.arxiv.org/abs/1704.05903); [MAST HLSP](https://archive.stsci.edu/hlsp/guvcat). Matched GALEX × Gaia DR2/SDSS catalogues are available (GUVmatch) — [arXiv:2007.03808](https://arxiv.org/pdf/2007.03808)
- GALEX extinction coefficients from 1,837 DA white dwarfs: R_FUV = 8.01, R_NUV = 6.79 — [MNRAS 489, 5046](https://academic.oup.com/mnras/article/489/4/5046/5565069?guestAccessKey=)
- **Li et al. (Sep 2025, ApJS):** about 386,500 A–M dwarfs in open clusters and the field, aged about 3 Myr to 10 Gyr, using normalized NUV emission f_NUV/f_J. A and early-F dwarfs show only a weak NUV decline. Late-F to G dwarfs show a clear decline, linked to magnetic-braking spin-down. Late-K and M dwarfs decline rapidly from the YSO phase to the main sequence — [arXiv:2509.18559](https://arxiv.org/abs/2509.18559)
- FUV–age calibration for FGK dwarfs in GALEX FUV versus Gaia BP−G space (Ap&SS 2022), extending Crandall et al. (2020) and applied to 660 GCS stars — [Springer](https://link.springer.com/article/10.1007/s10509-022-04074-1)
- (older) Broad-band UV activity–age relations show about 0.4 dex scatter when predicting age from FUV/NUV — [arXiv:1105.1377](https://arxiv.org/pdf/1105.1377)
- (older) TGAS × GALEX within 40 pc: about 3.6% of late-type dwarfs show NUV excess, corresponding to an age threshold of about 460 Myr under constant star formation — [arXiv:1705.01114](https://arxiv.org/pdf/1705.01114)
- T-GEX (2026): 37 FGK turnoff stars with GALEX, Gaia and Gaia-ESO data. About two-thirds show UV above the UV-normal locus, attributed to age-related excess — [arXiv:2607.27324](https://arxiv.org/html/2607.27324)
- Contaminant: Gaia–GALEX UV excess also selects white dwarf + FGK binaries, with 2,597 candidates within 350 pc — [arXiv:2609.25216](https://arxiv.org/abs/2609.25216). UV excess is also used to find metal-poor halo stars, so metallicity alters the photospheric UV — [arXiv:2605.29060](https://arxiv.org/pdf/2605.29060)

**Spectroscopic data-driven ages that touch activity**
- **Wang, Xiang et al. (Aug 2025, ApJS):** about 4 million LAMOST DR10 dwarfs with ages from XGBoost. Training labels come from wide binaries (the secondary inherits the primary's isochrone age) plus field and cluster stars. Precision is about 10–25% for K-type stars at S/N > 50, and worse for young stars. The age information comes mostly from chemical-abundance features (a "chemical clock" at R ≈ 1800). Masking the Ca II H&K and Hα lines gives almost identical ages — [arXiv:2508.03019](https://arxiv.org/abs/2508.03019); [PDF](https://arxiv.org/pdf/2508.03019)

### Inferences
- Activity relations are calibrated on the same cluster ladder as gyrochronology, plus isochrone-dated LAMOST/Kepler field stars (Ye+2024) or asteroseismic stars (LEGACY). They are not model-free beyond about 4 Gyr. The "flattening vs continued decline" disagreement after about 2–4 Gyr is unresolved, and the 2024–2026 LAMOST results (continued decline) rely on isochrone ages for field dwarfs.
- For photometry-only stars, GALEX NUV − optical/IR excess (for example f_NUV/f_J) is the only widely available activity proxy. About 83 M GALEX sources cover about 60% of the sky to NUV ≈ 20.8. An unknown but probably tens-of-millions fraction are Gaia dwarfs. A 0.3–0.5 dex scatter makes it a classifier for under about 0.5–1 Gyr versus older, not a clock. It must be decorrelated from [Fe/H], extinction (R_NUV ≈ 6.8) and WD companions.
- eRASS1 (about 139 k coronal stars, half sky) is mainly useful for young-star labels and saturation calibration at ≲0.5 Gyr.
- Activity cycles and rotational modulation add per-epoch scatter. Single-epoch survey activity indices (LAMOST, DESI) should be treated as noisy, with about 0.1–0.2 dex intrinsic scatter. This number was not directly sourced here; see Gaps.

### Gaps
- A DESI activity-index catalogue (Hα/Ca II) for dwarfs was not found in this session.
- A LAMOST LRS Ca II H&K database built from more than 1 million spectra was mentioned in search snippets, but the primary paper was not retrieved.
- Quantitative saturation thresholds (Ro_sat, log L_X/L_bol ≈ −3) and per-epoch cycle scatter were not extracted from 2023–2026 primary sources.
- eROSITA DR1 sky coverage (German half of the sky) and sensitivity limits were not verified from a primary source in this session.
- How well GALEX NUV excess measures age at fixed [Fe/H] for Gaia-scale dwarf samples is unquantified; no dedicated 2024–2026 NUV-age calibration with stated precision was found.

---

## Q4. Lithium depletion boundary (LDB) and Li ages for dwarfs: precision, mass range, surveys

### Takeaway
The LDB gives robust, nearly model-independent ages, with about 7–11% systematics and about 10% model dependence. It applies only to coeval groups aged about 20–200 Myr, and to the Hyades through brown dwarfs. For individual stars, the empirical EAGLES model (Gaia-ESO; 6,200 stars in 52 clusters, 2–6000 Myr) gives under 10% precision for young groups. It is mass-limited: M dwarfs are best at 10–100 Myr, K dwarfs at 30–300 Myr and G dwarfs at 100–1000 Myr. Systematics reach a factor of 2 in older clusters. GALAH DR4 (about 0.9 M Li equivalent widths) and LAMOST MRS (about 0.46 M stars) make Li a viable label for young dwarfs at the million-star scale.

### Cited Findings
- (older) Jeffries & Naylor (2001): LDB errors are about twice those claimed in the literature, with systematics roughly constant at 7–11% across ages. A depletion threshold of 90% versus 99% shifts ages by only about ±3%. Examples: Pleiades 122 Myr (+20/−17 experimental, ±11 systematic), α Per 85 Myr (+12/−10, ±8), IC 2391 48 Myr (±5, ±3) — [arXiv:astro-ph/0009179](https://ar5iv.labs.arxiv.org/html/astro-ph/0009179)
- (older) Burke et al. (2004): locating the LDB observationally (about 7–10%) and bolometric-correction uncertainty (about 6–10%) dominate the error. The method applies to clusters of about 20–200 Myr — [arXiv:astro-ph/0411111](https://ar5iv.arxiv.org/html/astro-ph/0411111)
- (older) Jeffries (2014 review): LDB ages differ by only about 10% between models. The LDB gives "robust, almost model-independent ages" but only for coeval groups. Initial Li adds about ±0.2 dex systematic when comparing with depletion predictions — [arXiv:1404.7156](https://web3.arxiv.org/pdf/1404.7156); see also Soderblom et al. (2014), "Ages of young stars" — [arXiv:1311.7024](https://ar5iv.arxiv.org/html/1311.7024)
- Hyades Li-depletion age of 695 Myr (Galindo-Guil et al. 2022), as cited — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- **EAGLES (Jeffries et al. 2023, MNRAS 523, 802):** an empirical Li equivalent-width versus Teff versus age model built from about 6,200 stars in 52 Gaia-ESO open clusters. Valid for 2–6000 Myr and −0.3 < [Fe/H] < 0.2. Precision is better than 10% for seven young moving groups. For older clusters, systematic errors can reach a factor of 2. Best sensitivity: M dwarfs at about 10–100 Myr, K dwarfs at 30–300 Myr, G dwarfs at 100–1000 Myr. Distributed as software — [arXiv:2304.12197](https://arxiv.org/abs/2304.12197); [MNRAS](https://academic.oup.com/mnras/article/523/1/802/7147327)
- EAGLES applied to Canis Major North gives an age consistent with the kinematic 23.7 ± 4.9 Myr (2026) — [arXiv:2610.36223](https://arxiv.org/pdf/2610.36223)
- (older) GALAH warm dwarfs: main-sequence Li depletion becomes ineffective after about 1–2 Gyr for most stars, leaving a plateau at older ages — [MNRAS Letters 497, L30](https://academic.oup.com/mnrasl/article/497/1/L30/5854574)
- **GALAH DR4 (2024):** 917,588 stars. The 3D-NLTE Li value-added catalogue has equivalent widths for 892,223 stars (97%), A(Li) detections for 417,825 (46%) and upper limits for 474,398 (52%). Many "detections" have very small equivalent widths and need caution — [arXiv:2409.19858](https://arxiv.org/html/2409.19858v2); [GALAH DR4 overview](https://www.galah-survey.org/dr4/overview/). A GALAH DR4 study of 22,442 stars with 3D-NLTE Li reconstructs the end of main-sequence Li depletion using YREC masses (2026) — [arXiv:2610.22492](https://arxiv.org/pdf/2610.22492)
- **LAMOST MRS DR9 Li:** 795,384 spectra of 455,752 stars, from pre-main-sequence to giants — [arXiv:2403.01815](https://arxiv.org/html/2403.01815v1)
- Li ages agree with more than 90% of gyro-interp rotational ages for Kepler FGK stars — [arXiv:2410.06246](https://arxiv.org/abs/2410.06246)

### Inferences
- Lithium is a strong young-star clock: under about 1 Gyr for G dwarfs and under about 0.3 Gyr for K and M dwarfs. Past those ages, upper limits mostly give lower bounds on age. It complements gyrochronology, which is weakest below about 0.3 Gyr because of the spread in initial rotation, and gives an independent check of the young end of any data-driven age model.
- EAGLES is trained on Gaia-ESO EW(Li). Applying it to GALAH, LAMOST or DESI needs cross-survey EW zero-point and resolution corrections, and its [Fe/H] range (−0.3 to +0.2) excludes much of the thick disk.
- The roughly 0.4 M GALAH Li detections plus LAMOST MRS give enough labels to train a "Li-young" classifier or regressor head on XP and photometry. The Li signal is in a narrow 6708 Å line that XP spectra cannot resolve, so any XP-based prediction would come from correlated properties, not the line itself.

### Gaps
- No Gaia-ESO final-release Li dwarf count was found; only the EAGLES calibration sample is given above.
- Dwarf-only counts within the LAMOST MRS Li catalogue, and any LAMOST LRS dwarf Li catalogue, were not found.
- No 2024–2026 LDB re-calibration with Gaia DR3 cluster parameters was retrieved.

---

## Q5. Combined approaches, empirical calibrations, and applying them at Gaia scale (2 B stars; photometry-only stars) with cross-calibration to isochrone and asteroseismic ages

### Takeaway
The proven combined approaches are: (i) stardate (older, 2019), where gyro and isochrone posteriors give up to 3× better precision than isochrones alone for main-sequence FGK stars with 5% periods; (ii) gyro-kinematic ages calibrated on the age–velocity-dispersion relation, reaching about 14 Gyr; (iii) wide-binary label transfer (isochrone age of a turnoff or subgiant primary given to a dwarf secondary), which trained the 4 M-star LAMOST data-driven age catalogue (10–25% for K dwarfs); and (iv) cluster plus Li plus rotation hierarchical models (gyro-interp + EAGLES). For a 2-billion-star catalogue, non-isochrone observables (periods for about 2 M stars, Li for about 1 M, X-ray for about 0.14 M, GALEX NUV for about tens of M) can serve only as label or validation sets. The anchor chain should be benchmark clusters, then wide binaries and asteroseismology, then field labels.

### Cited Findings
- **stardate (Angus et al. 2019, older):** combines gyrochronology with isochrone fitting for FGKM main-sequence and subgiant field stars. With 5% period uncertainties, main-sequence FGK ages become up to three times more precise than from isochrones alone. Gyro is optimal for cool main-sequence stars and isochrones near the turnoff — [arXiv:1908.07528](https://arxiv.org/pdf/1908.07528)
- **Gyro-kinematic ages** (Angus+2020; Lu+2021 with about 30,000 Kepler stars; Lu+2024 up to about 14 Gyr): use Galactic kinematics as a stellar-model-free age scale for old stars — [arXiv:2605.12647](https://arxiv.org/html/2605.12647); [arXiv:2310.14990](https://arxiv.org/pdf/2310.14990)
- **Cross-calibration through the age–velocity-dispersion relation:** [C/N] (giants) and gyro (dwarfs) ages placed on one scale. Gyro is valid for partially convective stars before WMB — [arXiv:2506.24010](https://arxiv.org/abs/2506.24010)
- **Wide-binary label transfer plus data-driven spectra:** 4 M LAMOST DR10 dwarfs, XGBoost, 10–25% precision for K dwarfs at S/N > 50, with signal from chemical clocks — [arXiv:2508.03019](https://arxiv.org/abs/2508.03019)
- **Rotation plus lithium:** Kepler field rotational ages (gyro-interp) validated by Li ages, agreeing above 90% — [arXiv:2410.06246](https://arxiv.org/abs/2410.06246)
- **Asteroseismic anchors:** median LEGACY uncertainties about 8% statistical and about 12% systematic. Gyro − asteroseismic offset about −13% below 6 Gyr and about −47% above. M dwarfs have no confirmed solar-like pulsators, and detection may need about 1 ppm (De Amorim+2026). Chemical clocks ([Y/Mg], [Y/Al]) reach about 1 Gyr for solar twins — [arXiv:2605.12647](https://arxiv.org/html/2605.12647)
- **Photometry-only route to rotation:** an empirical relation predicts stellar angular momentum from basic observables, without a period, to within 0.5 dex (3–500 Myr stars) — [arXiv:2405.08607](https://arxiv.org/abs/2405.08607)
- **Cluster-population gyro ages are much more precise than single-star ones:** ChronoFlow gives 0.08 dex total for clusters and 0.7 dex for individual stars — [arXiv:2412.12244](https://arxiv.org/abs/2412.12244)
- **Calibration risk from the absolute age scale:** benchmark-cluster ages scatter by up to about 50% (NGC 752 from 1.18 to 1.75 Gyr), and models overestimate G/early-K ages by 20–50% at 1.3 Gyr — [arXiv:2608.10077](https://arxiv.org/html/2608.10077v1)
- **Kinematic/demographic prior:** stars under 1 Gyr are 3–5× rarer than about 3 Gyr stars in the Kepler field (a "demographic cliff" from kinematic heating and a declining thin-disk SFR). This matters for priors and selection functions — [arXiv:2410.06246](https://arxiv.org/abs/2410.06246)
- **ML period pipelines with calibrated uncertainty:** conformalized quantile regression on Kepler periods ("Maunder") — [arXiv:2608.06604](https://arxiv.org/abs/2608.06604)

### Inferences
**Recommended label hierarchy for a data-driven Gaia dwarf-age model:**
1. **Gauge anchors:** about 10–30 benchmark clusters (α Per to M67), each with a free cluster-age parameter, plus LDB clusters (20–200 Myr), plus asteroseismic LEGACY and Kepler dwarfs (about 66 to a few hundred).
2. **Label transfer:** wide binaries, with a turnoff, subgiant or white-dwarf primary dated by isochrone or WD cooling and the dwarf secondary taking that age. This is the only scalable route that gives dwarfs ages over 4 Gyr without gyro or activity flattening.
3. **Field non-isochrone labels:** TARS, Kepler and ZTF periods converted with gyro-interp or ChronoFlow-type posteriors, used only where valid (Teff ≈ 3800–6200 K, 0.1–3 Gyr, RUWE < 1.4, not in a known binary). Add EAGLES Li ages from GALAH and LAMOST, R′HK from LAMOST, and L_X from eRASS1, each passed in as a likelihood with its own scatter rather than a point label. The Student-t multi-survey likelihood already used in this repo for spectroscopic labels is a natural fit.
4. **Gaia-wide inference:** with only XP plus UV–IR photometry, the age signal for dwarfs is weak. It would come from (a) metallicity and chemistry, through chemical clocks that XP only partially resolves; (b) GALEX NUV excess for young stars; (c) slight main-sequence evolution (luminosity above the ZAMS), which is effectively isochrone information; and (d) kinematics. Report broad posteriors and expect useful precision mainly for "young (<1 Gyr) versus old".

**Cross-validation diagnostics to build in:**
- Gyro versus asteroseismic offset as a function of age and Ro, to check whether WMB is modelled.
- Gyro versus Li agreement at under 1 Gyr (more than 90% expected).
- Recovery of benchmark cluster ages from member-level predictions.
- Wide-binary internal age consistency (|Δage| distribution).
- Recovery of the age–velocity-dispersion relation.
- Residuals against [Fe/H] at fixed Teff, which test the metallicity dependence of rotation and activity.

**Selection-function caution:** the period, X-ray and Li-detection label sets are all biased young. Weighting or explicit selection modelling is needed to avoid pulling Gaia-wide ages young. The roughly 3–5× young-star deficit in the field gives a check.

### Gaps
- No published Gaia-scale (≥100 M stars) dwarf age catalogue built from gyrochronology, activity or lithium labels was found; the largest found is LAMOST DR10's 4 M spectroscopic ages trained on wide binaries.
- How much age information Gaia XP spectra alone carry for dwarfs was not found in the sources searched. Precision for photometry-only (UV–IR) dwarfs is likewise unquantified.
- No study propagating ChronoFlow or gyro-interp posteriors into a neural-network training likelihood was found.
- Gaia DR4 (expected Dec 2026) rotation-period counts and its epoch photometry were not confirmed from Gaia documentation.
