# Stellar ages at Gaia scale (10^8 to 2x10^9 stars) and combining discordant age indicators into one calibrated scale (status as of 22 Nov 2026)

Scope note: research done 22 Nov 2026. Gaia DR4 is scheduled for 2 Dec 2026, so it was **not yet public** when these notes were written. Results before 2022 are marked "(older)". Some figures come from search-engine summaries of abstracts, not full texts; those are flagged where the attribution is uncertain.

## 1. Information content: where in the HR diagram do Gaia parallaxes + XP + photometry constrain age, and how precisely?

### Takeaway
Age information from L, Teff, [M/H] (what parallax + XP + photometry deliver) is concentrated at the main-sequence turn-off (MSTO) and especially the subgiant branch (SGB). There, spectroscopic-metallicity ages reach a realistic 5-10%. On the RGB and red clump, ages need a mass proxy and land at roughly 20-30% at best. On the unevolved lower main sequence, isochrones converge and the HRD carries essentially no age information. Grid-to-grid model systematics (about 20% for MS/SGB, sometimes over 80% on the RGB when mass is unknown) set the accuracy floor, not Gaia's parallax precision.

### Cited Findings
**HRD-region dependence (isochrone / photometric ages)**
- On the RGB and main sequence, isochrone fitting (SPInS on Gaia DR3 photometry + LAMOST DR8 abundances) "typically obtains very flat or ill-defined PDFs which are uninformative of the age of the star". Reliable isochrone ages come mainly from the turn-off and subgiant branch, where tracks of different masses are less degenerate — [SPInS / Gaia DR3 + LAMOST DR8 ages, arXiv:2410.15781](https://arxiv.org/pdf/2410.15781)
- The gain from parallax varies along the HRD. It is largest for lower-main-sequence stars and "smaller or close to zero for the main sequence turn-off region", where age depends more on Teff than on luminosity. On the lower MS, isochrones converge, so surface parameters "carry no useful age information" (older, 2018-2019) — [Howes et al., "Estimating stellar ages and metallicities from parallaxes and broadband photometry – successes and shortcomings", arXiv:1804.08321](https://arxiv.org/html/1804.08321.pdf); see also [Isochrone fitting in the Gaia era, arXiv:1804.06578](https://arxiv.org/pdf/1804.06578) (older)
- Even for the subgiant/turn-off benchmark stars with the most precise isochrone ages, the age "cannot be constrained to better than within 1 Gyr due to systematic uncertainties related to the use of different stellar models" (older, 2018) — [Benchmark ages for the Gaia benchmark stars, arXiv:1810.02829](https://arxiv.org/pdf/1810.02829)

**Realistic precision, empirically tested with wide binaries (2025)**
- A wide-binary consistency test (both components should share one age) found that subgiant ages from spectroscopic metallicities (Xiang & Rix 2022 catalogue) are generally consistent with their quoted errors, so **5-10% fractional age uncertainty is realistically achievable for subgiants**. Spectroscopic RGB and red-clump ages have reliable error bars but are **less precise, about 25-30%**. **Published photometric subgiant ages underestimate their true uncertainties by factors of 2-3**, and accurate abundances are "essential" — [How precisely can we measure the ages of subgiant and giant stars?, arXiv:2510.08675](https://arxiv.org/abs/2510.08675)
- A search summary attributes these figures to a 2025 study, possibly the paper above (attribution not verified against the full text): typical uncertainties of about 27% in age and 6% in mass for RGB stars, and about 19% in age and 9% in mass for red-clump stars — [arXiv:2510.08675 (ResearchGate copy)](https://www.researchgate.net/publication/396457868_How_precisely_can_we_measure_the_ages_of_subgiant_and_giant_stars)

**Model-systematics floor**
- Comparing widely used model grids gives about **5% systematic uncertainty in mass and about 20% in age** for main-sequence and subgiant stars. Interferometric angular diameters and bolometric fluxes set floors of about 2.4% in Teff, 2.0% in L and 4.2% in R. Open-source software (kiauhoku) is provided to estimate per-star grid systematics — [Tayar et al. 2022, ApJ 927, 31, arXiv:2012.07957](https://arxiv.org/abs/2012.07957); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10325927)
- Across four RGB model grids, ages can differ by **more than 80% when stellar mass is unknown** — [Morales, Tayar & Claytor 2025, ApJ 986, 229, "Model Choice Matters for Age Inference on the Red Giant Branch", arXiv:2504.17600](https://arxiv.org/html/2504.17600v1)
- A TESS asteroseismic study of **132,000 red giants** added an extra **10% systematic age uncertainty** for model choice — [Asteroseismically Inferred Ages of 132,000 Red Giants with TESS, arXiv:2505.08109](https://arxiv.org/pdf/2505.08109)

**What XP and RVS add beyond photometry (giants)**
- **SIDRA** (Almannaei et al., Dec 2024) is trained on BINGO (APOGEE-derived ages) for RGB and high-mass red-clump stars. On a held-out test set near tau = 10 Gyr, the residual scatter is **about 0.064 dex for SIDRA-XP** (input: stellar parameters derived from XP) and **about 0.12 dex for SIDRA-RVS** (input: RVS spectra). XP outperforms RVS for this task. Applied to **2,218,154 stars** — [arXiv:2412.09040](https://arxiv.org/abs/2412.09040)
- **XP generalist model** (Zhang, Bu et al., Mar 2026) is trained on XP spectra cross-matched with the APOGEE DR17 DistMass catalogue. It predicts Teff, log g, [M/H], mass and age, with test-set scatter of **sigma(M) about 0.114 Msun and sigma(age) about 1.334 Gyr** for evolved giants. It reproduces the giant-branch mass-luminosity relation — [A Generalist Model Including Evolved Star Mass and Age, arXiv:2603.03732](https://arxiv.org/abs/2603.03732)
- An all-sky catalogue of **6.5 million primary red-clump stars** (Yu & Chen, Aug 2026) uses spectroscopic estimates of asteroseismic parameters inferred from Gaia DR3 XP spectra. This shows XP carries information about seismic quantities and therefore mass — [summary page, arXiv:2608.28522](https://www.emergentmind.com/papers/2608.28522)
- On FLAME giant ages, the DR3 Apsis II paper says giant ages "are very dependent on their fitted mass". FLAME uses only L and Teff, which "suffers from significant degeneracies" — [Gaia DR3 Apsis II, arXiv:2206.05992](https://arxiv.org/pdf/2206.05992)

### Inferences
- For a Gaia-wide age map, the "age-informative" population is MSTO+SGB (5-10% if [M/H] and, ideally, [alpha/M] are good) plus giants with a mass proxy from XP, C/N-like or seismic-like features (about 0.06-0.12 dex statistical, 20-30% realistic). The lower main sequence needs non-HRD clocks (rotation, activity, kinematics) or population-level priors.
- XP adds age information for giants mainly through mass-sensitive features ([C/N]-like, log g/seismic proxies) and through [M/H]/[alpha/M] that break isochrone degeneracies. For MSTO/SGB stars, XP mainly helps through metallicity and extinction (the 2-3x error underestimate for photometric subgiant ages points to metallicity/extinction as the dominant missing information).
- Statistical precision from ML on XP (0.06 dex) is already below the model-scale accuracy floor (about 20%), so a calibrated scale has to report both a statistical term and a scale (anchor) term.

### Gaps
- No published formal Fisher-information (Cramér-Rao) study of age content in Gaia XP coefficients as a function of HRD position (2022-2026) was found. Evidence is empirical (ML test scatter, wide-binary tests).
- The often-quoted **7.5% median age uncertainty** for the Xiang & Rix (2022) subgiants could not be verified in this session (recalled, not confirmed).
- Quantitative age precision for stars with **only UV-to-IR photometry** (no parallax-quality spectra) was not found beyond the qualitative statements above. Howes et al. is pre-2022.

## 2. Existing Gaia-scale age catalogues: sizes, validation, failures

### Takeaway
Gaia-scale ages exist at about 10^6-10^7 stars with credible age information (StarHorse MSTO/SGB from about 1M RVS stars; SIDRA-XP 2.2M giants; NGBoost-XP 15.2M; Xiang & Rix about 250k subgiants). Gaia DR3 FLAME covers more stars, but its ages are prior-dominated in [M/H]/[alpha/Fe]. The recurring lessons:
- Ages are only trustworthy in the MSTO/SGB (and giants with mass proxies).
- Inadvertent priors can drive the oldest ages (the StarHorse corrigendum).
- Ages above about 9-10 Gyr tend to be underestimated.
- Every data-driven catalogue inherits the zero point of its training scale (astroNN, BINGO, DistMass).

### Cited Findings
**Gaia DR3 FLAME (Apsis)**
- FLAME derives ages and masses only from L and Teff, which "suffers from significant degeneracies", and the ages "rely heavily on the solar abundance assumption". Agreement with the literature is good for non-evolved stars, while giants are harder — [Gaia DR3 Apsis II, arXiv:2206.05992](https://arxiv.org/pdf/2206.05992)
- DR3 documentation validates FLAME ages against **three solar-metallicity open clusters** — [Gaia DR3 documentation, CU8 evolutionary-parameter QA](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_validation/ssec_cu8par_qa_evolution_aps.html)
- An OCA technical note on FLAME mass/age performance proposes linear corrections that require the true metallicity. Uncertainties of about ±0.5 Gyr remain, and giant ages "are typically always underestimated" — [OCA technical note (as returned by search; attribution of URL not double-checked)](https://dms.cosmos.esa.int/COSMOS/doc_fetch.php?id=1612899)
- A 2026 paper on the age of the Universe from the oldest Galactic stars notes that FLAME assumes solar [alpha/Fe] and "provides barely any constraints on the metallicity, making the prior completely dominant" — [arXiv:2607.00764](https://arxiv.org/pdf/2607.00764)
- One FLAME-asteroseismology comparison found a strong correlation, but on only **43 common stars**, with FLAME ages slightly overestimated (per search summary) — [Kepler meets Gaia DR3, arXiv:2501.18719](https://arxiv.org/pdf/2501.18719)
- A Gaia DR3 catalogue comparison found reliable ages for stars **younger than 9-10 Gyr when the relative uncertainty is under 50%**. Older stars' ages "tend to be underestimated" — [INAF open-access PDF (DR3 age catalogue)](https://openaccess.inaf.it/bitstreams/def49f70-4732-428d-bf22-9d5edba883dd/download)

**StarHorse 2023 (Queiroz, Anders et al.)**
- Distances and extinctions are given for **more than 10 million** stars in Gaia DR3 plus public spectroscopic surveys. Ages are given **only for MSTO+SGB stars**, because "age determination via isochrone fitting methods is not reliable outside of this regime". The Gaia RVS sample includes about **799,000 turn-off** and about **179,000 subgiant** stars — [A&A 673, A155 (2023)](https://www.aanda.org/articles/aa/full_html/2023/05/aa45399-22/aa45399-22.html); [AIP data page](https://data.aip.de/projects/aqueiroz2023.html)
- Typical age uncertainties are about **30% for MSTO+SGB**, falling to about **15% for SGB-only** stars (as used in a later paper) — [Super-metal-rich stars / young bar, arXiv:2311.16993](https://arxiv.org/html/2311.16993)
- **Corrigendum:** an unintended thin-disc star-formation-history prior (Mor et al. 2018) replaced the intended broad Gaussian prior. It "had an impact on the derived masses and ages of sub-giant and main-sequence turn-off stars of the thin disk, affecting mainly the oldest ages" — [Corrigendum, A&A 2023](https://www.aanda.org/articles/aa/full_html/2023/08/aa45399e-22/aa45399e-22.html)
- **SHBoost** (2024) transfers StarHorse/spectroscopic labels (Av, Teff, log g, [M/H], mass) to **217 million** Gaia DR3 XP stars. It provides **no ages**, according to its own table — [SHBoost, A&A 2024, arXiv:2407.06963](https://www.aanda.org/articles/aa/full_html/2024/11/aa51427-24/aa51427-24.html)

**Subgiant ages (Xiang & Rix 2022)**
- About **250,000 subgiants** from LAMOST + Gaia were dated with Bayesian fits to Yonsei-Yale isochrones, covering ages from about 1.5 to 13.8 Gyr. The result: the thick disc began forming about 13 Gyr ago, about 0.8 Gyr after the Big Bang — [Nature 603, 599 (2022), arXiv:2203.12110](https://arxiv.org/abs/2203.12110); [ADS](https://ui.adsabs.harvard.edu/abs/2022Natur.603..599X/abstract)
- A related 2022 study (Nataf et al.) used about 250,000 subgiants with *photometric* metallicities (per search summary) — [arXiv:2203.12110 search context](https://arxiv.org/pdf/2203.12110)

**XP-based parameter catalogues (inputs for ages, not ages themselves)**
- Teff, log g, [Fe/H], distances and extinctions for **220 million** XP stars (Zhang, Green & Rix 2023) — [MNRAS 524, 1855](https://academic.oup.com/mnras/article/524/2/1855/7209172)
- Systematic XP flux corrections and atmospheric parameters for **68 million** stars — [arXiv:2411.19105](https://arxiv.org/pdf/2411.19105)
- Metallicities for **20 million** giants from XP — [arXiv:2505.05281](https://arxiv.org/pdf/2505.05281)
- Millions of main-sequence binaries identified from XP (relevant as co-natal pairs and as contaminants of single-star ages) — [A&A, doi:10.1051/0004-6361/202556362](https://doi.org/10.1051/0004-6361/202556362)

**XP-based age catalogues (2024-2026)**
- **SIDRA-XP** gives ages for **2,218,154** giants (RGB + high-mass RC), with about 0.064 dex scatter at about 10 Gyr on its test set. Training scale: BINGO (APOGEE) — [arXiv:2412.09040](https://arxiv.org/abs/2412.09040)
- **NGBoost-XP** (Jul 2026) gives ages for **15,175,107 stars**. Inputs: Fallows & Sanders XP Teff, log g, [Fe/H], [C/Fe], [N/Fe], [alpha/M] with uncertainties. Training labels: astroNN APOGEE DR17 ages; the cross-match gives 489,913 stars, reduced to **135,569** after cuts (70/10/20 split).
  - The loss models training-label noise: total variance = model variance + label variance. Inputs are propagated with 200 Monte Carlo draws.
  - Test set: bias -0.119 Gyr, scatter **1.20 Gyr**. Against Miglio (Kepler seismic), Stone-Martinez (APOGEE NN) and Anders (APOGEE XGBoost), the scatter is **1.5-1.8 Gyr**, and NGBoost ages are **systematically younger**, which the authors attribute to the astroNN zero point.
  - No open-cluster validation. Training used only labels with age uncertainty below 2 Gyr — [arXiv:2607.17932](https://arxiv.org/html/2607.17932)
- **Generalist XP mass+age model** for giants (Mar 2026): sigma(age) about 1.33 Gyr — [arXiv:2603.03732](https://arxiv.org/abs/2603.03732)
- **XP red-clump catalogue** of 6.5 million stars from XP-inferred seismic parameters (Aug 2026) — [arXiv:2608.28522 summary](https://www.emergentmind.com/papers/2608.28522)

### Inferences
- The three failure modes to design against are:
  1. **Prior dominance**: FLAME [M/H]/[alpha/Fe]; the StarHorse SFH prior; the age-range truncation that compresses old ages.
  2. **Training-scale inheritance**: astroNN ages are systematically younger than Miglio/Anders. Each label-transfer catalogue sits on its parent's zero point.
  3. **Applicability-domain caps**: NGBoost covers 15M of about 220M XP stars. StarHorse gives ages only in the MSTO/SGB.
- A billion-star age product should therefore publish a per-star "age-informativeness" flag (posterior/prior contraction) and keep the training-scale identity as metadata.

### Gaps
- Exact FLAME DR3 counts (stars with flame_age and flame_age_spec) were not verified in this session.
- I found no XP-based catalogue that gives ages for MSTO/SGB dwarfs at about 10^8 scale. The existing XP age catalogues are giant-dominated (SIDRA) or limited to APOGEE-like parameter space (NGBoost).
- The Andrae et al. (2023) XP catalogue size was not re-verified here. It is cited only as an input to SoS II.

## 3. Statistical frameworks for combining multiple age indicators and catalogues into one calibrated scale

### Takeaway
Combination methods published by 2026 fall into five families:
1. Per-star joint likelihoods with hierarchical structure (stardate: isochrones + gyrochronology; chemical clocks; CHRONOS: lithium + cluster).
2. Hierarchical survey cross-calibration onto a reference scale with ML-learned corrections (Survey of Surveys DR3), the closest analogue to calibrating discordant *catalogues*.
3. Cross-calibrating indicators through a shared population relation (age-velocity dispersion).
4. Co-natal constraints (wide binaries, clusters) as model-independent consistency tests.
5. Explicit model-grid systematics terms (Tayar kiauhoku; added 10% systematic).

No published work found merges several *Gaia-scale age catalogues* with explicit per-catalogue offset/scale terms and conformal-calibrated uncertainties. That remains open.

### Cited Findings
**Per-star combination of indicators (hierarchical / Bayesian)**
- **stardate** (Angus et al. 2019, older) combines CMD/HRD isochrone placement with rotation periods (gyrochronology) in a hierarchical Bayesian model. Discrepancies between stardate and asteroseismic ages are "likely produced by differences between the MIST and BaSTI stellar evolution models", a direct example of model-scale systematics — [arXiv:1908.07528](https://arxiv.org/pdf/1908.07528)
- A hierarchical chemical-clock model combines several abundance-ratio clocks with Teff, [M/H] and log g (2022) — [A&A 2022, arXiv:2201.05228](https://www.aanda.org/articles/aa/full_html/2022/04/aa41125-21/aa41125-21.html)
- **CHRONOS I** (A&A, Jun 2026) is a hierarchical lithium-age model that jointly infers per-star Teff, A(Li) and a global cluster age, giving the Pleiades about **124.5 Myr**, consistent with lithium-depletion-boundary ages. It is motivated by "the lack of systematic cross-calibration between different stellar dating techniques (often yielding discrepancies of up to a factor of 2)". Its stated aim is a self-consistent absolute age scale that combines techniques in one hierarchical model — [A&A 2026, aa59708-26](https://www.aanda.org/articles/aa/full_html/2026/06/aa59708-26/aa59708-26.html); [arXiv:2605.25957](https://arxiv.org/pdf/2605.25957)
- The **Stellar Ages** algorithm applied to very young clusters (Dec 2025) combines per-star likelihoods hierarchically, so the most age-sensitive stars dominate. The paper quantifies systematic discrepancies between age tracers — [arXiv:2512.17033](https://arxiv.org/html/2512.17033v1)

**Cross-calibrating discordant surveys/catalogues onto one scale**
- **Survey of Surveys DR3** (Nov 2026) homogenises stellar parameters across **nine spectroscopic surveys**. It is hierarchical: first an internal reference scale built from APOGEE, GALAH and LAMOST-LRS, then ML regression of each survey's corrections onto that reference. GALAH/LAMOST Teff offsets relative to APOGEE fell from a **median 44.5 K to 6.2 K** — [arXiv:2611.07936](https://arxiv.org/pdf/2611.07936)
- **SoS II** (A&A 700, A195, 2025) gives parameters for about **23 million** stars. The spectroscopic part is calibrated externally against PASTEL high-resolution values (helpful for metal-poor giants), and Gaia DR3 and Andrae et al. (2023) parameters serve as starting points — [arXiv:2507.05901](https://arxiv.org/pdf/2507.05901); [A&A 2025](https://www.aanda.org/articles/aa/full_html/2025/08/aa55695-25/aa55695-25.html)
- A 2026 deep-learning model applied jointly to APOGEE DR17, GALAH DR3, DESI DR1 and Gaia RVS DR3 spectra reports cross-survey label agreement within model uncertainties — [arXiv:2604.25786](https://arxiv.org/html/2604.25786)
- **Training-label noise in age transfer:** the NGBoost-XP loss sets the label variance to model variance plus observational variance of the training age, and propagates input uncertainties by Monte Carlo. The authors found systematic offsets against other catalogues (1.5-1.8 Gyr scatter; younger zero point) — [arXiv:2607.17932](https://arxiv.org/html/2607.17932)

**Cross-calibrating indicators through population relations**
- [C/N] ages and gyrochronology ages were calibrated against the **same age-velocity-dispersion relation** (from APOKASC-3), then checked against asteroseismic benchmarks, wide binaries and open clusters. The goal is to minimise the bias between [C/N] and gyro ages — [arXiv:2506.24010](https://arxiv.org/html/2506.24010)

**Co-natal groups (clusters, wide binaries) as constraints**
- Wide binaries serve as a "model-independent benchmark" for age catalogues: both components must share an age. Spectroscopic subgiant errors pass the test, while photometric subgiant errors are underestimated by 2-3x — [arXiv:2510.08675](https://arxiv.org/abs/2510.08675)
- Gyrochronology constraints on **about 300 wide binaries** select period combinations that give consistent ages for both components, benchmarked on five open clusters (incl. the Pleiades) — [arXiv:2507.08266](https://arxiv.org/pdf/2507.08266)
- Wide binaries follow the same spin-down relation as open clusters. Some classic benchmarks (alpha Cen, 70 Oph) are now inconsistent, possibly because they are hierarchical systems (2023) — [A&A 675, A180, arXiv:2307.10836](https://arxiv.org/pdf/2307.10836)
- **Tension:** asteroseismic ages of individual red giants in clusters disagree with their host clusters' isochrone ages, which points to binary-interaction products or missing corrections — [Star-crossed Clusters, arXiv:2502.09582](https://arxiv.org/html/2502.09582)
- The Cool Stars 23 splinter summary (2025-2026) reports systematic offsets between lithium-depletion, gyrochronology and isochrone ages, and concludes that no single indicator is reliable without cross-calibration — [Open Clusters as Laboratories for Cool Star Evolution, arXiv:2609.09026](https://pith.science/paper/2609.09026)

**Quantifying model-scale (anchoring) systematics**
- Grid-to-grid comparison (about 20% age for MS/SGB) with software to compute per-star systematics — [Tayar et al. 2022](https://arxiv.org/abs/2012.07957)
- Over 80% RGB age differences across four grids when mass is unknown — [arXiv:2504.17600](https://arxiv.org/html/2504.17600v1)
- An added 10% model-systematic age term in a 132k-giant TESS catalogue — [arXiv:2505.08109](https://arxiv.org/pdf/2505.08109)

### Inferences
- The SoS DR3 recipe can be transplanted to ages directly. Choose a reference age scale (e.g., seismic APOKASC-3 for giants plus wide-binary/cluster-anchored subgiant isochrones), then learn per-catalogue correction functions of (Teff, log g, [M/H], [alpha/M], age) with smooth ML or GP regressors, fitted on overlap stars.
  - Co-natal constraints enter as likelihood terms: equal age for binary components and cluster members.
  - The cluster/wide-binary sets should be held out for validation of the calibrated uncertainties.
- A defensible error budget has three terms:
  1. statistical (per-star posterior, label-noise-aware);
  2. calibration (uncertainty of the learned per-catalogue map);
  3. anchor/model scale (spread across reference-scale choices, e.g., MIST vs PARSEC vs YY vs BaSTI, seismic scaling-relation corrections).

  The anchor term can be reported as a global or HRD-dependent multiplicative uncertainty (about 10-20%).
- Discordance between catalogues has a strong prior-driven component (FLAME metallicity prior; the StarHorse SFH prior). Before combining, catalogues should be re-weighted to a common prior, i.e. their likelihoods recovered by dividing out each catalogue's prior where it is documented. Otherwise priors are double-counted.

### Gaps
- No published application of **conformal prediction** or post-hoc calibration (e.g., coverage-checked quantile recalibration) to Gaia-scale stellar ages was found in this session. The NGBoost paper is the closest (label-noise-aware Gaussian), with no coverage test reported.
- No paper found that builds a "consensus age catalogue" merging FLAME + StarHorse + SIDRA + NGBoost + Xiang-Rix + seismic catalogues with explicit per-catalogue offset/scale terms.
- A 2025 Gaussian-process gyrochronology calibration (reported via search as using **185 white dwarf + main-sequence wide binaries** plus clusters, valid over about 0.67-14 Gyr, about 1 Gyr cross-validated uncertainty) could not be tied to a specific URL in this session, so it is listed here only as unverified.

## 4. Upcoming data: Gaia DR4/DR5, LSST, PLATO, Roman, 4MOST, WEAVE, SDSS-V

### Takeaway
Gaia DR4 (**2 Dec 2026**, 66 months of data, about 2.8 billion sources) is the step change. It brings:
- epoch photometry (about 122 billion rows);
- mean XP spectra for **about 1.82 billion** sources plus about 121 billion epoch XP spectra;
- RVS mean spectra for **250 million**;
- **18.2 million** rotational-modulation stars;
- Apsis XP parameters for about **1.0 billion** sources.

DR5 is not before end of 2030. PLATO (launch early 2027, first data about 2028) and Roman (launch window fall 2026 to May 2027, about 290k-600k bulge seismic giants) will provide new asteroseismic anchors. SDSS-V MWM (DR20, Jul/Aug 2026) and 4MOST/WEAVE supply spectroscopic labels for transfer.

### Cited Findings
**Gaia DR4 (scheduled, not yet released as of 22 Nov 2026)**
- ESA announced (7 May 2026) DR4 for **Wednesday 2 December 2026**. It is based on 66 months (5.5 yr) of data and for the first time contains all intended Gaia data products, for about **2.8 billion sources**. A small pre-release came out in June 2026 with a draft data model. **DR5 is "not before the end of 2030"** — [Gaia news 2026](https://www.cosmos.esa.int/web/gaia/news-2026); [Gaia release scenario](https://www.cosmos.esa.int/web/gaia/release); [DR4 page](https://www.cosmos.esa.int/en/web/gaia/data-release-4)
- DR4 contents table (page updated 14 Sep 2026; marked "under development"):
  - Data span: 25 Jul 2014 to 20 Jan 2020; reference epoch J2017.5; about 400 TB (DR3: 10 TB).
  - `gaia_source`: **2,000,033,276** rows; `all_source_astrometry` / `all_source_photometry`: 2,793,009,568 rows.
  - `epoch_photometry`: **121,845,163,512** rows.
  - Mean XP (`xp_continuous_mean_spectrum`): **1,824,229,813**; epoch XP (`xp_epoch_spectrum`): **121,070,897,406** rows (119 TB).
  - `rvs_mean_spectrum`: **250,022,127**; `all_source_rvs`: 312,247,580; `rvs_epoch_spectrum`: 6.91 billion.
  - `vari_summary`: 403,061,676; `vari_rotation_modulation`: **18,237,848**; `vari_eclipsing_binary`: 2,392,500; `vari_long_period_variable`: 4,437,454; `vari_rrlyrae`: 246,380.
  - `ap_xp`: **999,939,280**; `ap_rvs` and `ap_chem`: 35,825,564; `ap_xp_epoch`: 355,690. FLAME contributes to `ap_rvs` and `ap_xp` (no separate age count given).

  — [Gaia DR4 contents](https://www.cosmos.esa.int/web/gaia/dr4)

**PLATO**
- Launch on Ariane 6. A 2026 PLATO team presentation gives **2027** (start of nominal operations 2027, first calibrated data **2028**). A late-2026 arXiv paper gives **January 2027**, and an earlier Springer overview said end of 2026 — [PLATO team presentation (Cabrera, 2026)](https://indico.ict.inaf.it/event/3326/sessions/4663/attachments/12151/25372/01_Cabrera.pdf); [Exp. Astron. 2025 overview](https://link.springer.com/article/10.1007/s10686-025-09985-9); [Data-Driven Ages from Asteroseismology for FGK Dwarfs, arXiv:2610.28345](https://arxiv.org/pdf/2610.28345)
- Mission goals: masses to about 10%, radii to 1-2%, **ages to about 10%** for bright solar-like oscillators — [eoPortal PLATO](https://www.eoportal.org/satellite-missions/plato); predicted solar-like-oscillator detection yields in [arXiv:2401.07984](https://arxiv.org/pdf/2401.07984); Galactic-archaeology legacy case (older) in [arXiv:1706.03778](https://arxiv.org/pdf/1706.03778)

**Roman Galactic Bulge Time Domain Survey (GBTDS)**
- NASA: Roman launches **no later than May 2027**, possibly **as early as fall 2026**. The GBTDS revisits the bulge every **12 minutes** over six **70.5-day** seasons — [NASA Roman GBTDS asteroseismology](https://science.nasa.gov/mission/roman-space-telescope/asteroseismology-using-the-galactic-bulge-time-domain-survey)
- Yields: Weiss et al. 2025 give a lower limit of **about 290,000** seismic red giants and red-clump stars (**185,000 in the bulge**). A July 2026 slide summarises about 300,000 (about 200,000 bulge) and about 500,000 bulge RGB seismic candidates from VVV and 2MASS×Gaia. A 2023 projection was about 6×10^5 — [arXiv:2503.04999](https://arxiv.org/html/2503.04999v3); [Weiss, Roman 2026 conference slides](https://conference.ipac.caltech.edu/roman2026/files/TREVOR_WEISS_roman2026.pdf); [Huber et al. 2023, arXiv:2307.03237](https://arxiv.org/abs/2307.03237)

**LSST (Rubin)**
- Simulations (older, LSST Science Book 2009): periods are efficiently recoverable for G, K and early-M dwarfs with **r ≤ 20 and ages ≲ 2 Gyr**; 250 Myr solar analogues are measurable between 1 and 20 kpc — [LSST Science Book v2.0, arXiv:0912.0201](https://arxiv.org/pdf/0912.0201); [LSST Stars study group report (2016), arXiv:1607.04302](https://arxiv.org/pdf/1607.04302)
- Rubin First Look: **23 June 2025** (earliest published Rubin rotation-period work concerns asteroids) — [Rubin First Look asteroid lightcurves](https://pure.psu.edu/en/publications/lightcurves-rotation-periods-and-colors-for-vera-c-rubin-observat/)

**SDSS-V Milky Way Mapper**
- **DR19** — [arXiv:2507.07093](https://arxiv.org/pdf/2507.07093)
- **DR20** (Jul 2026): first all-sky BOSS spectra. OB targets rose from 38k (102k spectra) in DR19 to 175k (432k spectra); MWM targets total 1.2M excluding standards — [arXiv:2607.26149](https://arxiv.org/pdf/2607.26149)
- Press coverage (Aug 2026) cites "more than 2 million total spectra" — [phys.org, Aug 2026](https://phys.org/news/2026-08-galaxy-milky-mapper-survey-latest.html)
- MWM design goal: about 5-6 million stars; luminous red giants cover ages from about 1 Gyr upward — [SDSS DR18 MWM about](https://www.sdss.org/dr18/mwm/about/)

**4MOST / WEAVE**
- 4MOST design: within 5 years, cover most of the southern sky in 2-3 epochs, with spectra for millions of Gaia targets (4MIDABLE-LR disc/bulge survey) — [4MOST Survey 3, arXiv:1903.02469](https://arxiv.org/pdf/1903.02469)

**Current Gaia rotation (bridge to gyrochronology)**
- DR4 will carry **18.2 million** solar-like rotational-modulation stars (see contents table above) — [Gaia DR4 contents](https://www.cosmos.esa.int/web/gaia/dr4)

### Inferences
- DR4 epoch photometry and epoch XP for about 1.8 billion stars allow:
  - (i) rotation and activity clocks for the age-poor lower main sequence (18.2M rotators, about 50x more than any previous homogeneous set — DR3 count not verified here);
  - (ii) better binary flagging from photometric/RV/astrometric variability, a key systematic in single-star isochrone ages;
  - (iii) higher-S/N mean XP (66 vs 34 months), improving [M/H], [alpha/M] and mass proxies, the dominant limits on MSTO/SGB and giant ages.
- The DR4 `ap_rvs` set (about 36M with chemistry) is an internal Gaia label set large enough to train XP→age transfer on Gaia-only labels, reducing cross-survey zero-point issues.
- PLATO first data (about 2028) and Roman GBTDS (about 2027+) give new seismic anchor scales: dwarfs/subgiants (PLATO) and bulge giants (Roman). These will make the anchor-scale term in Section 3 measurable for regions currently lacking anchors.

### Gaps
- **The LSST 10-year survey start date** and any Rubin stellar rotation catalogues (2025-2026) could not be confirmed in this session.
- No 2026 operational status or star counts were found for **4MOST** or **WEAVE**.
- Whether **Roman** launched in fall 2026 could not be confirmed (only the "as early as fall 2026 / no later than May 2027" window).
- The Gaia DR3 rotation-period count (Distefano et al. 2023) was not re-verified.
- Specific PLATO seismic-yield numbers from arXiv:2401.07984 were not extracted.
- Whether DR4 FLAME will provide ages for all `ap_xp` sources, and with what priors, is not documented on the contents page.

## 5. Science uses at scale: required accuracy/precision, and pitfalls (selection effects)

### Takeaway
Ages around 10% for subgiants (Xiang & Rix) were enough to time the thick-disc onset at about 13 Gyr and resolve early Milky Way assembly. Giant ages at 25-30% (about 0.06-0.12 dex statistical) support thin/thick-disc and radial-migration trends but not fine chronology at old ages. Old-age compression (underestimates above about 9-10 Gyr), priors and training-scale offsets are the dominant pitfalls. The search found no quantitative 2022-2026 requirement study, such as "X% precision needed to separate thin/thick disc", or any formal treatment of selection effects in Gaia-scale age distributions.

### Cited Findings
- About 250k subgiant ages (Xiang & Rix) showed the thick disc began forming about 13 Gyr ago, about 0.8 Gyr after the Big Bang — [arXiv:2203.12110](https://arxiv.org/abs/2203.12110)
- StarHorse MSTO/SGB ages (about 30%, 15% for SGB) were used for chrono-chemical populations, the "genuine thick disk" and young alpha-rich stars. Applied to super-metal-rich stars, they gave a constraint on bar age — [A&A 673, A155](https://www.aanda.org/articles/aa/full_html/2023/05/aa45399-22/aa45399-22.html); [arXiv:2311.16993](https://arxiv.org/html/2311.16993)
- Pitfall, old-age compression: ages are reliable below about 9-10 Gyr when relative uncertainty is under 50%; older ages are underestimated — [INAF DR3 age catalogue PDF](https://openaccess.inaf.it/bitstreams/def49f70-4732-428d-bf22-9d5edba883dd/download)
- Pitfall, prior dominance: FLAME's metallicity prior is "completely dominant" — [arXiv:2607.00764](https://arxiv.org/pdf/2607.00764). The StarHorse SFH prior affected the oldest thin-disc ages — [Corrigendum](https://www.aanda.org/articles/aa/full_html/2023/08/aa45399e-22/aa45399e-22.html)
- Pitfall, training-scale inheritance: NGBoost-XP ages are systematically younger than seismic/APOGEE alternatives because of the astroNN zero point — [arXiv:2607.17932](https://arxiv.org/html/2607.17932)
- Pitfall, cluster vs single-star seismic tension, likely from binary products — [arXiv:2502.09582](https://arxiv.org/html/2502.09582)
- XP Galactic-disc structure studies (2026) show the scale of XP-based population mapping — [arXiv:2610.15007](https://arxiv.org/pdf/2610.15007); a 6D homogenised Gaia DR3 tracer catalogue to 250 kpc — [arXiv:2609.28652](https://arxiv.org/pdf/2609.28652)

### Inferences
- Selection effects arise from three sources:
  1. Age-informative subsets (MSTO/SGB, giants with mass proxies) are mass- and lifetime-selected: subgiants oversample by evolutionary-phase duration, and giants are biased toward stars above about 1 Msun at younger ages.
  2. Applicability cuts in ML catalogues (e.g., NGBoost's parameter range, uncertainty below 2 Gyr in training) carve the HRD non-uniformly.
  3. Magnitude limits (XP G < 17.65 in DR3) create distance-dependent HRD coverage.

  Population age distributions therefore need an explicit selection function (lifetime weighting × catalogue applicability × survey completeness) before being interpreted as star-formation histories.
- For halo/outer-halo work (old, metal-poor, alpha-enhanced), solar-[alpha/Fe] assumptions (FLAME) and training sets dominated by disc stars (APOGEE) give systematically biased old ages. Accuracy at old ages is limited by the anchor scale (about 1 Gyr at 12 Gyr from model systematics, per the benchmark-star result) more than by precision.

### Gaps
- No quantitative 2022-2026 requirement analysis was found that links age precision to specific Milky Way science goals (thin/thick disc separation, halo assembly timing, cluster discovery via age clustering).
- No dedicated 2022-2026 paper on selection-function modelling for Gaia-scale age distributions was found in the searches made.
