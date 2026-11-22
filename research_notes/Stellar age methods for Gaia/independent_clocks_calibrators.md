# Independent stellar clocks and age calibrators for anchoring or validating a consensus age scale (status as of Nov 2026)

Scope note: these notes cover age calibrators that are partly or fully independent of the main-sequence (MS) and giant isochrone or asteroseismic scales: white-dwarf (WD) cooling ages, cluster ages (LDB, MSTO, eclipsing binaries, globular clusters), chemical clocks, kinematic ages, nucleocosmochronology, and co-eval wide binaries. Results from before 2020 are marked "[older]". Items marked "[preprint]" are arXiv-only as far as I could check. Asteroseismology is covered by other researchers. It appears here only where it is used to calibrate another clock.

---

## 1. White-dwarf cooling ages and WD + MS wide binaries

### Takeaway
WD cooling physics does not depend on MS or giant models, and Gaia DR3 gives roughly 359,000 high-probability WDs. But a *total* age (cooling age + progenitor lifetime) needs an initial-final mass relation (IFMR) and a progenitor MS lifetime. These inputs keep total ages reliable only for WDs above ~0.63 Msun, cooler than ~12,000 K and outside the crystallisation, distillation and merger anomalies. In that regime about 25% total-age uncertainties are realistic. There are ~16,000 WD+MS wide pairs in El-Badry et al. (2021). Only a minority have both a usable WD age and a spectroscopic companion.

### Cited Findings
**Catalogues and sample sizes**
- Gaia DR2 yielded >260,000 WDs. DR3 raised this to >355,000. The number of high-probability candidates rose 38% from DR2 to DR3 — [Gaia WD revolution review, arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- Gentile Fusillo et al. (2021) list ~359,000 high-probability WD candidates (P_WD-based) out of 1,280,266 candidates in their EDR3 catalogue. Estimated completeness is 67–93% for G < 20. Only ~10% of DR3 candidates have medium-resolution spectroscopic follow-up — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- The 40 pc sample (O'Brien et al. 2024) has 1,076 spectroscopically confirmed WDs out of 1,081 candidates (~99% complete in spectral type). At least ~40 more WDs in this volume were missed, mostly in unresolved WD+MS binaries. Completeness within 40 pc is >97% — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- El-Badry et al. (2021) give ~1,400 WD+WD and ~16,000 WD+MS wide binaries. Rebassa-Mansergas et al. (2021b) found 112 WD+MS candidates within 100 pc, estimated at ~10% of the underlying population — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)

**Precision and validity regime (WD+WD consistency tests)**
- Heintz et al. (2022, ApJ 934, 148) built 1,565 wide WD+WD pairs plus 24 triples from Gaia EDR3. Among 423 high-fidelity pairs, 25% total-age uncertainties are generally appropriate for WDs >0.63 Msun and <12,000 K. Total ages are least reliable for WDs <0.63 Msun, whose progenitors spent a long time on the MS — [Heintz et al. 2022, ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ac78d9) / [arXiv:2206.00025](https://arxiv.org/abs/2206.00025)
- In the same work, total age depends heavily on the chosen IFMR. Small mass errors in low-mass WDs become large errors in progenitor lifetime and total age. Adding spectroscopic metallicities from 38 wide MS companions substantially reduces internal total-age uncertainties compared with assuming a uniform metallicity — [arXiv:2206.00025](https://arxiv.org/html/2206.00025)
- A 2024 test of *spectroscopic* WD ages with wide WD+WD binaries notes that the wide-binary method cannot be applied widely to field stars (especially cool MS stars), but works as a calibrator for other techniques — [arXiv:2405.02423](https://arxiv.org/pdf/2405.02423)
- Large scatter remains between WD-based and MS-based ages in WD+MS and WD+WD samples (Qiu et al. 2021; Heintz et al. 2024). Calibrations of age relations with WD companions include Fouesneau et al. (2019) [older], Rebassa-Mansergas et al. (2021a, 2023), Heintz et al. (2022), Moss et al. (2022) and Zhang et al. (2023) — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)

**Cooling-model systematics (by WD mass and age)**
- Crystallisation (latent heat plus phase separation) delays cooling by ~1 Gyr. Observations suggest an extra ~1 Gyr delay beyond that — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- Q-branch kinematics (Cheng et al. 2019) imply an extra cooling delay of ≳8 Gyr for 5–9% of WDs with 1.08–1.23 Msun. 22Ne/26Mg distillation can halt cooling for ~10 Gyr in ultramassive WDs. About 20% of massive WDs (0.8–1.3 Msun) may be WD+WD merger products, which makes their single-star cooling ages too young — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- Spectroscopic Teff is 2–5% higher and log g 0.02–0.05 dex higher than Gaia photometric values. Photometric masses near 4,000 K are ~20% too low (the "low-mass problem", e.g. Cunningham et al. 2024) — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- Empirical IFMRs come from Gaia 100 pc WDs (El-Badry et al. 2018 [older]), the 40 pc sample (Cunningham et al. 2024), wide binaries (Barrientos & Chanamé 2021) and clusters (Marigo et al. 2020). Intrinsic IFMR scatter is ~5% (Hollands et al. 2024). The median IFMR is statistically consistent across most studies and masses — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)

**Population-level WD cosmochronology**
- The 40 pc absolute-magnitude distribution is consistent with constant star formation over ~10.6 Gyr (Cukanovaite et al. 2023). Inner-halo WD ages are 10.9 ± 0.4 Gyr (Kilic et al. 2019 [older]) and 12 ± 0.5 Gyr (Torres et al. 2021). Halo WD ages remain sensitive to stellar-evolution systematics — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)

**WD+MS binaries applied to the age–metallicity relation (AMR)**
- Rebassa-Mansergas et al. (2021) used 46 unresolved SDSS WD+MS binaries plus 189 Gaia DR2 common-proper-motion pairs. They found a flat AMR with ~±0.5 dex [Fe/H] scatter at all ages — [IAC summary of arXiv:2105.13379](https://iac.es/es/ciencia-y-tecnologia/publicaciones/constraining-solar-neighbourhood-age-metallicity-relation-white-dwarf-main-sequence)
- Rebassa-Mansergas, Tejero-Gómez & Raddi (2026, A&A 709, A250) use Gaia DR3 wide WD+MS binaries. WD ages come from interpolating Gaia M_G and BP−RP on state-of-the-art cooling sequences (La Plata). [Fe/H] comes from literature surveys including LAMOST. They find a large [Fe/H] dispersion at all ages, consistent with earlier WD+MS and isolated-star results. The abstract does not give the sample size or age uncertainty — [arXiv:2604.15939](https://arxiv.org/abs/2604.15939); [A&A](https://doi.org/10.1051/0004-6361/202660074)

### Inferences
- **Model dependence.** The cooling clock is independent of MS/giant models. The *progenitor-lifetime* term reintroduces MS-model dependence. For WDs >0.63 Msun (progenitors ≳2–3 Msun) that term is a small fraction of the total age for systems older than ~1–2 Gyr. So high-mass WDs in wide binaries are the least model-dependent field-star anchor for the 1–10 Gyr range.
- **Best use.** Use the cooling ages as an *accuracy anchor* on a data-driven age scale, especially for MS dwarfs (K/M), where isochrone ages are nearly uninformative. Restrict to 0.63 ≲ M_WD ≲ 1.0 Msun and Teff ≲ 12,000 K. Exclude Q-branch and ultramassive objects. Propagate the ~25% total-age error and the ~5% IFMR scatter.
- **Expected anchor count.** From ~16,000 WD+MS pairs, after the WD mass/Teff cuts and requiring a companion with survey spectroscopy, the usable anchor sample is plausibly a few hundred to ~1–2 thousand. This is my estimate, not a published number.
- **Complementarity.** WD ages are weakest at <0.63 Msun and at young ages (cooling age ≪ progenitor lifetime), where clusters (LDB/MSTO) are strongest.

### Gaps
- Exact sample sizes and per-object total-age precision of Fouesneau et al. (2019), Qiu et al. (2021) and Rebassa-Mansergas et al. (2023, 2026) were not retrievable (A&A full text returned HTTP 403).
- I found no single published count of "WD+MS pairs with a high-quality WD total age plus companion spectroscopy" in Gaia DR3. Post-2021 WD+MS wide-binary catalogues (e.g. DR3 updates) were not checked.
- A quantitative comparison of cooling-model grids (La Plata vs Montreal vs BaSTI-IAC) in Gyr was not found in the visible text.

---

## 2. Star clusters: LDB, MSTO, eclipsing binaries, homogeneous catalogues, globular clusters

### Takeaway
LDB ages (~20–200 Myr) and eclipsing-binary-constrained ages in a handful of old open clusters are the most robust cluster anchors. NGC 6791 is now at 8.46 ± 0.66 Gyr with systematics marginalised. Homogeneous Gaia MSTO catalogues (Cantat-Gaudin+2020, Dias+2021, Hunt & Reffert 2023) cover thousands of clusters. They are all PARSEC-based with constrained or fixed metallicity, and they disagree at the 0.1–0.26 dex level. Globular-cluster absolute ages are now ~0.5–0.75 Gyr (4–6%) precise, limited by distance and reddening.

### Cited Findings
**Lithium depletion boundary (LDB)**
- Galindo-Guil et al. (2022, A&A 664, A70) homogeneously re-derived LDB ages for 9 open clusters and 3 moving groups. They used Gaia parallaxes and bolometric luminosities that do not rely on monochromatic bolometric corrections, and located the LDB with one method for all associations — [ADS](https://ui.adsabs.harvard.edu/abs/2022A&A...664A..70G/abstract); [arXiv:2205.06354](https://arxiv.org/abs/2205.06354)
- EAGLES (Jeffries et al. 2023, MNRAS 523, 802) is an empirical model of age-dependent Li depletion calibrated on 6,200 stars in 52 open clusters, homogeneously analysed by Gaia-ESO. Its young-cluster ages agree with the training ages and with several published model-insensitive LDB ages — [MNRAS](https://dx.doi.org/10.1093/mnras/stad1293); [ADS](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523..802J)
- IC 4665 is a known discordant case. Its low-mass Gaia CMD positions and K/M-star Li depletion are reported as inconsistent with the published LDB age of 32 (+4/−5) Myr. This comes from a ResearchGate copy of the Gaia-ESO paper, so treat it with caution — [ResearchGate](https://www.researchgate.net/publication/361935144_Lithium_depletion_boundary_stellar_associations_and_Gaia)
- In a 2026 study of MS+MS wide binaries (877,416 high-bound-probability pairs from El-Badry+2021), Teff governs Li depletion while rotation plays a minor role — [arXiv:2603.18603](https://arxiv.org/pdf/2603.18603) [preprint]

**MSTO and homogeneous Gaia cluster catalogues: inter-catalogue systematics**
- Hunt & Reffert (2023) used a neural-network isochrone fitter on Gaia DR3 with PARSEC models (ages, distances, extinctions; no metallicity reported). They validated clusters with a density test plus a Bayesian CNN CMD classifier, and drew 1,000 network samples per cluster — [arXiv:2303.13424](https://arxiv.org/pdf/2303.13424); summarised in [arXiv:2605.23802](https://arxiv.org/html/2605.23802)
- A 2026 Bayesian nested-sampling PARSEC analysis of 5,056 open clusters (free metallicity, Gaia XP priors for 1,201 clusters) gives Δlog(age) = ref − this work [preprint]. For its high-quality subset (η_fit ≥ 0.80) — [arXiv:2605.23802](https://arxiv.org/html/2605.23802):
  - vs Cantat-Gaudin+2020: +0.022 ± 0.259 dex (MAD 0.154, N = 1,230)
  - vs Dias+2021: +0.007 ± 0.247 dex (MAD 0.167, N = 1,045)
  - vs Hunt & Reffert 2023: −0.098 ± 0.203 dex (MAD 0.096, N = 2,337). HR23 is ~0.1 dex younger on average.
- The same paper states that Cantat-Gaudin+2020 fixes [Fe/H] at solar, Dias+2021 uses a position-dependent abundance gradient prior, and HR23 works within a fixed metallicity range. All use PARSEC. The CG20 offset is linked to ANN-fitting limits and DR2→DR3 astrometry and disappears for well-populated clusters. HR23 extinctions carry a documented ~0.4–0.5 mag systematic difference attributed to extinction definitions. The paper does *not* analyse age-dependent (old vs young) offsets — [arXiv:2605.23802](https://arxiv.org/html/2605.23802)

**Eclipsing binaries (EBs) in clusters**
- DEBCat (Southworth) lists detached EBs with both masses and radii measured to ≤2%, without significant stellar-theory input. It is almost entirely double-lined systems. It had 171 objects in 2014 and >195 in Oct 2017. I found no newer count in the search results — [arXiv:1411.1219](https://arxiv.org/pdf/1411.1219); [VizieR V/152](https://ui.adsabs.harvard.edu/abs/2017yCat.5152....0S)
- NGC 6791 [older]: Brogaard et al. (2012) used multiple EB members and found an age of ~8.3 Gyr. The remaining age uncertainty is dominated by CNO abundances — [A&A 543, A106](https://www.aanda.org/articles/aa/pdf/2012/07/aa19196-12.pdf). Brogaard et al. (2021) used the subgiant V56 to support that age, with ±0.3 Gyr from fitting only — [A&A 2021](https://www.aanda.org/articles/aa/full_html/2021/05/aa40911-21/aa40911-21.html)
- NGC 6791 (2026): Dufresne, Chaboyer & Rampalli combine Gaia DR3 CMD morphology (2D KS statistic) with coeval DEB constraints in (M, L), using 10,000 Monte Carlo isochrone sets. They marginalise over convection, opacities, diffusion, nuclear rates, distance, reddening and composition. Result: **8.46 ± 0.66 Gyr**, [Fe/H] = +0.280 ± 0.079, Y = 0.2968 ± 0.0158, (m−M)V = 13.333 ± 0.058, E(B−V) = 0.183 ± 0.024. They describe it as a benchmark for high-metallicity asteroseismic calibration — [arXiv:2605.21682](https://arxiv.org/abs/2605.21682) (ApJ version: [IOP](https://iopscience.iop.org/article/10.3847/1538-4357/ae6fa3)). One search summary quoted +0.61/−0.52 Gyr instead of ±0.66, possibly from a different version. Use ±0.66 from the abstract page.
- NGC 6819 lies in the Kepler field. The Kepler IRIS catalogue gives image-subtraction light curves for 9,150 stars in and around NGC 6791 and NGC 6819 — [arXiv:2112.05174](https://arxiv.org/pdf/2112.05174)

**Globular-cluster absolute ages**
- M92: Ying et al. (2023, AJ) give **13.80 ± 0.75 Gyr** (5.4%). They used 20,000 Monte Carlo DSEP isochrone sets and Voronoi-binned Hess-diagram fits to HST ACS data. Distance is ~80% of the error budget. Metallicity, [α/Fe] and helium-diffusion treatment are the other main terms — [arXiv:2306.02180](https://arxiv.org/pdf/2306.02180)
- Ying et al. (2025) analyse 8 Milky Way globular clusters ([Fe/H] = −2.31 to −0.77) and find absolute ages of ~11.5–13.5 Gyr with typical errors of 0.5–0.75 Gyr. Distance plus reddening is >50% of each error budget. The main stellar-physics terms are metal content, [α/Fe], mixing length and helium diffusion. Ages are clearly older at lower metallicity — [arXiv:2505.02969](https://arxiv.org/abs/2505.02969)

### Inferences
- **Model-independence ranking within clusters.** LDB is the most robust: it mainly needs the luminosity at which Li survives, plus Gaia distances. Next come EB-constrained isochrone ages, where the models are still used but the EBs pin down mass, helium and mixing. MSTO ages from homogeneous catalogues come last: they are fully isochrone-dependent and carry metallicity/extinction degeneracies.
- **Using catalogues against a data-driven scale.** Cluster members give coeval groups. The internal age dispersion a data-driven scale assigns to members is a direct *precision* test that does not depend on the catalogue's absolute ages. *Accuracy* tests should use LDB clusters (young end) plus EB clusters (NGC 6791 and other old EB clusters) as fixed points. The ~0.1 dex HR23 vs other-catalogue offset should be treated as the floor of systematic uncertainty when catalogue MSTO ages are the labels.
- **Old clusters.** Fixed or solar metallicity in CG20 and HR23 will bias ages for metal-rich old clusters such as NGC 6791 ([Fe/H] = +0.28). Prefer EB- or spectroscopy-informed ages for >3 Gyr clusters over catalogue ages.
- **Globular clusters** provide an old-end anchor (11.5–13.8 Gyr, 4–6%) for halo or thick-disc metal-poor populations. Their ages still rely on DSEP isochrones. They are mainly useful to check that a data-driven scale does not exceed ~14 Gyr and that it ranks metal-poor populations as oldest.

### Gaps
- Cluster-by-cluster LDB ages and precisions (e.g. Pleiades, α Per, IC 2391, IC 2602, NGC 2547, Blanco 1, β Pic MG) were not extracted from Galindo-Guil et al. (2022).
- I found no published, quantitative age-dependent bias analysis of Hunt & Reffert (2023/2024) for old (>1–3 Gyr) clusters. Hunt & Reffert (2024) cleaned-catalogue age changes were not checked. A "30–50% relative age error" figure appeared in search snippets but could not be traced to a primary source.
- No eclipsing-binary age with uncertainties was retrieved for NGC 6819, NGC 188, NGC 752 or M67 in this session. Their EB literature (e.g. Jeffries/Sandquist/Brogaard) still needs extracting.
- VandenBerg et al. (2013, ApJ 775, 134; ~55 GCs, relative ages) was not re-verified here. My recollection is ages of ~11.75 Gyr for metal-poor GCs with ±0.25–0.5 Gyr relative uncertainties, but this is not confirmed in this session.
- The current DEBCat count (2026) and the number of DEBCat systems in clusters were not found.

---

## 3. Chemical clocks: [Y/Mg], [C/N], [Ba/Fe] and others

### Takeaway
Chemical clocks are empirical and cheap, which makes them attractive as features or labels for a data-driven scale. But they are *calibrated on* isochrone, asteroseismic or cluster ages, so they inherit that age scale. They are also population-dependent. [Y/Mg] reaches ~0.45–1 Gyr precision for solar-type disc dwarfs, with explicit [Fe/H] terms, but recent NLTE and asteroseismic work says it is not universal. [C/N] in lower-RGB giants gives ~30% ages over 2–10 Gyr for [Fe/H] > −1.

### Cited Findings
**[Y/Mg] (s-process vs α)**
- Shejeelammal, Meléndez, Rathsam & Martos (2024, A&A) studied 48 metal-poor solar-type stars (HARPS) plus 185 solar twins and analogues, covering −0.71 ≤ [Fe/H] < +0.34. Ages come from Gaia-based isochrone fits, validated with open clusters and asteroseismology. The [Y/Mg]–age anticorrelation is strong and metallicity-dependent, and is similar for thin- and thick-disc stars. Accuracy is 0.45 Gyr and precision 0.99 Gyr — [arXiv:2407.07283](https://arxiv.org/abs/2407.07283); [A&A full text](https://www.aanda.org/articles/aa/full_html/2024/10/aa49669-24/aa49669-24.html)
  - Low-[Fe/H] linear fit: [Y/Mg] = −0.0362(±0.0004) × Age(Gyr) + 0.0867(±0.0285) for −0.71 ≤ [Fe/H] < −0.17. The slope is ~−0.034 Gyr⁻¹ above that.
  - Multivariate: Age(Gyr) = −27.701(±0.461) [Y/Mg] + 3.454(±0.329) [Fe/H] + 4.576(±1.013), with accuracy ~0.45 Gyr (asteroseismic calibrators) and ~0.88 Gyr (open-cluster calibrators) — [A&A 2024](https://www.aanda.org/articles/aa/full_html/2024/10/aa49669-24/aa49669-24.html)
- A non-LTE Mg and Y study calibrated on asteroseismic ages (A&A 2026) concludes that [Y/Mg] cannot be treated as a universal chemical clock. Its calibration must account for Galactic environment, metallicity and star-formation history — [A&A, doi:10.1051/0004-6361/202660152](https://doi.org/10.1051/0004-6361/202660152) (abstract-level via search snippet; full text not retrieved)
- A 233-star solar-neighbourhood study (2026) finds residual offsets in the [Y/Mg] clock that are independent of metallicity down to [Fe/H] ~ −0.17 but trend with [Fe/H] below that — [arXiv:2610.35917](https://arxiv.org/html/2610.35917) [preprint]
- An application to the open cluster Berkeley 32 treats the multivariate [Y/Mg] age as a complementary diagnostic, not a primary constraint — [arXiv:2607.15131](https://arxiv.org/pdf/2607.15131) [preprint]

**[C/N] (first dredge-up mass → age, giants)**
- Casali et al. (2019, A&A 629, A62) [older] derived a linear [C/N]–log(age) relation from Gaia-ESO plus APOGEE open clusters, excluding stars with signs of extra mixing — [arXiv:1907.07350](https://arxiv.org/pdf/1907.07350)
- Spoo et al. (2022, OCCAM VII) used 49 APOGEE DR17 clusters: log[Age(yr)] = 10.14(±0.08) + 2.23(±0.19)[C/N], valid for 8.62 ≤ log Age ≤ 9.82 (0.4–6.6 Gyr). They also studied red-clump mixing effects — [arXiv:2203.05463](https://arxiv.org/abs/2203.05463)
- Casali et al. (2026) fit a multivariate [C/N]–age–[Fe/H] relation on 9,122 Kepler giants with asteroseismic ages and APOGEE DR17 abundances, calibrated separately for lower RGB, upper RGB and red clump. They validate on K2, TESS and open clusters. Typical precision is ~30% for 2–10 Gyr, best on the lower RGB. Performance degrades at [Fe/H] < −1 (extra mixing), for upper RGB and RC stars, and outside 2–10 Gyr. The recommended domain is lower-RGB stars with [Fe/H] > −1 — [arXiv:2609.21928](https://arxiv.org/abs/2609.21928) [preprint]
- SPA/TNG CNO abundances for 28 open clusters (2026) give separate log-age fits for RGB stars below the bump and for core-He-burning stars. RC stars show lower [C/N] than RGB stars of the same age and metallicity — [arXiv:2605.29666](https://arxiv.org/pdf/2605.29666) [preprint]
- Work separating stellar processing from chemical evolution in red-giant C and N ("nature vs nurture") — [arXiv:2403.03249](https://arxiv.org/pdf/2403.03249)

### Inferences
- **Scale inheritance.** [Y/Mg] calibrations in the solar-twin tradition are tied to isochrone ages of dwarfs. The 2026 [C/N] calibration is tied to the Kepler asteroseismic scale. So chemical clocks are *not* independent anchors of the absolute scale. They are best used as (a) an independent *ranking* test (monotonic ordering within a population) and (b) a check on whether a data-driven age generalises across [Fe/H] and α populations.
- **Population dependence.** Both [Y/Mg] and [C/N] need explicit [Fe/H] (and evolutionary-stage) terms. Outside the solar neighbourhood or at [Fe/H] < −0.7 to −1, the calibrations are untested or known to fail. Do not use them as labels for metal-poor or halo stars, or for inner/outer-disc stars with different star-formation histories.
- **Leakage risk.** Spectra-trained ML age models can learn [C/N] or s-process features as proxies. If both the model and a chemical clock rely on the same spectral features, agreement is not independent validation.

### Gaps
- [Ba/Fe], [Y/Al], [Sr/Mg], [Eu/Ba] and other s-process/α clocks were not separately researched in this session, so there are no calibrations or scatter numbers for them here.
- The full text of the 2026 NLTE [Y/Mg] paper (NLTE correction sizes, scatter in Gyr) was not retrieved (A&A HTTP 403).
- I found no quantitative test of [Y/Mg] reliability far outside the solar neighbourhood (inner/outer disc, bulge) in this session.

---

## 4. Kinematic ages: age–velocity dispersion relation (AVR)

### Takeaway
The AVR is a *population-level consistency test*, not a per-star clock. Its shape depends on Galactic dynamical history (GMC and spiral heating, mergers, radial migration, birth dispersions). It also differs between low- and high-α discs and with Galactocentric radius. A data-driven age scale should reproduce the observed AVR trends at each R and α. It cannot be calibrated on them.

### Cited Findings
- Mackereth et al. (2019) [older; APOGEE + Gaia DR2] estimated ages with a Bayesian neural network trained on asteroseismic ages. Low- and high-[α/Fe] discs have clearly different AVRs and σz/σR ratios. The high-α AVR is roughly flat, and the low-α dispersion ratio varies with mean orbital radius — [MNRAS 489, 176](https://academic.oup.com/mnras/article/489/1/176/5552689); [OSTI](https://www.osti.gov/pages/biblio/1559338)
- APOGEE red-giant spectroscopic ages (A&A 2023) show a steepening of the AVR around ~9 Gyr, now seen across R = 5–13 kpc. Intermediate-age (2–6 Gyr) dispersions are surprisingly similar across that range — [A&A 2023](https://www.aanda.org/articles/aa/full_html/2023/10/aa46666-23/aa46666-23.html)
- A LAMOST–Gaia study of ~230,000 red-clump stars fits power-law AVRs. Thin-disc radial and vertical exponents decline with R, while the thick disc shows the opposite trend. The authors attribute the thin-disc AVR to long-term GMC and spiral heating and the thick disc to more violent processes such as mergers — [arXiv:2412.07089](https://arxiv.org/pdf/2412.07089)
- Simulations show the AVR is not set by heating alone. Declining birth (zero-age) dispersions and the evolving population of scatterers also shape it, so AVR slopes cannot be inverted to ages without a dynamical model — [arXiv:1701.03668](https://arxiv.org/pdf/1701.03668) [older]
- A kinematic constraint can reveal age-scale failures in another clock. Q-branch WD kinematics (Cheng et al. 2019) showed that 5–9% of 1.08–1.23 Msun WDs have an extra ≳8 Gyr cooling delay. Their kinematic ages were far older than their cooling ages — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- Kinematics of solar-neighbourhood WDs from SDSS DR19 + Gaia DR3 (2026) — [arXiv:2610.10675](https://arxiv.org/pdf/2610.10675) [preprint; existence only, contents not extracted]

### Inferences
- **How to use it.** Bin a candidate age scale in (age, [α/Fe], R_guide) and compare σR and σz versus age with published AVRs, for example the ~9 Gyr steepening and the flat high-α AVR. Age errors flatten AVRs, by smearing old and young stars together, and blur the thin/thick transition. So the sharpness of the 9 Gyr feature is a diagnostic of age *precision* at old ages.
- **Limits.** AVR agreement cannot pin the absolute zero-point: changing the scale by ±10% only stretches the time axis, which is degenerate with heating-law exponents. It works only as a consistency test. Kinematic ages are also selection-dependent, through the vertical-height and volume selection function.
- **Independence.** AVRs are independent of stellar models only if the ages being tested are. They are most useful for checking ML ages trained partly on asteroseismic labels in regimes (dwarfs, other R) without seismic coverage.

### Gaps
- I found no recent (2024–2026) paper using the AVR explicitly as a calibration or validation metric for an ML or multi-catalogue age scale. The "heating index" terminology was not found.
- Quantitative AVR exponents per R bin from arXiv:2412.07089 were not extracted.

---

## 5. Nucleocosmochronology (Th/U, Th/Eu, U-based ages)

### Takeaway
Radioactive chronometers are fully independent of stellar-evolution models. But they depend on r-process production ratios, and they are available for only a handful of r-process-enhanced, very metal-poor stars. Per-star uncertainties are ±2–5 Gyr. About 30% of stars with actinide detections are "actinide-boost" stars, for which Th/Eu ages fail but U/Th remains usable. They serve only as an old-end sanity check (ages ≲14 Gyr), not as a calibration anchor.

### Cited Findings
- Th/U ages need initial r-process actinide production ratios. One analysis gives uncertainty contributions of 1.6 Gyr from nuclear mass models and 3.3 Gyr from observations. Astrophysical conditions add about ±2 Gyr to the Th/U chronometer [older, 2009] — [arXiv:0912.1669](https://arxiv.org/html/0912.1669)
- With three nuclear mass models, ages of 11.8, 13.5 and 10.9 Gyr (±~1.6 Gyr) are obtained for HE 1523−0901, CS 31082−001 and BD +17°3248 respectively [older] — [arXiv:0912.1669](https://arxiv.org/html/0912.1669)
- CS 31082−001: U/Th age 15.5 ± 3.2 Gyr (Schatz et al. 2002) [older] — [ApJ 579, 626](https://iopscience.iop.org/article/10.1086/342939); [ADS](https://ui.adsabs.harvard.edu/abs/2002ApJ...579..626S)
- Th/Eu-based ages of r-process-enhanced metal-poor stars range from ~11 to 14 Gyr (review) [older] — [arXiv:1806.08955](https://arxiv.org/pdf/1806.08955)
- About 30% of metal-poor stars with actinide detections show CS 31082−001-like high Th ("actinide boost"). The R-Process Alliance (2026) compiles actinide abundances, variation and evolution in metal-poor stars — [arXiv:2604.12892](https://arxiv.org/pdf/2604.12892) [preprint]
- In actinide-boost stars, Th/Eu and U/Eu give unphysical ages (e.g. −9.5 Gyr and 5.8 Gyr in one star). Only Th/U (13.0 Gyr) is consistent, requiring a production-ratio boost factor of ~3.1. The search snippet does not make clear which paper this is; the likely source is the RPA paper on 2MASS J09544277+5246414 — [arXiv:1805.11925](https://arxiv.org/pdf/1805.11925) [older]
- LAMOST J122216.85−063345.2 (Gemini-S/GHOST, 2026) has U only as an upper limit, so its U/Th age is not a precise measurement with a well-defined uncertainty — [arXiv:2603.29246](https://arxiv.org/pdf/2603.29246) [preprint]
- The HERES survey dated strongly r-process-enhanced stars and measured Th in metal-poor stars [older] — [arXiv:0910.0707](https://arxiv.org/pdf/0910.0707); [arXiv:1112.4870](https://arxiv.org/pdf/1112.4870)

### Inferences
- **Model dependence.** These ages are entirely independent of MS/giant models and of Galactic dynamics. The dominant systematics are nuclear physics and r-process site conditions.
- **Precision and use.** At ±2–5 Gyr per star, for fewer than ~10 U-detected stars, these ages only test that a data-driven scale (a) does not assign >14 Gyr or young ages to r-II stars at [Fe/H] ≲ −2.5, and (b) roughly agrees with globular-cluster ages (11.5–13.8 Gyr) at low metallicity.
- **Sample coverage.** Coverage of the very-metal-poor regime is small, so this clock cannot calibrate scale or shape.

### Gaps
- I found no 2024–2026 *new* U-based age measurement with a well-defined uncertainty, only the 2026 RPA actinide compilation and an upper-limit case.
- The exact number of stars with measured U (rather than Th only) in 2026 was not established.

---

## 6. Wide binaries and co-moving pairs as co-eval constraints

### Takeaway
Gaia EDR3 gives ~1.2–1.3 million high-confidence wide binaries within ~1 kpc (El-Badry, Rix & Heintz 2021). There are ~877,000 high-bound-probability MS+MS pairs, ~16,000 WD+MS pairs and ~1,400 WD+WD pairs. MS+MS pairs test age *precision* through coevality, without requiring any absolute scale. WD+MS pairs transfer the WD cooling scale to MS companions. WD+WD pairs test the WD clock itself.

### Cited Findings
- El-Badry, Rix & Heintz (2021) built a catalogue of ~1.2 million high-confidence spatially resolved wide binaries from Gaia EDR3 and used it to empirically validate Gaia parallax uncertainties. Other papers describe it as ~1.3 million within 1 kpc — [arXiv:2101.05282](https://arxiv.org/pdf/2101.05282); [ResearchGate](https://www.researchgate.net/publication/349198218_A_million_binaries_from_Gaia_eDR3_sample_selection_and_validation_of_Gaia_parallax_uncertainties)
- A 2026 study starts from 877,416 high-bound-probability MS+MS wide binaries in that catalogue — [arXiv:2603.18603](https://arxiv.org/pdf/2603.18603) [preprint]
- ~1,400 WD+WD and ~16,000 WD+MS pairs are in the El-Badry et al. (2021) catalogue — [arXiv:2402.14960](https://arxiv.org/html/2402.14960)
- WD+WD pairs (1,565 pairs, 423 high-fidelity) were used to test the WD total-age methods themselves (25% errors appropriate for >0.63 Msun, <12,000 K) — [arXiv:2206.00025](https://arxiv.org/abs/2206.00025)
- Wide binaries have also been used to calibrate Gaia DR3 uncertainties — [arXiv:2405.03457](https://arxiv.org/html/2405.03457) — and in gravity tests where contamination by unresolved triples matters — [arXiv:2210.07781](https://arxiv.org/html/2210.07781); [MNRAS 2026 quality framework](https://academic.oup.com/mnras/article/547/2/stag342/8497444)
- Machine-learning wide-binary detection (2025) offers an alternative to likelihood-based selection — [arXiv:2506.19942](https://arxiv.org/pdf/2506.19942)

### Inferences
- **Precision test.** For a data-driven age scale, the distribution of Δage between binary components, normalised by the quoted errors, should be consistent with N(0,1) if the errors are well calibrated. Its width as a function of (Teff, log g, [Fe/H]) maps the empirical age precision. Restrict to pairs with high bound probability and well-separated components (no blending in spectra or photometry).
- **Combined with WD+MS.** Coevality propagates the WD cooling scale to the MS companion and then to its spectral twin population. This chain gives an accuracy anchor for K/M dwarfs that isochrones and asteroseismology cannot age.
- **Contamination and caveats.**
  - Chance alignments rise with separation and in dense fields.
  - Unresolved inner binaries (triples) bias photometric ages of one component.
  - Mass transfer or merger history breaks the coevality assumption, for example in WD+MS systems with past interaction, or blue-straggler-like companions.
  - Pairs near the MS have weak age sensitivity, so the most informative pairs contain at least one evolved (subgiant/giant/turn-off) component.

### Gaps
- The exact chance-alignment contamination rates versus separation from El-Badry et al. (2021) were not extracted.
- I found no published study (2021–2026) that uses MS+MS Gaia wide binaries specifically to quantify the precision of a spectroscopic, isochrone or ML age catalogue. Searches turned up only parallax, gravity and lithium applications.
- Post-2021 WD+MS wide-binary catalogues with spectroscopic companions (LAMOST/APOGEE/GALAH cross-matches) were not enumerated.

---

## 7. Cross-cutting: which calibrators are least model-dependent, and how they combine to anchor or test a data-driven multi-catalogue age scale

### Takeaway
No single clock covers the full parameter space. A defensible scheme uses:
1. **Absolute anchors** that are least dependent on MS/giant models: WD cooling ages for >0.63 Msun WDs in wide binaries (1–10 Gyr, ~25%), LDB clusters (20–200 Myr) and EB-constrained old clusters (e.g. NGC 6791, 8.46 ± 0.66 Gyr).
2. **Coevality precision tests**: cluster members and MS+MS wide binaries.
3. **Population-level consistency tests**: the AVR, chemical-clock ordering, and the globular-cluster and nucleocosmochronology old-end bound.

### Cited Findings
- WD total ages are reliable at the ~25% level only for >0.63 Msun and <12,000 K WDs — [arXiv:2206.00025](https://arxiv.org/abs/2206.00025)
- LDB ages have been homogenised with Gaia for 9 OCs + 3 moving groups. Empirical Li ages (EAGLES) are consistent with model-insensitive LDB ages — [arXiv:2205.06354](https://arxiv.org/abs/2205.06354); [MNRAS 523, 802](https://dx.doi.org/10.1093/mnras/stad1293)
- The NGC 6791 age with marginalised systematics, 8.46 ± 0.66 Gyr, is a high-metallicity benchmark — [arXiv:2605.21682](https://arxiv.org/abs/2605.21682)
- Homogeneous MSTO catalogue ages disagree at σ ≈ 0.20–0.26 dex (MAD 0.10–0.17 dex), with a ~0.1 dex mean offset for HR23 — [arXiv:2605.23802](https://arxiv.org/html/2605.23802)
- GC absolute ages are 11.5–13.8 Gyr with 0.5–0.75 Gyr errors dominated by distance and reddening — [arXiv:2306.02180](https://arxiv.org/pdf/2306.02180); [arXiv:2505.02969](https://arxiv.org/abs/2505.02969)
- Chemical clocks inherit calibrator scales and are population-dependent: [Y/Mg] from isochrone and asteroseismic calibrators, [C/N] from Kepler asteroseismology — [arXiv:2407.07283](https://arxiv.org/abs/2407.07283); [arXiv:2609.21928](https://arxiv.org/abs/2609.21928); [A&A 2026 NLTE](https://doi.org/10.1051/0004-6361/202660152)
- AVRs depend on α-population and radius and reflect dynamical history — [arXiv:2412.07089](https://arxiv.org/pdf/2412.07089); [A&A 2023](https://www.aanda.org/articles/aa/full_html/2023/10/aa46666-23/aa46666-23.html)

### Inferences
**Ranking by independence from MS/giant stellar-evolution models** (most independent first):

| Rank | Clock | Independent of | Still depends on | Precision | Range |
|---|---|---|---|---|---|
| 1 | Nucleocosmochronology | Stellar models | Nuclear and r-process physics | Poor (±2–5 Gyr) | Tiny sample |
| 2 | WD cooling, high-mass WDs | MS/giant models | Cooling models, IFMR | ~25% | 1–10 Gyr; thousands possible via wide binaries |
| 3 | LDB | Mostly model-insensitive | Pre-MS Li-burning physics, bolometric luminosity | Strong | Young only |
| 4 | Kinematic AVR | Stellar models | Galactic dynamics | Statistical only | Population level |
| 5 | EB-constrained cluster ages | Constrains models | Models, with He, mixing and composition pinned | ~5–8% | Few clusters |
| 6 | Globular-cluster MSTO | — | Isochrones; distance-limited | ~4–6% | Old, metal-poor only |
| 7 | Homogeneous MSTO catalogues | — | Fully isochrone (PARSEC); fixed or limited [Fe/H] | ~0.1–0.25 dex systematic | Thousands of clusters |
| 8 | Chemical clocks | Not an absolute anchor | Calibrator scale | — | Population-specific |

**Complementarity map**
- Young (<0.3 Gyr): LDB plus young-cluster MSTO and EAGLES Li.
- Intermediate (0.3–3 Gyr): MSTO clusters, high-mass-WD binaries, [C/N] for lower-RGB stars above ~2 Gyr.
- Old disc (3–10 Gyr): high-mass-WD binaries, NGC 6791 and other EB clusters, [Y/Mg] for solar-type dwarfs, [C/N].
- Halo (>10 Gyr): GCs, halo WD luminosity function (10.9–12 Gyr), Th/U as a ceiling.
- K/M dwarfs (no seismic or isochrone leverage): WD+MS binaries are the main absolute anchor.

**Practical protocol for a multi-catalogue consensus scale**
1. Fix the zero-point and stretch with the independent anchors: WD+MS high-mass, LDB and EB clusters.
2. Test precision with internal dispersion in clusters and Δage in MS+MS wide binaries.
3. Test shape and population transfer with the AVR by α and R, and [Y/Mg] or [C/N] monotonicity within their validity domains.
4. Treat MSTO-catalogue ages as labels with an added ~0.1–0.2 dex systematic. Never treat chemical-clock ages as independent of the asteroseismic or isochrone scale they were calibrated on.

### Gaps
- I found no published 2024–2026 study that combines WD-binary, LDB, EB-cluster, wide-binary and AVR tests into one validation suite for a Gaia/spectroscopic ML age catalogue. This appears to be an open methodological niche.
- Quantitative overlap counts between these anchor samples and specific survey catalogues (APOGEE, GALAH, LAMOST, Gaia XP) were not computed or found.
