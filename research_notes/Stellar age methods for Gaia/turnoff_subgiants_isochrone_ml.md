# Ages of main-sequence turn-off (MSTO) and subgiant (SGB) stars: isochrone/Bayesian catalogues and data-driven/ML transfer (state of the art to Nov 2026)

Notes compiled 2026-11-22. "Older" means published before 2023; I mark these where they come up. Every bullet links to the source it came from. A bullet marked "(secondhand)" was taken from a search-engine summary or a citing paper, not checked against the original.

## Large isochrone-based age catalogues: methods, sizes, quoted precision, stellar types

### Takeaway
The benchmark catalogue for turn-off and subgiant stars is still Xiang & Rix (2022): about 247k LAMOST DR7 subgiants, fitted to Y2 isochrones with spectroscopic [Fe/H] and [α/Fe] plus Gaia parallaxes, median age error 7.5%. Its errors were checked with wide binaries in 2025, and an update to about 320k subgiants (often cited as "X25") now exists. Other catalogues differ in how much they trust their quoted errors and in what data they use. Nataf+2024 (about 400k subgiants, photometry only, MIST) quotes 8–10%. StarHorse 2023 (about 2.5M MSTO+SGB) quotes about 30%, or 15% for SGB only. Berger+2020 (Gaia-Kepler, all types, MIST) has a median of 56%. Gaia DR3 FLAME (BaSTI, Gaia-only) ages are of limited use for individual stars.

### Cited Findings
**Xiang & Rix 2022 (Nature; LAMOST DR7 + Gaia EDR3)**
- Bayesian fit of Gaia parallax, LAMOST stellar parameters and Gaia+2MASS photometry to Yonsei-Yale (YY/Y2) isochrones. — [Nature](https://www.nature.com/articles/s41586-022-04496-5)
- Fitted quantities: Teff, M_K, [Fe/H] and [α/Fe], plus photometry. Spectra need SNR > 20 and Teff ≤ 6800 K. Stars with M_K < 0.5 are excluded to avoid horizontal-branch contamination. Stars whose spectro-photometric and Gaia distances disagree by more than 2σ are removed, as are Gaia-flagged unresolved binaries. — [Shariat, El-Badry & Bhattacharjee 2025/26, arXiv:2510.08675](https://arxiv.org/pdf/2510.08675)
- Median fractional age uncertainty is 7.5%. The catalogue is described as the largest and most precise isochrone-age sample at the time. — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675); ["There is no place like home" birth-radius paper, MNRAS 535, 392](https://academic.oup.com/mnras/article/535/1/392/7824438)
- Sample size: 247,103 stars as used by Banik et al. 2026. After their quality cuts, 146,810 stars within 5 kpc remain. The YY grid extends to 20 Gyr, so ages are not capped at the age of the Universe. — [Banik et al. 2026, arXiv:2607.00764](https://arxiv.org/abs/2607.00764). Other papers describe the sample as "~250,000" or "~200,000", and one used 217,672 stars after cuts. — [MNRAS 535, 392](https://academic.oup.com/mnras/article/535/1/392/7824438)
- Science results: p(age, [Fe/H]) splits into two nearly disjoint parts at 8 Gyr. Thick-disc formation began about 13 Gyr ago, about 0.8 Gyr after the Big Bang. Star formation peaked about 11 Gyr ago, at the time of the GSE merger. — [arXiv:2203.12110](https://arxiv.org/abs/2203.12110)

**Xiang et al. 2024/2025 update ("X25")**
- An updated subgiant catalogue extends Xiang & Rix 2022 to 320,028 subgiants, with Teff, log g, [Fe/H], [α/Fe], age and kinematics. YY isochrones were kept because α-enhanced versions are available. The citing paper labels it both "Xiang et al. 2025" and "Xiang 2024", with DOI 10.12149/101467 (NADC). (Secondhand, from a search summary of citing papers. I did not read the catalogue paper.)
- Feltzing, Feuillet & Bensby 2026 use 319,835 LAMOST subgiants with precise literature ages. They date the disc spin-up to a mean age of 12.1 ± 2.8 Gyr (median 12.4 Gyr), at −1.25 < [Fe/H] < −0.9. — [arXiv:2605.20959](https://arxiv.org/abs/2605.20959)
- In Boin et al. 2026, X25 has the lowest dispersion of all literature catalogues against NEST/BaSTI ages: median σ = 0.79 Gyr, with near-zero offset. — [Boin et al. 2026, arXiv:2603.09540](https://arxiv.org/html/2603.09540)

**Nataf, Schlaufman, Reggiani & Hahn 2024 (ApJ 976, 87): photometric subgiants**
- 289,759 near-solar-metallicity subgiants plus 112,062 metal-poor subgiants, including more than 3000 with [Fe/H] ≲ −1.5. Inputs are Gaia DR3 distances, UV-to-mid-IR photometry and 3D extinction maps. Validated on M67 and NGC 188. Median statistical precision is about 8% in age and 0.06 dex in [Fe/H]. With systematics for non-single or variable stars, it is 9% and 0.12 dex. — [arXiv:2407.18307](https://web3.arxiv.org/abs/2407.18307); [ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ad7c4e)
- Uses the `isochrones` package with MIST. Metallicity comes mainly from UV photometry (GALEX NUV, SDSS/SkyMapper u), with no spectroscopy. The "Primary Sample" covers −0.5 ≲ [Fe/H] ≲ 0.5, and the metal-poor "annex" is selected in the UV. — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675)

**StarHorse (Queiroz et al. 2023, A&A 673, A155; spectroscopic surveys + Gaia DR3)**
- Ages for about 2.5 million MSTO+SGB stars, with typical uncertainty about 30%, falling to about 15% for SGB-only stars. (Secondhand, from a university-hosted abstract summary.) — [IAC summary](https://iac.es/en/science-and-technology/publications/starhorse-results-spectroscopic-surveys-and-gaia-dr3-chrono-chemical-populations-solar); [ADS](https://ui.adsabs.harvard.edu/abs/2023A&A...673A.155Q)
- Erratum (A&A 676, C6): a star-formation-history prior based on Mor et al. 2018 was used for the thin disc by mistake, where a broad Gaussian was intended. This changed the ages and masses of thin-disc MSTO/SGB stars, mostly the oldest ones. Corrected ages replace the originals. — [Erratum](https://www.aanda.org/10.1051/0004-6361/202245399e)
- Tomasetti et al. 2025/26 (A&A 707, A111) re-run StarHorse with a uniform 0–20 Gyr prior on the catalogue of arXiv:2402.00561. About 3000 MSTO/SGB stars have age > 12.5 Gyr and σ < 1 Gyr. After cuts, 160 "bona fide" stars remain, with an age peak at 13.6 ± 1.0 (stat) ± 1.3 (syst) Gyr. — [arXiv:2509.02692](https://arxiv.org/abs/2509.02692)

**Gaia DR3 FLAME (Apsis; Gaia-only)**
- FLAME derives mass, age and evolutionary stage from BaSTI models (Hidalgo et al. 2018). Stage index: main sequence below about 390, turn-off at about 390, subgiants 420–490, giants above 490. — [Apsis I, A&A 674](https://www.aanda.org/articles/aa/full_html/2023/06/aa43688-22/aa43688-22.html)
- FLAME ages are most reliable for turn-off stars and less certain for giants and main-sequence stars. Ages of older stars tend to be underestimated. FLAME agrees with SPInS within 1σ for main-sequence stars, and the ages depend strongly on the assumed solar abundance. (Secondhand search snippets from the ESA validation document and Apsis II.) — [ESA FLAME validation doc](https://dms.cosmos.esa.int/COSMOS/doc_fetch.php?id=1612899); [Apsis II](https://arxiv.org/pdf/2206.05992)
- In the Hyades and Pleiades, Apsis spreads cluster members over a range of ages instead of one age. The ages follow the Galactic-population input template, with earlier-type stars assigned systematically younger ages, and cluster [M/H] is underestimated by 0.1–0.2 dex. Individual ages are of very limited use and should be used only statistically. — [Brandner et al. 2023, A&A 677, A162](https://arxiv.org/abs/2306.03132)
- FLAME in Gaia DR4 (Creevey et al., submitted June 2026) is expected to give results for about 500 million sources. It agrees statistically with the Sun and with well-characterised stars, but the abstract gives no precision numbers. — [arXiv:2607.00264](https://arxiv.org/abs/2607.00264)

**Other catalogues**
- Sanders & Das 2018 (older): distances, masses and ages for about 3 million stars from APOGEE, GES, GALAH, LAMOST, RAVE and SEGUE combined with Gaia DR2. Extinction maps serve as priors, and spectroscopic masses are used for giants where available. The abstract gives no typical uncertainties. — [arXiv:1806.02324](https://arxiv.org/abs/1806.02324)
- Kordopatis et al. 2023 (A&A): ages, masses, extinctions and orbits from Gaia DR3 GSP-Spec parameters. — [A&A 2023](https://www.aanda.org/articles/aa/full_html/2023/01/aa44283-22/aa44283-22.html). In Boin et al. 2026 this catalogue has the highest dispersion against NEST of the catalogues compared: median σ = 1.61 Gyr. — [arXiv:2603.09540](https://arxiv.org/html/2603.09540)
- GALAH DR4 (Buder et al. 2025): isochrone ages from PARSEC+COLIBRI as likelihood-weighted averages over isochrone points. The log age grid runs from 6.19 to 10.17 in 0.01 dex steps, and [Fe/H] is converted to [M/H] with a Salaris & Cassisi prescription. (Secondhand, as described in Boin et al. 2026 per search summary.) — [arXiv:2603.09540](https://arxiv.org/html/2603.09540). DR4 cluster checks show bimodal GALAH ages for some clusters, with peaks near 10^7 yr and 5×10^9 yr. (Secondhand, figure page.) — [A&A figure F10](https://www.aanda.org/articles/aa/full_html/2025/11/aa54112-25/F10.html). Against NEST: median σ = 1.21 Gyr, with GALAH ages younger at old ages. — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
- Berger et al. 2020, Gaia-Kepler catalogue I (older): 186,301 Kepler stars. Inputs are g−Ks-based Teff, DR2 parallaxes and spectroscopic [Fe/H] where available, fitted with isoclassify/MIST. Median uncertainties: 4% radius, 7% mass, 13% density, 56% age. — [arXiv:2001.07737](https://arxiv.org/abs/2001.07737). Ages and their errors grow toward cooler Teff, and all late-K to M dwarf ages are uninformative. — [Berger et al. 2020 II](https://arxiv.org/pdf/2005.14671)
- Gaia-Kepler-TESS-Host catalogue (Berger et al. 2023, Gaia DR3): ages for 3387 Kepler, 618 K2 and 6017 TESS hosts. Against asteroseismic ages (Silva Aguirre et al.), the median offset is 0.7% and the scatter 31%. Stars with uninformative posteriors cluster near half the 30 Gyr grid maximum. — [arXiv:2301.11338](https://arxiv.org/html/2301.11338v1)
- Xiang et al. 2017 (older, ApJS): Bayesian isochrone ages for about 0.93 million LAMOST MSTO+SGB stars from Teff, M_V, [Fe/H] and [α/Fe]. — [arXiv:1707.06236](https://arxiv.org/abs/1707.06236). A later analysis puts the median age error near 34%. (Secondhand.)
- Boin et al. 2026, NEST (A&A, April 2026): MLP emulators trained on BaSTI, MIST, PARSEC, Dartmouth and SYCLIST grids map ([M/H], M_G, BP−RP) to age. Applied to LAMOST DR10 (935k LRS, 88k MRS), GALAH DR3 (13k) and DR4 (251k), and APOGEE DR17 (147k), about 1.3M stars in total. Sample cuts: 3.5 < log g < 4.45 and a "golden" cut of σ_τ/τ < 10%, σ_τ < 1 Gyr. The median per-star σ is 0.37 Gyr (BaSTI), and network-to-network σ is 0.22 Gyr. — [arXiv:2603.09540](https://arxiv.org/html/2603.09540)

### Inferences
- In practice, SGB ages reach "≤10%" only when spectroscopic [Fe/H] and [α/Fe], Gaia parallaxes and a careful SGB selection are all available. Mixed MSTO+SGB catalogues such as StarHorse and Berger include near-MS stars, which drives typical errors to 30–56%.
- There are two kinds of large catalogue. Per-star fits (Xiang & Rix, Nataf, StarHorse, GALAH DR4) and emulated fits (NEST) all inherit one grid's age scale. They can only be compared after zero-point offsets of about 1–2 Gyr are taken into account (see the next section).

### Gaps
- I did not retrieve primary numbers for Mints & Hekker (UniDAM), BASTA (Aguirre Børsen-Koch et al. 2022), or the GALAH DR4 MSTO age-precision section. Their sizes and precision are not reported here.
- I could not confirm the X25 catalogue's own paper, data release or precision figures. The "320,028" number and the YY-versus-MIST note (MIST ages "slightly higher" than YY) come from a search summary of a citing paper.
- I could not read the Sanders & Das 2018 per-stage age errors or the Gaia DR3 FLAME age-uncertainty distributions for turn-off and subgiant stars (Fouesneau et al. 2023, A&A 674, A27).

## Model dependence and systematics: grids, physics, priors, extinction, and measured age differences

### Takeaway
Model systematics set an absolute floor of roughly 1–2 Gyr, or about 10–20%, for turn-off and subgiant ages. That is well above the 5–10% statistical precision of the best catalogues. Measured examples:
- PARSEC ages run about 1.5 Gyr younger than BaSTI.
- Geneva/SYCLIST ages run up to 3–4 Gyr younger at old ages.
- A 0.2 dex error in [α/Fe] shifts ages by 1–2 Gyr.
- α-enhanced versus YY models differ by about 13%.
- Photometric versus spectroscopic metallicity gives an age ratio of 0.94 ± 0.13.
- Unintended star-formation-history priors change the oldest ages.

### Cited Findings
- **Grid-to-grid offsets (Boin et al. 2026, same stars and inputs, relative to BaSTI):**
  - PARSEC is mostly about 1.5 Gyr lower (1–2 Gyr in the conclusions).
  - SYCLIST/Geneva is lower by up to 3 Gyr at old ages (Sec. 2.2), or "as much as 4 Gyr" (conclusions). The paper is internally inconsistent here.
  - MIST shows little offset (median σ 0.31 Gyr) except near its 12 Gyr training-grid limit.
  - Dartmouth shows no offset but large scatter (σ 0.40 Gyr).
  - BaSTI is older than PARSEC and Geneva above about 4 Gyr and younger below about 1–2 Gyr.
  - These results already include metallicity zero-point shifts (MIST −0.057, Dartmouth +0.046, Geneva −0.075 dex). Without them, BaSTI would be older than all the other grids.
  - The authors state that model physics contributes about 1–2 Gyr of systematic plus random uncertainty.
  - [arXiv:2603.09540](https://arxiv.org/html/2603.09540)
- **α-enhancement:** a 0.20 dex offset in [α/Fe] can shift subgiant isochrone ages by 1–2 Gyr (Xiang & Rix 2022, Extended Data Fig. 4). — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675). NEST uses solar-scaled grids with [M/H] = [Fe/H] + 0.76[α/Fe]. The authors warn that α effects can shift older-star ages and that solar mixtures were not homogenised across surveys. — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
- **Grid choice on the same subgiants (secondhand):** a 2025 planet-host study finds its ages about 12.9% younger than Xiang & Rix 2022, with 10.8% scatter. It attributes the offset partly to α-enhanced models from Sun et al. (2023a). — [arXiv:2510.10092](https://arxiv.org/pdf/2510.10092)
- **Metallicity source:** Nataf+2024, with photometric [Fe/H] and MIST, versus Xiang & Rix 2022, with spectroscopic [Fe/H] and [α/Fe] and YY: Nataf [Fe/H] is higher by 0.19 ± 0.10 dex and the age ratio is 0.94 ± 0.13. — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675). NEST finds that Nataf ages diverge from its own above about 5 Gyr. — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
- **Isochrone morphology:** compared with PARSEC at the same age and metallicity, MIST has redder RGBs and bluer turn-offs at low [Fe/H] (redder at high [Fe/H]), with overall differences within about 20%. — [Kordopatis et al. 2023](https://www.aanda.org/articles/aa/full_html/2023/01/aa44283-22/aa44283-22.html). The updated BaSTI paper (older, 2018) reports good agreement with PARSEC and MIST on the MS, TO and SGB, with the larger differences elsewhere. — [Hidalgo et al. 2018](https://arxiv.org/pdf/1802.07319)
- **Gaia benchmark stars (Sahlholdt et al. 2019, MNRAS 482, 895; older):**
  - For subgiants with the most precise ages (β Hyi, Procyon, η Boo), different models move the age mode by about 1 Gyr, against a statistical uncertainty of about 0.25 Gyr.
  - With survey-like parameter errors, stars with log g ≳ 4.4 get no age information.
  - For one turn-off star, Y2 gives 8 Gyr and PARSEC 11 Gyr, because the main sequence sits in slightly different places in the two grids. PARSEC and MIST came out about 2 Gyr older than literature values.
  - [arXiv:1810.02829](https://arxiv.org/html/1810.02829); [MNRAS](https://academic.oup.com/mnras/article/482/1/895/5123732)
- **Grid-spread error floor (Tayar et al. 2022, ApJ 927, 31):** across widely used grids, the spread is about 4% in radius, about 5% in mass and about 20% in age for main-sequence and subgiant stars. The model-dependent part varies in a complicated way with L, Teff and [Fe/H]. — [arXiv:2012.07957](https://arxiv.org/abs/2012.07957). The study used near-solar stars, and its floors may slightly underestimate the truth. — [Eastman & Tayar 2022](https://arxiv.org/abs/2209.14301). Similar studies are still needed for evolved stars. — [RGB model-choice paper, 2025](https://arxiv.org/html/2504.17600v1)
- **Systematic budgets for the oldest stars disagree by about a factor of 3:**
  - Banik et al. 2026 (Xiang & Rix sample) quote ±0.4 Gyr total systematic from quality cuts, isochrone model and initial helium. A search snippet gives the components as about 0.3 Gyr (isochrone model) and 0.15 Gyr (helium). Their oldest-star age is 13.80 (+0.20/−0.16) Gyr. — [arXiv:2607.00764](https://arxiv.org/abs/2607.00764)
  - Tomasetti et al. 2026 (StarHorse MSTO/SGB) quote ±1.3 Gyr systematic. — [arXiv:2509.02692](https://arxiv.org/abs/2509.02692)
- **Priors:**
  - A StarHorse thin-disc SFH prior (Mor et al. 2018), applied by mistake, biased MSTO/SGB ages, mostly the oldest. — [Erratum](https://www.aanda.org/10.1051/0004-6361/202245399e)
  - Gaia DR3 Apsis ages follow the Galactic-population template, and earlier-type stars come out younger. — [Brandner et al. 2023](https://arxiv.org/abs/2306.03132)
  - Grid age caps truncate posteriors. Sanders & Das 2018 has no ages above 13 Gyr, and the MIST training grid in NEST ends at 12 Gyr. — [Boin et al. 2026](https://arxiv.org/html/2603.09540). The YY grid runs to 20 Gyr. — [Banik et al. 2026](https://arxiv.org/abs/2607.00764)
- **Extinction:** NEST omits extinction uncertainties, so its errors are lower bounds. It restricts samples to parallax > 0.33 mas (about 3 kpc) because the Vergely et al. 2022 maps give no A_V errors. NEST drops Teff as an input because spectroscopic Teff conflicted with observed colours and dominated the error budget. — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
- **Diffusion and mixing length (unverified attribution):** an older analysis of old field stars treats evolutionary ages as uncertain by about 1 Gyr from model systematics plus up to another 1 Gyr from the unknown efficiency of atomic diffusion. — [Age dating of old field stars: uncertain efficiency of atomic diffusion](https://www.researchgate.net/publication/308837584_Age_dating_of_old_field_stars_Challenges_from_the_uncertain_efficiency_of_atomic_diffusion). (Attribution from a search summary; the snippet may mix sources.)
- **NGC 6791 with eclipsing binaries plus the MSTO/SGB window** (Dufresne, Chaboyer & Rampalli 2026): 8.46 ± 0.66 Gyr, about 8%, after marginalising over convective mixing, diffusion and opacities in 10,000 Monte Carlo isochrone sets. The fit gives Y = 0.2968 ± 0.0158 and [Fe/H] = +0.280 ± 0.079, and the authors report "no single dominant contributor" to the error budget. — [arXiv:2605.21682](https://arxiv.org/abs/2605.21682)

### Inferences
- When the same stars and inputs are run through different grids, the age differences (about 1–2 Gyr, about 15–20%, rising at old ages) are comparable to or larger than the 5–10% statistical errors of the best catalogues. Any ML model trained on one catalogue inherits that catalogue's zero-point, and the label space needs to carry a record of which grid it came from.
- Composition inputs ([α/Fe], solar mixture, the [Fe/H]→[M/H] mapping) appear to be as important as the grid itself for subgiants. A 0.2 dex α error is roughly equivalent to switching from BaSTI to PARSEC.
- Even when good data are available, published systematic budgets for the same astrophysical quantity (the oldest-star age) differ by about 3×: ±0.4 versus ±1.3 Gyr. No community-standard error floor exists.

### Gaps
- I found no primary 2023–2026 quantitative study that isolates the age effects of convective overshoot, mixing length, helium (ΔY/ΔZ) and atomic diffusion for SGB field stars. A search summary mentioned a GALAH study with overshoot biases of up to −7% (mild) and −13% (strong) and small ΔY/ΔZ effects, but I could not identify or verify its source.
- I found no direct YREC-versus-others comparison for turn-off and subgiant ages.
- I did not obtain unresolved-binary effects for turn-off and subgiant ages. Possible leads: ["Towards breaking the degeneracy between stellar ages and unresolved multiplicity", arXiv:2609.39907](https://arxiv.org/pdf/2609.39907) and ["Double Trouble" on GALAH DR4 binaries, arXiv:2610.22707](https://arxiv.org/pdf/2610.22707).

## Why the subgiant branch gives the most precise isochrone ages, and achievable precision (spectroscopy+parallax vs photometry only)

### Takeaway
Subgiants are crossing quickly from core to shell hydrogen burning. Their luminosity tracks mass, and therefore age, while Teff changes rapidly, so the isochrones are well separated in the HR diagram. With Gaia parallaxes plus spectroscopic [Fe/H] and [α/Fe], wide binaries confirm realistic precision of 5–10%. Photometry-only pipelines quote 8–10% but are wrong by 2–3×, which suggests about 16–30% in reality. Main-sequence dwarfs with log g ≳ 4.4 carry essentially no isochrone age information at survey precision.

### Cited Findings
- Subgiants are chosen because, during the transition from core to shell hydrogen burning, isochrone ages can be very precise. — [Nataf et al. 2024](https://web3.arxiv.org/abs/2407.18307). A later paper attributes the precision to the rapid evolution of Teff and luminosity across the HR diagram (secondhand search summary).
- **Wide-binary test (Shariat, El-Badry & Bhattacharjee, OJAp, v4 May 2026):**
  - Binaries with both components evolved: 12 in Xiang & Rix, 61 in Nataf (16 in the Primary Sample), 76 in Wang+2023 giants (21 with SNR > 50).
  - Xiang & Rix: σ_z = 2.50, driven by one outlier whose [Fe/H] is discrepant at 3.25σ. Without it, σ_z = 0.81 (N = 11), so the quoted errors are accurate and the realistic precision is 5–10%.
  - Nataf full sample: σ_z = 2.73 (MAD-based 1.22). Here 77% of systems have |z| < 2 and 23% have |z| > 2. The Primary Sample gives σ_z ≈ 2.2, so errors are underestimated by 2–3×. Possible causes are photometric blends, extinction bias and metallicity errors. A variability cut explains only one outlier.
  - LAMOST giants (Wang+2023): σ_z = 1.08 (0.75 at SNR > 50), with realistic precision of about 25–30%.
  - The authors attribute the Xiang & Rix advantage to the greater intrinsic age sensitivity of subgiants plus spectroscopic abundance anchoring.
  - Caveat: wide binaries test only relative errors. Shared metallicity and Teff systematics cancel, and a global age-scale offset would not show up.
  - [arXiv:2510.08675 (PDF)](https://arxiv.org/pdf/2510.08675); [abstract](https://arxiv.org/abs/2510.08675)
- **Photometry only:** Nataf 2024 quotes about 8% statistical and about 9% including systematics. — [arXiv:2407.18307](https://web3.arxiv.org/abs/2407.18307). They conclude that UV photometry is as useful as spectroscopy for subgiant ages. — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675). The wide-binary test above contradicts that conclusion.
- **Gaia-only (FLAME DR3):** individual ages have very limited informative power, and cluster members are spread over ages. — [Brandner et al. 2023](https://arxiv.org/abs/2306.03132)
- **Dwarfs carry little information:** with survey-like errors, isochrone fitting gives no age information at log g ≳ 4.4. — [Sahlholdt et al. 2019](https://arxiv.org/html/1810.02829). Late-K and M dwarf ages in Berger 2020 are uninformative. — [Berger et al. 2020 II](https://arxiv.org/pdf/2005.14671). NEST notes that main-sequence ages are poorly constrained, and its young clusters (log τ < 9) come out systematically too old because of overestimated MS ages. — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
- **Mixed MSTO+SGB vs SGB-only:** StarHorse quotes about 30% versus about 15% (secondhand). — [IAC summary](https://iac.es/en/science-and-technology/publications/starhorse-results-spectroscopic-surveys-and-gaia-dr3-chrono-chemical-populations-solar). Xiang et al. 2017 (MSTO+SGB): median about 34% (secondhand). — [arXiv:1707.06236](https://arxiv.org/abs/1707.06236)

### Inferences
- Precision achievable, roughly in descending order:
  - Spectroscopic [Fe/H] and [α/Fe] plus parallax, SGB only: 5–10%, validated.
  - Photometric/UV [Fe/H] plus parallax, SGB: about 16–30% in reality, inferred as 2–3× the quoted 8–10%.
  - Mixed MSTO+SGB from spectroscopy: about 15–35%.
  - All-type isochrone catalogues: about 56% median.
  - Gaia-only FLAME: weak per-star information.
- On top of all of these comes the absolute model floor of about 10–20%.
- For Gaia XP-based work, metallicity accuracy (and [α/Fe]) is the limiting factor. XP-quality [M/H] errors of about 0.1–0.2 dex translate to ages roughly 1–2 Gyr off for old subgiants.

### Gaps
- I found no published wide-binary or cluster test of XP-metallicity-based subgiant ages.
- The quantitative age-precision gain from parallax precision (for example, a distance-dependent curve) was not obtained.

## Data-driven/ML age transfer: methods, what they inherit, out-of-distribution behaviour, performance

### Takeaway
"ML ages" for turn-off, subgiant and dwarf stars come in three kinds:
1. Emulators of isochrone grids (NEST, 2026). These are fast and reproduce Bayesian fits for the same grid, but inherit that grid's systematics.
2. Supervised transfer from catalogue labels: LAMOST-spectrum XGBoost ages for about 4M dwarfs trained on wide-binary-inherited isochrone ages (2025), and Gaia XP NGBoost ages for 15M stars trained on APOGEE astroNN ages (2026).
3. Cannon-type transfer from asteroseismic dwarfs (2026).

Reported scatters are about 1.0–1.8 Gyr, or 10–25%. These models inherit the zero-point of their training labels, appear to rely partly on chemical-clock features instead of stellar physics, and restrict output to the label domain. Out-of-distribution tests are generally absent. As of the end of 2026, I found no dedicated Gaia XP model trained specifically on subgiant isochrone ages.

### Cited Findings
- **NEST, Boin et al. 2026 (grid emulator):**
  - MLPs trained directly on evolutionary grids, not on catalogue ages. For the same grid they reproduce SPInS-like Bayesian ages.
  - Test-split MAD is about 35 Myr. A typical star runs about 60,000× faster than SPInS (about 20 s per star).
  - Cluster validation (13 open clusters, 1 globular cluster): MAD 0.20 Gyr against literature ages, versus 0.27 Gyr for SPInS.
  - No rotation, solar-scaled grids, extinction errors omitted.
  - [arXiv:2603.09540](https://arxiv.org/html/2603.09540); [A&A](https://www.aanda.org/articles/aa/full_html/2026/04/aa58436-25/aa58436-25.html)
- **Wang, Xiang et al. 2025 (ApJS; LAMOST DR10, about 4 million dwarfs):**
  - XGBoost on LAMOST spectra (R ~1800, SNR > 50).
  - Training labels: wide binaries, where the primary's isochrone age (MSTO/SGB primary, DD-Payne DR9 parameters plus MIST) is assigned to the dwarf companion, plus field and cluster stars.
  - Precision 10–25% for K dwarfs, worse for younger stars. Tests suggest the age information comes largely from chemical-abundance features (a "chemical clock").
  - [arXiv:2508.03019](https://arxiv.org/abs/2508.03019); training details per [HTML](https://arxiv.org/html/2508.03019)
- **Hou, Wu, Zhao, Li & Zhao, July 2026 (Gaia XP, NGBoost):**
  - Inputs: XP-derived Teff, log g, [Fe/H], [C/Fe], [N/Fe] and [α/M] from Fallows & Sanders.
  - Training labels: APOGEE DR17 astroNN ages. 489,913 cross-matches, reduced to 135,569 after cuts.
  - The loss includes label-age error, and 200-draw Monte Carlo propagates input errors. The catalogue has 15,175,107 stars.
  - Test set: mean residual −0.119 Gyr, scatter 1.200 Gyr. Against three literature catalogues the scatter is σ ≈ 1.5–1.8 Gyr. Ages are systematically younger than Miglio et al. (seismic) and Anders et al. (XGBoost) because of a different zero-point.
  - Coverage is limited to the reference domain (Teff about 3700–6030 K, log g about 0.66–4.83). No out-of-distribution or cluster tests.
  - [arXiv:2607.17932](https://arxiv.org/html/2607.17932)
- **SHBoost (Khalatyan et al. 2024):** XGBoost transfers StarHorse labels from about 8M spectroscopic stars to 217M XP stars. Precision: A_V 0.20 mag, log Teff 0.01 dex, log g 0.20 dex, [M/H] 0.18 dex, mass 12%. The authors explicitly do not derive ages, because age is "not directly encoded" in XP. — [arXiv:2407.06963](https://arxiv.org/abs/2407.06963)
- **SIDRA (MNRAS 2025/26; giants):** trained on APOGEE-derived ages and applied to 2,218,154 XP giants. Shown as a giant-branch comparator. — [arXiv:2412.09040](https://arxiv.org/abs/2412.09040)
- **The Cannon on asteroseismic dwarfs (Lam, Behmard, Ballard, Casey, Tayar & Goldberg, Oct 2026, under review at ApJ):**
  - 323 Kepler FGK dwarfs with asteroseismic ages; inputs are APOGEE spectra.
  - RMS scatter 1.0 Gyr against seismic ages.
  - Applied to 3,810 Kepler dwarfs and to 48,828 APOGEE FGK dwarfs.
  - [arXiv:2610.28345](https://arxiv.org/abs/2610.28345)
- **Gaia XP metallicities as an age aid:** XP [M/H] helps constrain open-cluster ages when the RGB is sparse. — [arXiv:2510.10385](https://arxiv.org/html/2510.10385)
- **Abundance-age ("chemical clock") calibrations** give ages about 0.7 Gyr younger on average than asteroseismology for LEGACY stars. — [A&A 2021, doi:10.1051/0004-6361/202039212](https://doi.org/10.1051/0004-6361/202039212)

### Inferences
- What transferred ages inherit:
  - The training grid's zero-point. A 1–2 Gyr inter-grid offset carries straight through.
  - The training catalogue's priors and age caps.
  - Label selection functions (for example, APOGEE cross-match cuts; Xiang & Rix's Teff ≤ 6800 K and M_K > 0.5 cuts).
- When a model relies on abundance features, it learns a Galactic age-abundance relation, not stellar physics. That relation may not hold for populations with different chemical-evolution histories (halo, accreted, bulge, metal-poor). Here a model trained on the disc is exposed to out-of-distribution failure.
- For XP specifically: reported XP-age scatter is about 1.2–1.8 Gyr (Hou 2026), several times the SGB isochrone precision (about 0.3–0.8 Gyr). That gap is consistent with the SHBoost authors' view that XP encodes age weakly except through [M/H], log g and Teff combined with parallax.
- A promising approach for XP would be a physics route: XP to [M/H] and [α/M], then parallax, photometry and an isochrone emulator such as NEST. Its precision is limited by XP metallicity accuracy, and photometric-[Fe/H] subgiant ages are known to underestimate their errors by 2–3×.

### Gaps
- I found no 2023–2026 study that trains an ML model on Xiang & Rix subgiant ages using Gaia XP inputs and validates it out of distribution.
- No quantitative out-of-distribution stress tests (metal-poor, high-α, distant or high-extinction stars) were reported for the 2025–2026 ML age catalogues.
- I did not fetch earlier data-driven MSTO and dwarf age papers: The Payne/DD-Payne age labels, the Anders et al. XGBoost giant ages, and the LAMOST DR8 neural-network MSTO/SGB ages via StarHorse. A search summary said 35,096 and 243,768 SGB+MSTO stars in LAMOST DR8 LRS/MRS, but I could not verify the source or the split.

## Benchmarks for turn-off ages and known failure modes

### Takeaway
Validation rests on:
- clusters (M67, NGC 188, NGC 6791, OC/GC sets; MAD about 0.2 Gyr for grid-emulated ages),
- wide binaries (relative errors only),
- detached eclipsing binaries in clusters (NGC 6791 at about 8%),
- asteroseismic dwarfs (LEGACY at about 10%; isochrone-versus-seismic scatter about 31% with a median offset of about 0.7%).

Documented failure modes:
- Galactic or SFH prior pull, which biases toward young ages (StarHorse erratum; FLAME template).
- Old-age caps and truncation (MIST 12 Gyr in NEST; no ages above 13 Gyr in Sanders & Das; old ages underestimated in FLAME).
- Young-cluster age inflation from MS contamination.
- Metal-poor and α-enhanced regimes where solar-scaled grids and photometric metallicities fail.

### Cited Findings
- **Clusters:**
  - NEST dated 13 open clusters and 1 globular cluster to a MAD of 0.20 Gyr. Clusters with log τ < 9 came out systematically too old, and NGC 2287 and NGC 3532 show scatter across multiple isochrones. The cluster age errors exclude systematics. — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
  - Nataf validated on M67 and NGC 188. — [arXiv:2407.18307](https://web3.arxiv.org/abs/2407.18307)
  - GALAH DR4 cluster ages are bimodal for some clusters (secondhand). — [A&A F10](https://www.aanda.org/articles/aa/full_html/2025/11/aa54112-25/F10.html)
  - Apsis ages fail to recover single-age clusters in the Hyades and Pleiades. — [Brandner et al. 2023](https://arxiv.org/abs/2306.03132)
- **Eclipsing binaries plus MSTO/SGB:** NGC 6791 at 8.46 ± 0.66 Gyr, offered as a calibrator for Galactic archaeology and asteroseismology. — [Dufresne et al. 2026](https://arxiv.org/abs/2605.21682)
- **Wide binaries** validate Xiang & Rix errors (σ_z ≈ 0.81) and expose Nataf's underestimated errors (σ_z ≈ 2.2–2.7). They cannot detect global offsets. — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675)
- **Asteroseismic dwarfs:**
  - Gaia-Kepler-TESS-Host isochrone ages versus Silva Aguirre et al. asteroseismic ages: median offset 0.7%, scatter 31%. — [arXiv:2301.11338](https://arxiv.org/html/2301.11338v1)
  - LEGACY (66 stars, older, 2017) has about 10% average age uncertainty, against 56% for isochrone ages. For 54 shared planet hosts the median ages are about 3.06 Gyr (seismic) and 3.79 Gyr (isochrone). Seismic ages do not exceed the age of the Universe, possibly because of a grid cutoff, so the largest differences appear at old ages. (Secondhand, citing papers.) — [PAST I, arXiv:2102.09424](https://arxiv.org/pdf/2102.09424); [Gyro-kinematic ages, arXiv:2102.01772](https://arxiv.org/html/2102.01772)
  - NEST cites known discrepancies between asteroseismic and isochrone ages (Tayar & Joyce 2025) but makes no quantitative comparison. — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
- **Gaia benchmark stars:** about 1 Gyr spread between models for the best subgiants. — [Sahlholdt et al. 2019](https://arxiv.org/html/1810.02829)
- **Young-star / prior pull:** StarHorse's unintended Mor+2018 SFH prior biased the oldest thin-disc MSTO/SGB ages. — [Erratum](https://www.aanda.org/10.1051/0004-6361/202245399e). Apsis ages follow the population template. — [Brandner et al. 2023](https://arxiv.org/abs/2306.03132)
- **Old-age caps and truncation:**
  - Sanders & Das 2018 has no ages above 13 Gyr, and the MIST training grid in NEST ends at 12 Gyr, which distorts the old-star age-metallicity relation (steeper [Fe/H] decline to −0.2 to −0.5 dex at 14 Gyr). — [Boin et al. 2026](https://arxiv.org/html/2603.09540)
  - FLAME underestimates older ages (secondhand). — [ESA doc](https://dms.cosmos.esa.int/COSMOS/doc_fetch.php?id=1612899)
  - Grids that extend to 20 Gyr (YY; StarHorse with a uniform 0–20 Gyr prior) give oldest-star ages of 13.6–13.8 Gyr, consistent with ΛCDM. — [Banik et al. 2026](https://arxiv.org/abs/2607.00764); [Tomasetti et al. 2026](https://arxiv.org/abs/2509.02692)
- **Metal-poor regime:** the Nataf metal-poor "annex" needs UV selection, because the SGB locus overlaps the MSTO and the base of the RGB in an age- and metallicity-dependent way. — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675). Xiang & Rix's single wide-binary outlier had a 3.25σ [Fe/H] mismatch, showing how metallicity errors produce age outliers. — [Shariat et al. 2025](https://arxiv.org/pdf/2510.08675)

### Inferences
- No current benchmark constrains the absolute age scale of field subgiants to better than about 5–8%. NGC 6791 with eclipsing binaries is at about 8%, and clusters agree to about 0.2 Gyr only within one grid. Wide binaries test precision, not accuracy.
- An ML model trained on these labels should be validated with:
  - wide-binary σ_z for relative errors;
  - cluster MAD by age bin, which flags young-cluster inflation;
  - behaviour near the old end, which tests for caps and truncation;
  - a metal-poor or high-α holdout.

### Gaps
- I did not retrieve 2023–2026 numbers for globular-cluster subgiant ages from field-style pipelines, or for LEGACY/Kepler isochrone offsets split into dwarfs and subgiants.
- No systematic catalogue-to-catalogue offset matrix (for example Xiang & Rix vs StarHorse vs FLAME vs GALAH DR4) was found beyond the NEST dispersions: X25 0.79, GALAH DR4 1.21 and Kordopatis 1.61 Gyr; StarHorse described as low with no number.
