# Released stellar-parameter benchmarks vs current MSA residuals

**Prepared:** 2026-10-22. Comparison of the current MSA long-model residuals with
published XP/photometry catalogues. **This is not an apples-to-apples benchmark:**
the label surveys, quality cuts, magnitude/colour distributions, evolutionary mix,
tail fractions and even the reported metric (MAE vs MedAD vs mean precision) differ.
Use it to locate MSA in the landscape, not to claim a ranking.

---

## 1. Current MSA numbers (native APOGEE DR17 + GALAH DR3 labels, n=97,695)

Long noise-aware Gaussian + reconstruction ("selected 99M"; point metrics only — its
density replay is not certified):

| Label | MedAD | MAE | RMS | bias | tail [Fe/H]<−2 (n=141) |
|---|---:|---:|---:|---:|---:|
| T_eff | 41.2 K | 58.8 K | 91.1 K | +2.9 K | MedAD 136.9 K, MAE 161.3 K, bias +155.1 K |
| log g | 0.083 dex | 0.120 dex | 0.182 dex | −0.002 dex | MedAD 0.143, MAE 0.200, bias +0.131 |
| [Fe/H] | 0.052 dex | 0.076 dex | 0.116 dex | −0.002 dex | MedAD 0.333, MAE 0.393, bias +0.388 |

Matched-budget 10M diagonal Gaussian (useful control): T_eff 48.7/70.3 K,
log g 0.101/0.143, [Fe/H] 0.066/0.097 (MedAD/MAE); tail [Fe/H]<−2:
178/199 K, 0.310/0.395, 0.724/0.793. W1 quantile (diagnostic mean): FeH MAE 0.091,
tail FeH 0.583.

Caveats: local preliminary scoring; test is complete-case with spectroscopic labels;
tail n=141; the long arm is a different budget and its distributional metrics are not
certified.

---

## 2. Published benchmarks

### 2.1 Andrae et al. 2023 — GSP-Phot (Gaia DR3 XP + photometry + parallax)

Bayesian isochrone forward model, ~471M sources; zero-shot physical model (not trained
on the labels it is validated against). Validation against literature values
([Andrae et al. 2023, A&A 674, A27](https://arxiv.org/abs/2206.06138), Tables 1–3):

| Reference | T_eff MedAD (K) | log g MedAD (dex) | [M/H] MedAD (dex) |
|---|---:|---:|---:|
| APOGEE DR16 | 169 | 0.218 | 0.210 |
| GALAH DR3 | 110 | 0.059 | 0.210 |
| LAMOST DR4 | 110 | 0.104 | 0.204 |
| RAVE DR6 | 160 | 0.252 | 0.198 |

GSP-Phot uncertainties are known to be **underestimated** by factors ~1.6 (distance)
to 3.4 ([M/H]) in the split-epoch test; [M/H] is the weakest parameter and is
systematically biased. The APOGEE T_eff RMSD is 1294 K because of the
temperature–extinction degeneracy for high-extinction giants; the GALAH RMSD is 228 K.

### 2.2 Andrae, Rix & Chandra 2023 — XGBoost metallicities for 175M XP stars

[ApJS 267, 17](https://arxiv.org/abs/2302.02611). Inputs: full XP coefficients,
narrowband fluxes, broadband magnitudes, CatWISE, parallax; trained on APOGEE plus a
metal-poor augmentation. **Reported mean precision:** 50 K in T_eff, 0.08 dex in
log g, 0.1 dex in [M/H]; all-sky down to [M/H] ≈ −3; 17M bright red giants vetted.
This is the closest published analogue to the MSA input set.

### 2.3 Srivastava, Niedzielski & Smiljanic 2026 — XGBoost [Fe/H] from Gaia DR3 XP

[arXiv:2607.18707](https://arxiv.org/abs/2607.18707). 139 Gaia-derived features,
trained on 1.20M LAMOST DR10 AFGK stars. Held-out test: **MAE 0.052 dex, RMSE 0.072,
MedAD 0.039, negligible bias**; external GALAH DR4 MAE 0.066, APOGEE DR17 MAE 0.068;
open clusters 0.041. On the same stars GSP-Phot has MAE 0.242. The authors state
0.05–0.07 dex for AFGK with [Fe/H] ≳ −2.5; below that the predictions are
**lower bounds** (the metal-poor tail is not calibrated). Catalogue/model on Zenodo.

### 2.4 Nizovkina et al. 2025 — XP metallicities in open clusters

[A&A 703, A100](https://arxiv.org/abs/2510.10385). Gaia XP metallicities vs
high-resolution spectroscopy: systematic difference **0.1–0.15 dex**; median
individual error **~0.065 dex**; cluster ages with XP-prior isochrone fitting agree
with high-resolution-constrained ages to <0.03 ± 0.13 dex. Supports a ~0.05–0.1 dex
XP metallicity floor for cluster-quality stars.

### 2.5 SAGES IV 2025 — log g from photometry + XP-derived synthetic photometry

[arXiv:2509.15112](https://arxiv.org/abs/2509.15112). XGBoost trained on LAMOST;
log g accuracy **0.177 dex overall** and **0.222 dex for GK stars with DDO51**,
improving on 0.224/0.302 without DDO51. Not a pure-XP benchmark, but a recent
photometric log g reference.

### 2.6 StarHorse ages

[Anders et al. 2026, IAU Symp. 408 proceedings](https://arxiv.org/abs/2609.39907):
state-of-the-art ages are only precise to **~7%** and accurate to **~10–15%** in the
best case, and remain model/prior-dependent. This is the relevant age benchmark until
Gaia DR4 FLAME.

### 2.7 Gaia DR4 FLAME (not released yet)

Gaia DR4 goes live **2 Dec 2026**; `ap_rvs` (~35.8M rows) and `ap_xp` (~1B rows) will
carry FLAME ages/masses/luminosities
([Gaia DR4 content](https://www.cosmos.esa.int/web/gaia/dr4)). No released FLAME
validation metrics exist yet; it will be a model/prior-dependent teacher, not truth.

---

## 3. Where MSA sits

- **Global [Fe/H]:** MSA long MedAD 0.052 / MAE 0.076 is in the same range as the best
  released XP catalogues (Andrae 0.1 dex mean precision; Srivastava 0.052–0.068 MAE;
  Nizovkina ~0.065 individual). It is **not clearly better**: the metrics, label
  surveys and quality cuts differ, and MSA is trained on the same APOGEE/GALAH labels
  it is tested on, which is not a zero-shot cross-survey validation.
- **Global T_eff / log g:** MSA long MedAD 41 K / 0.083 dex, MAE 59 K / 0.120 dex
  are competitive with Andrae's 50 K / 0.08 dex mean precision and better than
  GSP-Phot's literature MedADs, but the comparison is again label/selection-dependent.
- **Metal-poor tail:** MSA explicitly measures it and shows the problem
  (+155 K, +0.13 dex log g, +0.39 dex [Fe/H] at [Fe/H]<−2). Released XP catalogues
  either do not report this regime or explicitly decline to claim it (Srivastava:
  below [Fe/H]≈−2.5 the values are lower bounds). This is where the programme can
  contribute, not by beating global averages.
- **Ages/distances:** no released all-sky XP age catalogue with validated systematics
  exists yet; StarHorse's 7%/10–15% is the current reference, and DR4 FLAME is the
  next step change.

---

## 4. What a fair comparison needs

Run the released catalogues through the **same test cohort and slices**:

1. Crossmatch the 97,695 test source IDs with Gaia DR3 `gspphot` columns,
   the Andrae 175M catalogue, and the Srivastava 2026 catalogue (and GSP-Spec where
   available).
2. Compute identical MedAD/MAE/RMS/bias, tail [Fe/H]<−2, stage/tier slices, and
   residual-vs-[Fe/H] trends on the **common stars**.
3. Compare uncertainties (coverage/PIT) on the same objects; do not compare
   quoted-precision numbers from different samples.
4. Publish both directions: our predictions on their validation stars, their
   predictions on ours.

This is a small CPU job; it is the only way to turn the indicative table above into a
defensible ranking.

---

## 5. References

- Andrae et al. 2023, GSP-Phot: https://arxiv.org/abs/2206.06138
- Andrae, Rix & Chandra 2023, 175M XP metallicities: https://arxiv.org/abs/2302.02611
- Srivastava et al. 2026, XP [Fe/H]: https://arxiv.org/abs/2607.18707
- Nizovkina et al. 2025, XP metallicities in clusters: https://arxiv.org/abs/2510.10385
- SAGES IV 2025, log g: https://arxiv.org/abs/2509.15112
- Anders et al. 2026, StarHorse ages: https://arxiv.org/abs/2609.39907
- Gaia DR4 content: https://www.cosmos.esa.int/web/gaia/dr4
