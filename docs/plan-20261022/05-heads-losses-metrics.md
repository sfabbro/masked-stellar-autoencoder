# 05 — Heads, losses and metrics: statistically correct prediction architecture

`heads-design` (read-only, 2026-10-22). Only file written: this one. **[V]** re-read in code/git this session · **[P]** producer/report claim, corroborated not re-executed · **[D]** design decision proposed here (not implemented, not measured). `src/…` = `src/masked_stellar_autoencoder/…`; all `file:line` at HEAD `0476628`.

## 1. Target vector and transforms

| # | Target | Native unit | Model coord. | Positivity | Error kind | Role |
|---|---|---|---|---|---|---|
| 0 | Teff | K | `y`=Teff/K (**identity**) | y>0 | sym. K (median 68.7 K [P] `04:122`) | primary |
| 1 | logg | dex cgs | `y`=logg (identity) | — | sym. dex (0.1011 [P]) | primary |
| 2 | FeH | dex | `y`=[Fe/H] (identity) | — | sym. dex (0.0580 [P]) | primary |
| 3 | distance | kpc | `s=log(d/kpc)` | d>0 ⟹ auto | **none** (raw signed ϖ+σϖ) | headline |
| 4 | age | Gyr | `τ=log(age/Gyr)` | age>0 ⟹ auto | **asym.** 5/16/50/84/95 pct | headline |
| 5 | alpha | dex | linear, **source-tagged** | — | sym. dex + per-source κ_o | plug-in |

Standardize fit-only `u_j=(T_j(y_j)−c_j)/w_j` [V] `multilabel_experiments.py:36-43`; frozen order `(teff,logg,fe_h,s,τ,alpha)`. `eval_ensemble.py:46-63,74-95` inverts **positionally** (`i==5` parallax, `i==0` teff) [V] ⇒ any reorder silently breaks inversion (F12). For flows, order coordinates so the **most-observed** dimension is last (fewest partials in the autoregressive tail; §1 last para).

**Native-unit density (Jacobian).** For coordinate-wise `T`: `log p_Y(y|x) = log p_U(u|x) − Σ_{j∈obs} log w_j − Σ_{j∈obs} log|T'_j(y_j)|`, with `T'=1` (identity), `T'=1/d` (log kpc; `+log1000` if pc), `T'=1/age` (log Gyr). **Verified defect:** `scripts/run_multilabel_screen.py:337` computes **one scalar** `jacobian=Σlog(label_scale)` and subtracts it from every row [V] — valid only for identity transforms; with distance/age it is wrong by `+log d`/`+log age` per row. **It must become per-row, per-observed-coordinate before any distance/age density number is quoted.**

**Teff linear-Kelvin.** Wired today (affine standardizer on raw K, [V]); quoted labels/errors are symmetric K and the metric layer is in K (`point.csv.units ∈ {K,dex}` [P] `04:29`); at 3.5–7 kK a 70 K error is 1–2%, so for the *model* linear-vs-log is a 1e-4 NLL effect. For the *label term* it is not a choice: log-Teff × additive-K error is a different integral.

**Is log-Teff admissible for a flow? Yes, and arguably better** (Teff posteriors are right-skewed), but only with the explicit convolution `p_obs(Teff_obs|x)=∫p_U(log t)(1/t)N(Teff_obs;t,σ_K)dt`. A flexible flow (NSF/BPF) already represents skew, so log-space is not needed for correctness.

**Decision:** identity/K primary; `ν=log(Teff/K)` a **declared ablation** scored in native K through that integral, accepted only if it wins PIT/`pit_local_w1` in the dwarf **and** giant splits without K-MAE regression. `asinh` is **inadmissible** for distance/age (admits ≤0) and unused elsewhere.

**Partial labels — marginalize, never impute.** Closed form exists only for (mixtures of) Gaussians: `p(y_obs|x)=N(y_obs;μ_obs,(Σ+R)_{obs,obs})`. The exact masked marginal **already exists** and iterates all `2^d−1` patterns [V] `joint_prediction.py:184-189` — reuse it, do not re-derive.

Flows **cannot**: `NormalizingFlowLoss` raises on feature-wise masks [V] `torchregress/losses/nflows.py:635-643` and `pipeline/steps.py:521-526` drops any row with a missing label (`mask=usable.all(dim=-1)`) [V] ⇒ flows are **silently complete-case**. That is a *selection* (complete-case correlates with survey/brightness/tier, hence with the target) and must be printed with n, never implicit.

Options: declared complete-case cohort, per-label marginal heads, or `2^d−1` sub-flow sum (d≤3 only; 63 passes at d=6 is prohibitive).

## 2. Head families — capability matrix

`PL` = partial-label support (E exact, C complete-case, S sub-flow). Cost relative to diagonal Gaussian, same encoder.

| Family | Parameterization | d | PL | Tail | Calibration | Failure mode | Cost | Status in MSA [V] |
|---|---|---|---|---|---|---|---|---|
| Huber point | 6 scalars | 6 | E | robust, no density | CP box only | `NaN` density row `:232` | 1× | **wired** `:84` |
| Diag Gaussian | μ,logσ² | 6 | E | light | ok if noise modelled | variance→label noise; `min_variance` clamp `:114`; width≠difficulty | 1× | **wired** `:109-116` |
| Full-cov Gauss | μ,tril Σ | ≤8 | **E** | light | best cheap joint | PD needs jitter (**forbidden**); O(d³) | 1.15× | not in screen; ✓`ResidualJointHead` `:42-104` |
| Low-rank Gauss | μ,W(d×r),diag d² | ≤8 | E | light | good | truncation bias in tails | 1.2× | export ✓`steps.py:11`, unused |
| MDN diag/full K | K(w,μ,Σ) | ≤6 | **E**/component | multi-modal | needs SBC; weight collapse | component collapse; K unidentifiable; `min_std` silently active | 1.3–1.7× | not wired; ✓`mdn.py:26` |
| NSF/cINN/MAF | spline/affine coupling | 3 | **C** | flexible, bounded (NSF±8) | ok if convolved | **better joint NLL, worse point MAE** [P] BRIEF:67; f32 replay fails; under-resolved MC | 2–4× | **wired** `:117-151` |
| BPF | Bézier identity tails | 3 | C | empirical tails | untested | old ±1e-5 tail jump (F4) ⇒ needs `8973bcd` | 2–4× | wired; **ranking blocked** |
| Multi-quantile | q_τ per τ | marg. | E | none | CP-native | crossing; no joint | 1× | ✓export; **pipeline only** `steps.py:510` |
| Expectile | e_τ | marg. | E | none | no CP guarantee | e_τ ≠ q_τ (not intervals) | 1× | export, unused |
| SLS / SLSConformal | volume-min coupling | ≤6 | C | implicit | CP-composable | needs long warmup; unverified here | 2–3× | ✓`sls.py:1`, unused |
| NFM / CFM | flow matching | ≤6 | C | density only if Jacobian tracked | — | ODE error; needs `4e81a72` | 3–5× | export, unused (grep) |
| Ensemble mixture | `(1/M)Σp_m` | any | E/member | — | needs a fitted-mixture test | scored as one density, uncalibrated | M× | point+CP only [P] |

**MSA wiring.**
- `MultiLabelHead` accepts only `kind∈{huber,gaussian,flow}` (`:84`), `flow_type∈{nsf,maf,cinn,bpf}` (`:117-134`), `noise_aware` (`:142-151`), `flow_objective∈{nll,contrastive,nll_contrastive}` (`:88`).
- `run_multilabel_screen.py:878` has **no** kind whitelist beyond the head's own ⇒ extension is additive, but each new kind needs its own scoring branch (`:302-331` assumes `moments()`/`log_prob()`; huber→NaN guarded `:306,:329`).
- `pipeline/steps.py:10-13` imports `MultivariateGaussianLoss`, `LowRankGaussianLoss`, `MultiQuantileLoss`, `QuantileCrossoverLoss`, `CQR` but uses only quantile+CQR `:505-527`.
- `multilabel_experiments.py:186,196,199` divides by hardcoded **3** ⇒ wrong for d≠3.

**Implement order:** `kind="mvgaussian"` (masked) → `kind="mdn"` (diag then full) → `kind="quantile"` → distance head (§4) → age head (§5) → flows for d≤3 under the convolution gate (§3.3).

## 3. Losses and likelihoods

`L_obs = −(1/N)Σ_i[ log p_U(u_i^{obs}) − Σ_{j∈obs}log w_j − Σ_{j∈obs}log|T'_j(y_ij)| ]`, `N` = observed eligible **coordinates** for marginal heads (`LossSum(total,count,rows)` contract [V] `joint_prediction.py:239`) or **rows** for joint flows. Declare `loss_denominator ∈ {per_coordinate,per_row}`: a flow NLL and a per-coordinate Gaussian NLL are not the same unit and must never share a column.

1. **Observed-label vs intrinsic.** `p_obs(y_obs|x)=∫p_intr(y|x)N(y_obs;y,R)dy` — two estimands. `p_intr` is **not identified** from `p_obs` without `R`; deconvolution is ill-posed and worst exactly in the tails we care about. Never call `p_intr` a PDF of truth; never score `p_intr` against noisy labels and call it calibration (it rewards over-wide distributions). Stamp `density_kind∈{intrinsic,observed_label}` on every NLL/CRPS. **Verified aliasing to fix:** `observed_lp = log_prob(...,σ) if noise_aware else lp` [V]`:322-326` ⇒ non-noise-aware arms' "observed" **is** "intrinsic"; `moments()` is always intrinsic (no `+σ²`) [V]`:201-218` ⇒ saved `variance` and `observed_log_prob` are inconsistent for every noise-aware arm. Save both accessors explicitly.
2. **Analytic Gaussian convolution.** `N(μ,Σ+R)=N(μ,Σ)*N(0,R)` **exactly** for Gaussian `p_intr` and diagonal `R` — no MC, no approximation. Already correct for the diagonal head [V]`:180-184` and via `Σ+R` in the masked marginal [V]`joint_prediction.py:222`.
3. **Flows: MC with fixed nodes.** `log p_obs = logsumexp_m log p_intr(y_obs+σz_m) − log M`, `z_m` saved buffers [V]`:152-153,:225-231`. Quadrature is unavailable for d>1: GH only when `n_features==1`, analytic only for diagonal MDN [V] `noise_aware_density.py:220-245` ⇒ a 3-D flow is MC-only. **Gate:** `|NLL_MC64−NLL_MC512|/|NLL_MC512| ≤ 1e-3` declared *ex ante*, M≥64 QMC-antithetic. This is the F6 lesson (GH 16→256 never converged, `.594→.255` [P]): no un-gated integration.
4. **Label noise / hierarchical.** `R_s = κ_s²diag(σ_quoted²)+b_sI` per **source tag**, with **fitted** `κ_s`, never an invented floor. The repo already measures it: DESI−APOGEE scatter 0.0746 vs quoted 0.0102 ⇒ `κ_DESI≈7.3` [P] BRIEF:86; GALAH−APOGEE 0.048 dex. Report `κ_s` in the certificate.
5. **Input EIV.** **Not identifiable** from the 140-slot encoder — no per-star input σ enters the model (kept in a separate noise record by design [P] `design:31`). Do not claim EIV correction; admissible only as a declared sensitivity (`losses.eiv.InputNoiseAugmentationLoss` exists) in the input-noise ≪ input-spread regime. The attenuation slope (`04:53`) is a dilution *diagnostic* — report it, never divide predictions by it (breaks calibration).
6. **Tail weighting / DRO.** Weights change the estimand. **Primary = population-weighted** (uniform over locked test); tail-weighted/group-DRO is a **declared secondary arm** with weights, ESS and stratum definition recorded. At FeH<−2 the test stratum is **n=141** ⇒ ESS collapses; no weight vector buys precision — report Poisson-limited, do not "fix" it. No DRO exists in `torchregress` (`BalancedMSELoss`/`BinReweightedMSELoss`/`BalancedMSEBMC` are imbalance reweighters) ⇒ implement.
7. **Reconstruction / multi-task.** `L = L_obs + λL_recon + λ_sL_spec`, `λ=0.01/0.1` are starting values, not optima [P] `design:89`. Denominators: `L_obs` over observed coords, `L_recon` over **134 scored features** (136-ch shape kept, PARALLAX/EBV inactive [P]`design:55`), accumulate with actual microbatch denominators.
8. **Ensemble as mixture.** `p_ens=(1/M)Σp_m` in log space; point = mixture mean. Do **not** reweight on the same cal split used for CP (destroys the exchangeability the guarantee needs). Member spread alone is never uncertainty [P] `design:93`; only point+residual CP exist today [P] failures-doc:56.

## 4. Distance head (highest risk; needs no new labels)

Latent `s=log(d/kpc)`. Two products, **named differently forever**: **(A) predictive** `p_θ(s|x)`, a normalized PDF, fitted by `L_A` below, claim = "predictive distribution of log-distance given features". **(B) assisted** profile-likelihood confidence set `C_α={s: −2[ℓ(s)−ℓ(ŝ)]≤χ²₁(1−α)}` with `ℓ(s)=logN(ϖ_obs;z+e^{−s},σ_eff)`, claim = "confidence set", **may be empty or unbounded**. **(B′) forbidden rename:** `p_θ(s|x)L(s)/Z` normalized is a **posterior under a learned population prior**, not prior-free frequentist inference — may be reported only so labelled [P] `design:104`.

**Training loss — marginalized over `s`, through the raw signed parallax. No StarHorse targets, no `1/ϖ`, negative ϖ retained:**
```
L_A = −Σ_i log ∫ p_θ(s|x_i)·N(ϖ_i | z_i + exp(−s), σ_eff,i) ds        σ_eff² = κ_ϖ²σ_ϖ,quoted² + σ_zp²(G,colour)
```
*Quadrature:* diagonal-Gaussian `p_θ`, `s=μ+σu` ⇒ GH `∫ = π^{−1/2}Σ_k w_k N(ϖ|z+exp(−(μ+σu_k)),σ)`; no importance weights needed (the `e^{−h²}` decay, `h=e^{−s}`, dominates `h` growth; `logsumexp` avoids underflow). Flow/MDN ⇒ QMC-64 from `p_θ`, fixed saved nodes.

*Identifiability, said out loud:* `κ_ϖ` is identifiable from fit-set residual excess variance, but for faint/large-σϖ stars the likelihood is flat in `s` and the head is **prior-dominated** — report the mass fraction from `p_θ` vs `L` (e.g. likelihood-ratio weight at the mode) instead of hiding it.

*Two matched arms:* **feature-only** `p(s|x)` vs **measurement-assisted** `p(s|x,ϖ_obs)`; the assisted one must not reuse the same ϖ twice [P] `design:49` and keeps an **unchanged-measurement fallback**.

*Zero point:* declared/audited value (+ optional G/colour term as a penalized nuisance fitted on fit only); never estimated from the verification reference.

*Dust:* joint `(s,A_V)` **only with a 3-D map** — 2-D SFD is a line-of-sight total so `A_V(d)` is unknown ⇒ carry EBV as declared context, model the degeneracy in the predictive covariance, report by EBV quartile [P] `design:35-38`.

*Reporting:* kpc **and** ϖ mas always; `d`-space MAE, ϖ-space bias; `frac_ci_empty`, `frac_ci_upper_unbounded` (never cosmetically capped [P] `design:104`).

*Refs:* Hipparcos nearby = **negative control** (raw Gaia 0.96 mas vs free 42.30 / assisted 3.23 mas on 4,166 HIP [P] `04:113` ⇒ **no beyond-Gaia claim**); clusters/seismic later; **never** score against catalogue distances built from the same Gaia parallax; never bin distance by inverted noisy ϖ.

*Required tests:* (1) narrow-error limit `σ_eff→0` ⇒ `L_A→−log p_θ(s*)+log ϖ`, `s*=log(1/ϖ)`, to `1e-8` at `σ_eff=1e-6`; (2) float64 `gradcheck`, 3 rows; (3) `ϖ=−5 mas` finite, NLL monotone decreasing in `s` beyond the mode, argmin at large `d`; (4) empty/all-invalid batch ⇒ differentiable 0 + count 0 (pattern [V]`:279-280`); (5) GH 16→256 relative convergence `<1e-3` declared before the run; (6) 1-D Gaussian `p_θ` vs `scipy.integrate.quad` float64 to `1e-10`; (7) assert measured ϖ never enters `1/x`/`exp` except via `h(s)`; (8) unit identity `s=0⟹1 mas⟹1 kpc`, `s=log10⟹0.1 mas` [V]`joint_prediction.py:290-297` — **no 1000× factor, do not "fix" it**; (9) batch1-vs-batch128 float64 identical.

## 5. Age head

`τ=log(age/Gyr)`; StarHorse 5/16/50/84/95 pct are **noisy, asymmetric, model-conditioned**, not truth.

1. **Multi-quantile (0.16,0.5,0.84) in τ** (+5/95) with `QuantileCrossoverLoss` — matches the reported statistics, no distributional assumption, asymmetric-native, proper (pinball); already exported and used [V]`steps.py:510-513`. **Primary.**
2. **Split-normal in τ**: `σ_l=τ_50−τ_16`, `σ_r=τ_84−τ_50`, `p=√(2/π)/(σ_l+σ_r)·exp(−(τ−τ_50)²/2σ_{l,r}²)`. *Approximation* — its 16/84 pct are only approximately `τ_50∓σ_{l,r}`; report the error, do not claim exact quantile reproduction.
3. **Flow in τ** — genuinely bimodal (young/old solutions), the one target where a flow is a priori motivated; subject to §3.3 and the P1 certificate.

*Declare the bias:* fitting network quantiles to StarHorse quantiles treats labels as known ⇒ errors-in-variables on the **spread itself** ⇒ predicted widths come out **narrower than the true predictive spread**. Define the estimand as `p(age_obs|x)` and say calibration is only as good as StarHorse.
*Anchors:* APOKASC3 `table4.dat` 15,808 [P] — **held out entirely** (fit/tune/cal banned), external validation only; Kepler-field giants ⇒ not representative, not truth; 50 LEGACY stars [P] descriptive only.
*Reporting:* bias/MAE in **Gyr and log Gyr**, coverage 0.68/0.90, width decomposition, **>10 Gyr slice** (22,830 identity-test rows exist; the >10 Gyr count **needs verification** [P]`04:84`), n<20 ⇒ descriptive. No physical-truth claim.

## 6. Alpha head

APOGEE `α/M` ≠ GALAH `α/Fe` ≠ RVS `alphafe` — **different observables**. Model `p(α_o|x)` per source tag with a hierarchical link `α_o=a_o+b_oα*+ε_o`, `ε_o~N(0,κ_o²)`.

*Gauge:* per-source `(a_o,b_o)` is unidentified up to a global shift/scale ⇒ declare a reference source (`a=0,b=1`, GALAH n=550,854 [P]) or constrain `Σn_o a_o=0`.
*Identifiability:* `(a_o,b_o)` are pinned **only** by overlap stars, and the tail has **no** overlap pairs [P] BRIEF:87.
*Joint FeH–α:* model the pair jointly (2-D) so the FeH–α degeneracy lives in the covariance — marginal α MAE alone is misleading.

*Gates before any α claim:* (1) native α column per source with units/quality flags; (2) measured overlap offsets/scatter with n (`GALAH−APOGEE median −0.0045, scatter 0.048` [P] is the template); (3) source-tagged head beats a single-α baseline on overlap-consistent held-out stars; (4) cal-split coverage; (5) 2-D (FeH,α) level-set coverage. Until all five: **no α number, F05 stays "NOT RUN"** (`04:175`). This round builds only the interface (`TargetSpec(source_tag=…)` + per-source output layer + gauge constraint + `κ_o` nuisance).

## 7. Calibration and metrics

**Split conformal — finite-sample rank, never plug-in.** `q̂=s_(⌈(n+1)(1−α)⌉)`. Verified defects: `conformal.py:54` uses `np.quantile(s,1−α)` (offset `+6.02e-4` scaled at n=2046, α=0.1), `:60-63` returns **silent zero offsets** when <5 finite rows, `:18-19` implements two-sided as two one-sided gaps without splitting the level [P]. Required: exact rank; `max(q_lo−y,y−q_hi)` at one level **or** per-tail `α/2`; **raise** when `⌈(n+1)(1−α)⌉>n`.
**Joint region — rank-score primary, Bonferroni secondary.** Primary: the same rank rule on a multivariate nonconformity score (Mahalanobis radius `(y−μ)ᵀΣ⁻¹(y−μ)` or `−log p̂`) ⇒ finite-sample **marginal joint** coverage that uses the dependence. Secondary: Bonferroni box (`α/6`). Report both + region volume + level-set/integration method. The existing 91.9% all / 29.1% tail joint number [P] BRIEF:66 is **Bonferroni-only** ⇒ relabel it.
**Mondrian** by pre-registered strata (rank rule *within* stratum). State quantization: at `n=141`, `α=0.1` the index `⌈142·0.9⌉=128` exists but is the 128th of 141 ⇒ achievable coverage is discrete; report index + exact finite-sample α. CQR needs monotone rearrangement (or the crossover penalty) first. **SLS/SLSConformal** rank third — same rank rule, saved level sets, replay-checked; not before P1.
**Proper scores:** NLL (same family **and** same `density_kind` only), CRPS (marginal; flows from samples via `crps_from_samples`, **never** a Gaussian proxy [P] failures-doc:58), energy, variogram (the only dependence tests; native units per coordinate for reporting, standardized for ranking, never across unit conventions — F2 unit-shift `Σlog(scale)=5.4979676` makes `rtol+atol` non-invariant [P]`02:14`). PIT/rank/SBC are per-coordinate ⇒ marginal only; SBC needs ≥thousands per stratum; `tarp_coverage` for joint regions; `highest_posterior_density_coverage` for level sets.
**Metric uncertainty:** grouped paired bootstrap (unique `source_id` **and** HEALPix nside=64 block; duplicates exist, `cross_survey_duplicate_candidates=4,860` [P]`04:98`), 1,000 reps, seed recorded; `paired_bootstrap_ci` has **no** group arg [P] ⇒ implement `grouped_paired_bootstrap`. Multiplicity: state Bonferroni over the pre-registered slice count or BH-FDR. n<20 ⇒ descriptive.
**Cannot be claimed** (verbatim into the release): (1) coverage/NLL vs measured native labels, never physical truth; (2) CP intervals are absolute-residual boxes, not predictive densities; (3) α blocked until §6; (4) age is StarHorse-conditioned, seismic ≠ truth; (5) no beyond-Gaia distance claim; (6) flow density/NLL rankings blocked until the float64 certificate + `8973bcd` re-score; (7) level sets ≠ rectangular intervals; (8) a unit shift of a log-density is not a tolerance change.

### 7.1 Emitted schema (one immutable dir per arm; `null_reason` enum — **never a bare NaN**)

`prediction-certificate.json` (extends `02:143-152`; **new keys bold**):
```
run_id, source_sha, library_commit, checkpoint_sha256, cohort_sha256, scaler_sha256,
eval{dtype,device,batch_size,torch}, labels:[6], units:["K","dex","dex","kpc","Gyr","dex"],
target_transforms:["identity","identity","identity","log_kpc","log_gyr","identity"],
**jacobian{form:"per_row_per_coordinate", applied, terms["-log w","-log d","-log age"]},
**density_kind:"intrinsic"|"observed_label",
**loss_denominator:"per_coordinate"|"per_row",
**noise_model{kind:"diag"|"full"|"none", source_tagged, kappa{APOGEE,GALAH,DESI,…}},
**partial_label_policy:"gaussian_exact_marginal"|"complete_case"|"subflow_sum",
**n_observed_per_label{…},
**conformal{method:"split_rank", alpha, n_cal, order_stat_index, finite_sample_exact,
             joint_method:"rank_score_mahalanobis"|"bonferroni_box", strata[6]},
**density_integration{method:"analytic"|"gh"|"qmc_fixed", nodes, seed, buffer_sha256,
                       convergence{gh16,gh32,gh64,gh128,gh256,rel_change}},
n{fit,tune,cal,test}, gates{}, verdicts{point,density,cp,joint_levelset}, supersedes{}
```

`metrics.csv`, one row per `model × label × stratum`: `run_id,model_id,label,unit,stratum,n,n_eff,mae,rmse,bias,nmad,r2,atten_slope,outlier_frac,tail_mae,crps,nll,nll_space(intrinsic|observed_label),pit_ks,pit_cvm,pit_local_w1,cov_68,cov_90,cov_95,cov_nominal,width_90,interval_score_90,calib_slope,ci_lo,ci_hi,bootstrap_kind(star|block|two_way),bootstrap_reps,multiplicity,evidence_status(verified|producer_only|strict_replay_failed),null_reason,notes`.

`joint.csv`: `run_id,model_id,label_set,stratum,n,joint_cov_90,joint_cov_90_bonferroni,joint_cov_90_rank_score,region_volume_native,levelset_method,tarp_coverage,energy_score,variogram_score,replay_status`. `distance.csv`: `run_id,kind(mas|kpc),stratum,n,bias_mas,mae_mas,p68_mas,mae_logd,bias_logd,frac_err_q50,frac_err_q90,cov90_mas,cov90_kpc,frac_ci_empty,frac_ci_upper_unbounded,ci_kind(profile|predictive),ref_type(HIP|cluster|seismic),evidence_status`. `age.csv`: `run_id,stratum,n,bias_Gyr,mae_Gyr,bias_logGyr,mae_logGyr,cov68,cov90,width90_log,gt10Gyr_n,gt10Gyr_bias_log,seismic_ref,evidence_status`. `schema.json`: versioned fields (`name,dtype,unit,required,valid_range,null_reason_enum`) + `gates[]` + `provenance{}`. Every row carries `verified` and a `failure_id` back-reference when failed — **never erase an old failed verdict**.

## 8. Prioritized design recommendations

| # | Build first | Why | Retire / defer |
|---|---|---|---|
| **H1** | **Full-cov Gaussian, native units, observed-label (`Σ+R`), exact masked marginal** | cheapest correct joint: dependence + exact partial labels + analytic convolution; replaces the diagonal-Gaussian *uncertainty* claim; code exists [V] | diagonal Gaussian as the *uncertainty* model; `min_variance` clamp `:114` |
| **H2** | **MDN K=4 diag → K=8 full**, same masked marginal per component | dwarf/giant bimodality is the one known structure a Gaussian cannot express; marginal stays exact | — |
| **H3** | **Distance `p_θ(s|x)` through the marginalized raw-ϖ likelihood, GH-32; feature-only vs assisted** | only statistically correct distance path; **needs no new labels**; removes the inversion/point-target trap | StarHorse/point distance targets; `1/ϖ`; ridge pilots as *the* product (keep as negative control) |
| **H4** | **Age multi-quantile (0.16,0.5,0.84)+ in τ with crossover; split-normal control** | exported, proper, asymmetric-native, no distributional assumption; APOKASC3 held out | age as a Gaussian-in-Gyr point target; any age "truth" language |
| **H5** | **NSF for age and the (Teff,logg,FeH) triple — only after P1 certificate + `8973bcd`** | skew/multimodality is real for age; for the triple it must be *earned* vs H1/H2 | BPF/NSF **density rankings**; forward inversion F5/F6 demoted; F7/F8 retired |
| **H6** | **α interface only** (source-tagged layer + gauge + κ_o) | labels land in parallel; build the plug-in contract now | any α number / F05 tile |

**Minimum numerical tests per head.**
- *Gaussian/MDN:* diag-Σ ≡ existing `GaussianNLLLoss` to `1e-12`; `Σ_patterns p(marginal)` partition identity; PD without jitter (raise, never `+εI`); K=1 MDN ≡ Gaussian to `1e-10`; mixture NLL invariant to component permutation; float64 `gradcheck`; `min_std`/`min_variance` never active — report `min(model_σ)/min(label σ)`.
- *Quantile:* non-crossing output after the penalty; pinball recovers exact Gaussian quantiles on synthetic data; pinball over a quantile grid ≡ discrete CRPS up to a constant.
- *Conformal:* the rank rule equals the exact finite-sample index on synthetic exchangeable data (coverage ≥`1−α` over 2,000 replications, and the plug-in version shown to fail); zero-n raises.
- *Distance/age:* the §4 list plus Gyr/log round-trip and positivity guard.
- *Every arm:* batch1-vs-batch128 and float64-CPU-vs-float64-CUDA identical IDs/roles/labels; saved `nodes` buffer SHA; `loss_denominator` + `density_kind` in the certificate (the F1/F2 failure class is only closed this way).

## 9. What the Lead must put in the plan

1. **A frozen `TargetSpec` registry** — 6 targets × {unit, transform (`identity`/`log_kpc`/`log_gyr`), positivity, error kind (sym / asym-5pct), `source_tag`, `observable_kind` (physical|proxy), `status ∈ {trainable, interface_only, blocked}`}. **No config may carry a target outside the registry** (kills F12/F15).
2. **Per-row Jacobian** in the scoring path (`run_multilabel_screen.py:337` is one scalar today) and explicit `density_kind` + `loss_denominator` in every certificate — otherwise **no distance/age density number is admissible**.
3. **Ordered build H1→H2→H3→H4**, each with its §8 minimum tests and gated on the P1 float64 certificate; **H5 (flows) only after the `8973bcd` pin bump and a versioned re-score**.
4. **A partial-label policy decision**: `gaussian_exact_marginal` default; flows are `complete_case` **and that restriction must be printed as a selection** (n + induced bias), never silent (`pipeline/steps.py:521-526` drops rows today).
5. **Distance as two named products** (predictive PDF / profile-likelihood confidence set) with `frac_ci_empty`/`frac_ci_upper_unbounded` reported; **no** StarHorse distance target, **no** `1/ϖ`, negative ϖ retained; Hipparcos stated as a negative control.
6. **Age: APOKASC3 fully held out**; report Gyr *and* log Gyr, coverage, >10 Gyr slice, and the "StarHorse-conditioned, not truth" limitation.
7. **α gated on the five §6 conditions**; F05 stays a "NOT RUN" tile.
8. **Conformal rank fix** (exact order statistic, raise on small n) + **joint rank-score region primary / Bonferroni box secondary**, with region volume and level-set method; relabel the existing 91.9%/29.1% joint numbers as Bonferroni.
9. **The §7.1 schema** as the single emitted contract (certificate + `metrics/joint/distance/age.csv` + `schema.json` + `null_reason` enum), with `grouped_paired_bootstrap` (star **and** HEALPix block), multiplicity stated, failed verdicts preserved forever.
10. **Explicitly out of scope this round** (name the deferral so nobody infers it): input-EIV correction, 3-D-dust joint `(s,A_V)`, SLS/SLSConformal, NFM/CFM, DRO/group-weighted estimands, and any "physical truth" claim.
