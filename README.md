# masked-stellar-autoencoder

The **Masked Stellar Autoencoder (MSA)** is a deep learning foundation model designed to reconstruct and analyze **Gaia low-resolution spectra** for Galactic archaeology. By leveraging masked autoencoding and residual architectures, MSA learns robust representations of *Gaia* BP/RP spectra, photometry, and positional information and enables fine-tuned predictions of key stellar labels.

---

## Features

- **Masked autoencoding**: Learns to reconstruct missing or masked spectral regions, improving robustness to incomplete data.  
- **Residual encoder–decoder architecture**: Captures nonlinear stellar features while preserving fine spectral details.  
- **Multi-label fine-tuning**: Predicts stellar parameters including:  
  - Effective temperature (*T*<sub>eff</sub>)  
  - Metallicity ([Fe/H])  
  - Alpha enhancement ([α/Fe])  
  - Surface gravity (*log g*)  
  - Stellar age
  - Parallax ($\varpi$)  
- **Quantile regression**: Provides uncertainty-aware predictions with enforced quantile ordering at 16<sup>th</sup>, 50<sup>th</sup>, and 84<sup>th</sup> intervals.  
- **Applications beyond Gaia magnitude limits**: Infers stellar parameters even for stars too faint to have low-resolution spectra in *Gaia* DR3.  

---

## Scientific Motivation

- **Galactic archaeology**: Use stellar parameters to trace the formation and evolution of the Milky Way.  
- **Ultra metal-poor stars**: Identify and characterize ancient stellar populations.  
- **Dark matter dominated systems**: Search for chemo-dynamical signatures of dwarf galaxies and globular clusters.  
- **Survey integration**: Bridge *Gaia* with complementary spectroscopic surveys (APOGEE, GALAH, etc.).  

---

## Repository Structure

```bash
data/ # Scripts or instructions for data preprocessing
models/ # Model architectures (Masked Autoencoder, prediction heads)
training/ # Training loops, schedulers, and loss functions
notebooks/ # Metrics, validation scripts, visualization tools, and exploratory analysis
configs/ # Config file examples for running the model
batch_scripts/ # Example slurm files for pre-training in batches
README.md # This file
requirements.txt # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/aydanmckay/masked-stellar-autoencoder.git
cd masked-stellar-autoencoder
pip install -r requirements.txt
```

Requirements include:
* torch (PyTorch)
* sklearn (scikit-learn)
* numpy, pandas, scipy
* astropy, h5py
* matplotlib
* [rtdl_num_embeddings](https://github.com/yandex-research/rtdl-num-embeddings)

---

## Usage

### Pretraining (Masked Autoencoding)
```bash
python training/pretrain_msa.py --config configs/pretrain.yaml
```
### Fine-tuning on Stellar Labels
```bash
python training/finetune_msa.py --config configs/finetune.yaml
```
### Evaluation

Evaluation is performed in notebooks in notebooks/

---

## Training & Fine-tuning Guide (Pedagogical Overview)

This section explains *why* the training choices exist and how to configure them.

### 1) Data expectations
**Pretraining** (`training/pretrain_msa.py`)  
- Input: HDF5 file with `feature_cols` and `error_cols` in the config.  
- Features include Gaia XP coefficients + photometry + auxiliary columns (e.g., parallax, pmra/pmdec).  
- Errors are used for optional Gaussian perturbation (data augmentation).  

**Fine-tuning** (`training/finetune_msa.py`)  
- Input: FITS table with label columns (value + error) and the same `feature_cols`.  
- Labels are standardized (per-label `StandardScaler`) while features are robust-scaled (`RobustScaler`).  

### 2) Why RobustScaler for features and StandardScaler for labels?
- **Features** include outliers, missing values, and survey systematics. `RobustScaler` (median/IQR) is stable.  
- **Labels** are supervised targets; `StandardScaler` (mean/std) keeps label distributions well behaved for regression.  

### 3) Pretraining strategy (masked reconstruction)
The model learns representations by *reconstructing masked inputs*. This improves robustness to missing survey data.

Key config fields in `configs/pretrain.yaml`:
- `training.xp_masking_ratio`, `training.m_masking_ratio`  
  Control masking of XP coefficients vs. photometric bands.  
- `training.force_mask_cols`  
  Always mask specific columns (e.g., `['PARALLAX']`) so the model can reconstruct them from photometry.  
- `model.use_mask_indicators`  
  Appends a binary mask channel to the encoder input so masked values are explicit.  
- `training.pert_features`, `training.pert_scale`  
  Adds Gaussian noise scaled by errors (optional).  
- `training.presaved`  
  Optional checkpoint to resume from.  
- `training.mask_ranges`  
  Optional overrides for XP/photometry masking ranges (auto-detected by feature name by default).

**Mask indicators note**  
If `model.use_mask_indicators: true`, the encoder input dimension doubles (`len(feature_cols) * 2`) and
old checkpoints will not load. Pretrain from scratch in this mode.

**Scaler sampling (for very large datasets)**  
To fit the feature scaler without loading the entire dataset:
- `training.scaler_keys` (subset of HDF5 groups)
- `training.scaler_max_rows` (cap sampled rows)
- `training.scaler_seed` (reproducibility)

### 4) Fine-tuning strategy (predict labels)
Fine-tuning trains a prediction head on top of the encoder. For uncertainty, we use *quantile regression* at the 16th/50th/84th percentiles.

Key config fields in `configs/finetune.yaml`:
- `finetuning.lf: 'quantile'`  
  Enables quantile loss for predictive uncertainty.  
- `finetuning.mask`, `finetuning.mask_prediction`  
  `mask` applies input masking during fine-tuning.  
  `mask_prediction` can optionally force predictions to use masked inputs.  
- `finetuning.force_mask_cols`  
  Always mask columns during fine-tuning (useful for leakage control).  
- `finetuning.multitask_weight`  
  Weight applied to reconstruction loss when `multitask: true`.  

### 5) Parallax strategy (leakage-free + error-aware)
Parallax is special because Gaia provides high-quality measurements nearby but becomes noisy at distance.
We want:
1) Use Gaia parallax to improve *other* labels (ages, metallicities).  
2) Predict parallax for distant stars using photometry *without leakage*.  

**Two-pass parallax prediction (leakage-free)**  
Enable:
```yaml
finetuning:
  parallax_use_masked_pred: true
```
- Pass A (full inputs): predicts all labels.  
- Pass B (parallax-masked): predicts *parallax only*.  
- Parallax output is replaced by the masked-pass prediction.  

**MLE regularizer (optional, default OFF)**  
This adds a frequentist term that nudges photometric parallax toward Gaia when Gaia is precise:
```yaml
finetuning:
  parallax_mle_weight: 0.1   # start small
  parallax_sigma_scale: 1.0  # inflate Gaia errors if needed
  parallax_sigma_floor: 0.0  # add systematic floor
```
Mathematically (in label space):
$$
\mathcal{L}_\mathrm{MLE} \propto \frac{(\mu_{\pi,\mathrm{phot}} - \pi_{\mathrm{Gaia}})^2}{\sigma_{\mathrm{Gaia}}^2 + \sigma_{\mathrm{phot}}^2 + \sigma_\mathrm{floor}^2}.
$$

For a deeper, step-by-step guide see: `docs/training-and-finetuning.md`.

---

## Results

* Reconstruction of masked *Gaia* XP spectra
* Improved metallicity and age predictions compared to traditional regression models
* Robust generalization across surveys

---

## Citation
If you use this model, or the predictions made by this model, in your research, please cite:
```bash
@article{mckay2025msa,
  title={Extending the Reach of Gaia with Masked Stellar Autoencoders},
  author={McKay, Aydan and Fabbro, Sebastien},
  year={2025},
  journal={In preparation}
}
```

---

Developed by Aydan McKay, as part of MSc research in Galactic Archaeology and machine learning applications to stellar populations.
