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
pyproject.toml # Project metadata, dependencies, and Pixi configuration
pixi.lock # Resolved Pixi environments
```

---

## Installation

```bash
git clone https://github.com/aydanmckay/masked-stellar-autoencoder.git
cd masked-stellar-autoencoder
pixi install
```

Runtime dependencies are declared in `pyproject.toml`; `pixi.lock` records the
resolved versions for reproducible local and CANFAR environments.

---

## Usage

### Pretraining (Masked Autoencoding)
```bash
pixi run python -m masked_stellar_autoencoder.training.pretrain_msa --config configs/pretrain.yaml
```
### Fine-tuning on Stellar Labels
```bash
pixi run python -m masked_stellar_autoencoder.training.finetune_msa --config configs/finetune.yaml
```

### Stellar-parameter pipeline

The restart is `masked_stellar_autoencoder.pipeline`. It keeps the commands above. See [docs/pipeline.md](docs/pipeline.md) for the scaling, missing-data, and checkpoint contract.

```bash
pixi run pytest tests/test_pipeline.py -q
pixi run python -m masked_stellar_autoencoder.pipeline.train \
  --config configs/pipeline.yaml --data /path/to/table.h5
```

The CLI counts stars in a named HDF table. It does not train.

### Alliance Narval / Slurm

See [batch_scripts/README.md](batch_scripts/README.md) for `narval_pretrain.slurm`, `narval_finetune.slurm`, venv setup, and `configs/*.narval.example.yaml` path templates (`$SCRATCH/...`).

### Methodology (parallax, distances, dust)

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the frozen statistical policy and YAML `preprocessing` keys.
### Evaluation (paper tables)

After fine-tuning, export metrics and a LaTeX fragment with:

```bash
pixi run python -m masked_stellar_autoencoder.training.eval_ensemble --config configs/finetune.yaml \
  --checkpoints path/to/member1.pth path/to/member2.pth --out results/my_run
```

See [RUNLOG.md](RUNLOG.md) for the paper–code gap audit and [docs/experiment_matrix.md](docs/experiment_matrix.md) for the ablation protocol. Record pilot multitask comparisons in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

### Tests

```bash
pixi run test
```

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
