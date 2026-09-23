"""Stellar-parameter pipeline.

Training sees a StellarBatch. Readers (HDF now, HATS later) only fill that.
"""

from masked_stellar_autoencoder.pipeline._path import ensure_torchregress

ensure_torchregress()
