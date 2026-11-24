"""MSA v11 encoder: token set encoder over every measured band, heteroscedastic masked
reconstruction and the XP filter-integral consistency term.

Modules: ``schema`` (input contract), ``config`` (strict run config), ``model``,
``masking``, ``losses``, ``filters`` (torchgaiaxp passband integrals), ``data``
(HATS streaming with the msa-split-v1 assignment), ``synthetic``, ``train``,
``estimate``. Entry point: ``masked_stellar_autoencoder.training.pretrain_v11``.
Doc: ``docs/programme/encoder-v11.md``.
"""
