"""v11 streaming dataset (HATS layout + msa-split-v1 assignment) and a CPU smoke run
of the pretraining entry point on a tiny synthetic sky."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("torchgaiaxp")

from masked_stellar_autoencoder.v11.config import load_config  # noqa: E402
from masked_stellar_autoencoder.v11.data import (  # noqa: E402
    HatsStream,
    MicroBatcher,
    RowBlock,
    StreamSpec,
    fit_scalers,
    split_partition,
    split_unit,
)
from masked_stellar_autoencoder.v11.filters import (  # noqa: E402
    build_operator,
    select_fi_bands,
)
from masked_stellar_autoencoder.v11.schema import (  # noqa: E402
    ContractReader,
    load_registry,
)
from masked_stellar_autoencoder.v11.synthetic import write_synthetic_trees  # noqa: E402

REG = load_registry()


def test_split_hash_matches_frozen_test_vectors():
    # datasets.yaml hash.test_vectors
    keys = np.asarray(
        [4185355646116022528, 4398046511104, 4295806720, 4295806753], np.int64
    )
    u = split_unit(keys)
    np.testing.assert_allclose(
        u, [0.662500593453, 0.798229566688, 0.852028592613, 0.882257348434], atol=1e-12
    )
    assert split_partition(u).tolist() == [0, 1, 2, 3]


@pytest.fixture(scope="module")
def sky(tmp_path_factory):
    root = tmp_path_factory.mktemp("v11sky")
    return write_synthetic_trees(
        root, REG, n_rows=1500, n_partitions=3, seed=11, shuffle_xp_rows=True
    )


def _spec(sky, partitions=("FIT",), **kw):
    fi = select_fi_bands(REG)
    reader = ContractReader(REG, "gaia_keyed_v1a")
    op = build_operator(
        REG, [b for b in fi if b in {REG.bands[i].name for i, *_ in reader.plan}]
    )
    base = dict(
        trees=[(sky["anchor"], None), (sky["xp"], None)],
        assignment=sky["assignment"],
        partitions=list(partitions),
        reader=reader,
        registry=REG,
        operator=op,
        read_batch_rows=128,
        shuffle_rows=256,
        tokens_per_microbatch=8000,
    )
    base.update(kw)
    return StreamSpec(**base)


def _collect(stream):
    out = [mb for mb in stream]
    return {k: torch.cat([mb[k] for mb in out]) for k in out[0]}, out


def test_stream_selects_exactly_the_partition_and_excludes_reserved(sky):
    all_rows, _ = _collect(HatsStream(_spec(sky)))
    sid = all_rows["source_id"].numpy()
    want = sky["source_id"][sky["partition"] == 0]
    assert len(sid) == len(set(sid))
    assert set(sid) == set(want)  # every FIT row once, nothing else (no cap)
    reserved = set(sky["source_id"][sky["partition"] == 4])
    assert reserved and not (reserved & set(sid))
    dev, _ = _collect(HatsStream(_spec(sky, partitions=("DEV",))))
    assert set(dev["source_id"].numpy()) == set(sky["source_id"][sky["partition"] == 1])


def test_join_puts_xp_on_the_right_star(sky):
    rows, _ = _collect(HatsStream(_spec(sky)))
    xi = REG.xp_indices
    has = torch.isfinite(rows["values"][:, xi]).all(1).numpy()
    sid = rows["source_id"].numpy()
    truth_xp = set(sky["source_id"][sky["truth"]["has_xp"]])
    assert set(sid[has]) == truth_xp & set(sid)
    assert np.isnan(
        rows["values"][~torch.as_tensor(has)][:, xi].numpy()
    ).all()  # no fill
    # observed-XP synthetic fluxes only where XP exists, and consistent with G-normalised XP
    syn = rows["xp_syn"].numpy()
    assert np.isfinite(syn[has]).all() and np.isnan(syn[~has]).all()
    assert (rows["xp_var"].numpy()[has] > 0).all()
    # input class from the assignment
    cls = rows["input_class"].numpy()
    assert (cls[has] == 0).all() and (cls[~has] != 0).all()


def test_stream_is_reproducible_and_shards_workers(sky):
    s = HatsStream(_spec(sky))
    a, _ = _collect(s)
    b, _ = _collect(s)
    assert torch.equal(a["source_id"], b["source_id"])
    s.set_epoch(1)
    c, _ = _collect(s)
    assert set(c["source_id"].tolist()) == set(a["source_id"].tolist())
    assert not torch.equal(c["source_id"], a["source_id"])  # reshuffled
    w0 = list(s.blocks(0, 2))
    w1 = list(s.blocks(1, 2))
    ids0 = set(np.concatenate([b.source_id for b in w0]))
    ids1 = set(np.concatenate([b.source_id for b in w1]))
    assert not ids0 & ids1 and ids0 | ids1 == set(a["source_id"].tolist())


def test_microbatches_respect_token_and_row_budgets():
    rng = np.random.default_rng(0)
    n, nb = 3000, len(REG)
    vals = np.full((n, nb), np.nan, np.float32)
    ntok = rng.integers(3, 140, n)
    for i, k in enumerate(ntok):
        vals[i, :k] = 1.0
    blk = RowBlock(
        np.arange(n),
        np.zeros(n, np.uint8),
        vals,
        vals.copy(),
        np.zeros((n, 2), np.float32),
        np.zeros((n, 2), np.float32),
    )
    mb = MicroBatcher(
        [16, 32, 64, 128, 160],
        tokens=4096,
        shuffle_rows=500,
        rng=rng,
        n_bands=nb,
        max_rows=100,
    )
    out = []
    for i in range(0, n, 250):
        out.extend(mb.add(blk.take(np.arange(i, min(i + 250, n)))))
    out.extend(mb.flush())
    seen = np.concatenate([o.source_id for o in out])
    assert sorted(seen.tolist()) == list(range(n))
    for o in out:
        assert len(o) <= 100
        assert len(o) * o.n_tokens().max() <= 4096 + 160  # padded tokens within budget


def test_scalers_are_robust_and_cover_every_band(sky):
    sc = fit_scalers(HatsStream(_spec(sky)), REG, n_rows=10_000, min_count=20)
    g = REG.index("gaia_g")
    assert 10 < sc["center"][g] < 20 and sc["scale"][g] > 0
    assert (
        sc["count"][REG.index("galex_fuv")] == 0
        and sc["scale"][REG.index("galex_fuv")] == 2.0
    )
    assert len(sc["center"]) == len(REG)


def test_missing_assignment_rows_fail(sky, tmp_path):
    import shutil

    import pyarrow.parquet as pq

    from masked_stellar_autoencoder.v11.data import hats_partitions, partition_files

    bad = tmp_path / "assignment"
    shutil.copytree(sky["assignment"], bad)
    o, p = hats_partitions(bad)[0]
    f = partition_files(bad, o, p)[0]
    t = pq.read_table(f)
    pq.write_table(t.slice(1), f)
    with pytest.raises(ValueError, match="missing from the msa-split-v1 assignment"):
        list(HatsStream(_spec(sky, assignment=str(bad))))


def test_smoke_pretraining_run_and_resume(tmp_path):
    from masked_stellar_autoencoder.v11.train import Pretrainer

    cfg = load_config(
        {
            "run": {
                "output_dir": str(tmp_path / "run"),
                "device": "cpu",
                "precision": "fp32",
                "seed": 3,
                "resume": True,
            },
            "data": {
                "source": "synthetic",
                "workers": 0,
                "read_batch_rows": 256,
                "shuffle_buffer_rows": 512,
                "tokens_per_microbatch": 16384,
                "synthetic": {
                    "n_rows": 1600,
                    "n_partitions": 2,
                    "seed": 5,
                    "directory": str(tmp_path / "sky"),
                },
                "scalers": {"fit_rows": 1000, "min_count": 10},
            },
            "model": {
                "d_model": 32,
                "n_heads": 4,
                "n_latents": 4,
                "n_layers": 2,
                "cross_attn_layers": [0],
                "value_frequencies": 4,
                "band_frequencies": 4,
                "decoder_hidden": 32,
                "decoder_layers": 1,
            },
            "optim": {"lr": 0.003, "warmup_steps": 2},
            "train": {
                "batch_rows": 400,
                "presentations": 8000,
                "max_steps": 6,
                "log_every": 3,
                "eval_every": 6,
                "eval_rows": 300,
                "checkpoint_every": 3,
            },
        }
    )
    last = Pretrainer(cfg).run()
    assert last["step"] == 6 and np.isfinite(last["loss"])
    for k in (
        "rec_xp",
        "rec_phot",
        "fi_calibrate",
        "fi_sed",
        "fi_phot",
        "dev_rec_xp",
        "dev_rec_xp_hidden",
    ):
        assert k in last and np.isfinite(last[k]), k
    run = tmp_path / "run"
    man = json.loads((run / "manifest.json").read_text())
    assert man["parameters"]["total"] > 0 and man["fi_operator"]["bands"]
    assert (run / "checkpoints" / "latest.pt").exists() and (
        run / "scalers.json"
    ).exists()
    # resume continues from the checkpoint
    cfg.train.max_steps = 8
    last2 = Pretrainer(cfg).run()
    assert last2["step"] == 8
    lines = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert any(x.get("step") == 8 for x in lines)


def test_checkpoint_loads_into_a_longer_band_list(tmp_path):
    """Band tables are matched by name, so a model loads into a registry with new bands."""
    from pathlib import Path

    from masked_stellar_autoencoder.v11.config import ModelConfig
    from masked_stellar_autoencoder.v11.model import MSAv11
    from masked_stellar_autoencoder.v11.schema import load_registry as lr
    from masked_stellar_autoencoder.v11.train import load_model_state

    text = (
        Path(__file__).resolve().parents[1]
        / "src/masked_stellar_autoencoder/v11/contract.yaml"
    ).read_text()
    longer = text.replace(
        "  - {name: galex_nuv, survey: galex, kind: mag, system: AB, wavelength_nm: 231.6}\n",
        "  - {name: galex_nuv, survey: galex, kind: mag, system: AB, wavelength_nm: 231.6}\n"
        "  - {name: galex_new, survey: galex, kind: mag, system: AB, wavelength_nm: 260.0}\n",
    )
    p = tmp_path / "contract.yaml"
    p.write_text(longer)
    reg2 = lr(p)
    cfg = ModelConfig(
        d_model=16,
        n_heads=4,
        n_latents=2,
        n_layers=1,
        cross_attn_layers=[0],
        value_frequencies=2,
        band_frequencies=2,
        decoder_hidden=16,
        decoder_layers=1,
    )
    a = MSAv11(REG, cfg)
    b = MSAv11(reg2, cfg)
    load_model_state(b, a.state_dict(), list(REG.names))
    j_old, j_new = REG.index("tmass_j"), reg2.index("tmass_j")
    torch.testing.assert_close(a.bands.id.weight[j_old], b.bands.id.weight[j_new])
    v = torch.full((2, len(REG)), float("nan"))
    v[:, [REG.index("gaia_g"), REG.index("ps1_g")]] = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]]
    )
    v2 = torch.full((2, len(reg2)), float("nan"))
    v2[:, [reg2.index("gaia_g"), reg2.index("ps1_g")]] = v[
        :, [REG.index("gaia_g"), REG.index("ps1_g")]
    ]
    a.eval()
    b.eval()
    torch.testing.assert_close(a.encode(v, v * 0 + 0.01), b.encode(v2, v2 * 0 + 0.01))
