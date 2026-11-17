import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from masked_stellar_autoencoder.training.multisurvey_finetune import (
    SurveyRecords,
    build_survey_maps,
    soft_consensus,
)


def scalers():
    out = []
    for _ in range(6):
        s = StandardScaler()
        s.fit(np.array([[0.0], [2.0]]))  # mean 1, scale 1
        out.append(s)
    return out


def records():
    # rows 0..3; surveys: a (mapped), b (mapped), seis (anchor, log g only)
    return dict(
        row=np.array([2, 0, 0, 3, 2, 2]),
        survey=np.array([0, 0, 1, 1, 2, 1]),
        value=np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, np.nan],
                [1, 1, 1, 1],
                [np.nan, 3, np.nan, np.nan],
                [2, 2, 2, np.nan],
            ],
            dtype="f4",
        ),
        error=np.ones((6, 4), "f4"),
        survey_names=np.array(["a", "b", "seis"]),
        survey_is_anchor=np.array([False, False, True]),
    )


def test_gather_groups_records_by_row():
    r = SurveyRecords(records(), scalers(), n_rows=4, device="cpu")
    idx, pos = r.gather(torch.tensor([2, 1, 0]))
    assert pos.tolist() == [0, 0, 0, 2, 2]  # row 2 has 3 records, row 1 none
    vals = r.value[idx][:, 1].tolist()  # log g scaled by mean 1
    assert sorted(vals[:3]) == [1.0, 1.0, 2.0] and sorted(vals[3:]) == [5.0, 9.0]
    assert r.survey_map[idx].tolist().count(-1) == 1  # one anchor record
    assert r.mapped_names == ["a", "b"]
    assert r.support[:, 3].tolist() == [True, True]  # alpha observed by a and b


def test_maps_start_at_identity_and_zero_sum_holds():
    r = SurveyRecords(records(), scalers(), n_rows=4, device="cpu")
    maps = build_survey_maps(r)
    x = torch.linspace(-3, 3, 7)
    draws = torch.zeros(1, 7, 6)
    draws[0, :, :4] = x[:, None]
    out = maps(draws, torch.zeros(7, dtype=torch.long))
    torch.testing.assert_close(out[0, :, :4], draws[0, :, :4])
    with torch.no_grad():
        maps.bias.uniform_(-1, 1)
        maps.raw_slope.uniform_(-1, 1)
    bias, slopes = maps.coefficients()
    for j in (0, 2, 3):  # zero-sum targets
        assert abs(float(bias[:, j].mean())) < 1e-6
        torch.testing.assert_close(slopes[:, j].mean(0), torch.ones(8))
    assert float(soft_consensus(maps)) > 0  # log g is free, penalised softly


def test_consensus_runs_over_member_surveys_only():
    r = SurveyRecords(records(), scalers(), n_rows=4, device="cpu")
    maps = build_survey_maps(r, min_records=3)  # a has 2 Teff records, b has 3
    assert maps.members[:, 0].tolist() == [False, True]
    with torch.no_grad():
        maps.bias.uniform_(-1, 1)
        maps.raw_slope.uniform_(-1, 1)
    bias, slopes = maps.coefficients()
    assert abs(float(bias[1, 0])) < 1e-6  # the single member is the identity
    torch.testing.assert_close(slopes[1, 0], torch.ones(8))
    assert abs(float(bias[0, 0])) > 1e-4  # the non-member stays free


def test_spectra_groups_count_once():
    from masked_stellar_autoencoder.training.multisurvey_finetune import (
        ConsensusSurveyMap,
        survey_groups,
    )

    names = ["apogee_dr17", "sdss_dr19_astra_x", "galah_dr4"]
    g = survey_groups(names, {"apogee": ["apogee_", "sdss_dr19_astra_"]})
    assert g[0] == g[1] != g[2]
    support = torch.zeros(3, 6, dtype=torch.bool)
    support[:, 0] = True
    maps = ConsensusSurveyMap(support, support, [0], groups=g)
    torch.testing.assert_close(
        maps.member_weight[:, 0], torch.tensor([0.25, 0.25, 0.5])
    )
    with torch.no_grad():
        maps.bias.uniform_(-1, 1)
    bias, _ = maps.coefficients()
    assert abs(float((maps.member_weight[:, 0] * bias[:, 0]).sum())) < 1e-6


def test_label_loss_runs_with_both_map_kinds():
    from masked_stellar_autoencoder.training.multisurvey_finetune import (
        MultiSurveyWrapper,
    )

    r = SurveyRecords(records(), scalers(), n_rows=4, device="cpu")
    assert r.coverage.shape == (2, 4, 16**3)
    for kind in ("spline", "conditional"):
        w = MultiSurveyWrapper.__new__(MultiSurveyWrapper)
        w.device = "cpu"
        w.configure_multisurvey(
            r,
            map_kind=kind,
            anchor_weight=1.0,
            trust={"factor": 10, "prefixes": ["a"], "targets": ["fe_h"]},
        )
        y_head = torch.randn(4, 6, 3).sort(-1).values.requires_grad_()
        y_rest = torch.full((4, 6), float("nan"))
        loss = w._survey_label_loss(y_head, y_rest, torch.arange(4))
        loss.backward()
        assert torch.isfinite(loss) and y_head.grad.abs().sum() > 0
        if kind == "conditional":
            assert w.survey_maps.static_weight[0, 2] == 10.0
            assert (w.survey_maps.precision[:, :3] > 0).all()  # seen this batch


def test_unlabelled_stream_adds_reconstruction_and_consistency(tmp_path):
    import h5py
    from sklearn.preprocessing import RobustScaler

    from masked_stellar_autoencoder.models.model import (
        FinetuneContext,
        PredictionHead,
        make_model,
    )
    from masked_stellar_autoencoder.training.multisurvey_finetune import (
        MultiSurveyWrapper,
    )

    cols = ["bp_1", "bp_2", "rp_1", "G", "BP", "PARALLAX"]
    torch.manual_seed(0)
    model = make_model(6, [16, 16], 4, "elu", 4, "layer")
    with h5py.File(tmp_path / "d.h5", "w") as f:
        f.create_dataset("train", data=np.zeros(2, [("G", "f4")]))
        w = MultiSurveyWrapper(
            model=model,
            datafile=f,
            scaler=RobustScaler().fit(np.random.rand(10, 6)),
            feature_cols=cols,
            error_cols=[None] * 6,
            recon_cols=cols[:4],
        )
    w.device = "cpu"
    w.ft = PredictionHead(16, 6, torch.nn.ReLU())
    X = np.random.default_rng(0).normal(size=(50, 6)).astype("f4")
    w.configure_semisupervised(X, np.ones_like(X), ratio=1.0)
    names = FinetuneContext.__dataclass_fields__
    ctx = FinetuneContext(**{k: None for k in names})
    ctx.maskft, ctx.multitask, ctx.pert_features = True, True, False
    rec, cons = w._unlabelled_loss(8, ctx)
    (rec + cons).backward()
    assert torch.isfinite(rec) and float(cons) > 0  # two maskings disagree a bit
    assert any(p.grad is not None for p in w.ft.parameters())
    # shrinking the common scale (which the survey maps could undo) must not
    # lower the consistency term
    head = w.ft
    for k in (0.5, 0.1):
        w.ft = lambda z, k=k: k * head(z)
        torch.manual_seed(1)
        _, c = w._unlabelled_loss(64, ctx)
        torch.manual_seed(1)
        w.ft = head
        _, c1 = w._unlabelled_loss(64, ctx)
        torch.testing.assert_close(c, c1, rtol=0.05, atol=0)
