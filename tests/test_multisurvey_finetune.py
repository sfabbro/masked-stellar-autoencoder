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
