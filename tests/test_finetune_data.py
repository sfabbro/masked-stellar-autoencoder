from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from masked_stellar_autoencoder.training.finetune_data import (
    _augment_below_feh,
    _filter_metal_poor,
    _read_selected_fits,
    _scale_features,
    _scale_paired_label_blocks,
    _split_data,
    prepare_finetune_arrays,
)


def test_prepare_finetune_arrays_invalid_label_scaler():
    config = {
        "data": {
            "ft_datafile": "dummy.fits",
            "feature_cols": ["f1"],
            "classes": ["teff", "fe_h"],
            "error_cols": ["e_f1"],
            "recon_cols": ["r1"],
        },
        "finetuning": {},
        "preprocessing": {"label_scaler": "invalid_scaler_type"},
    }

    mock_df = pd.DataFrame(
        {
            "teff": [5000.0] * 20,
            "fe_h": [0.0] * 20,
            "f1": [1.0] * 20,
            "e_teff": [10.0] * 20,
            "e_fe_h": [0.1] * 20,
            "e_f1": [0.1] * 20,
        }
    )

    with patch(
        "masked_stellar_autoencoder.training.finetune_data._read_selected_fits",
        return_value=mock_df,
    ) as mock_read:
        with pytest.raises(
            ValueError,
            match="preprocessing.label_scaler must be 'standard', 'robust', or 'power', got 'invalid_scaler_type'",
        ):
            prepare_finetune_arrays(config)
        assert mock_read.call_args.args == (
            "dummy.fits",
            ["teff", "fe_h", "f1", "e_f1"],
        )


def test_prepare_finetune_arrays_rejects_feature_values_as_uncertainties():
    config = {
        "data": {
            "ft_datafile": "unused.fits",
            "feature_cols": ["f1"],
            "error_cols": ["f1"],
            "classes": ["teff", "e_teff"],
        },
        "finetuning": {},
    }
    with pytest.raises(ValueError, match="duplicates data.feature_cols"):
        prepare_finetune_arrays(config)


def test_selected_fits_columns_preserve_prepared_arrays_and_splits(tmp_path):
    from astropy.table import Table

    n = 60
    source = Table(
        {
            "teff": np.linspace(4200.0, 6200.0, n),
            "e_teff": np.full(n, 40.0),
            "logg": np.linspace(1.0, 4.5, n),
            "e_logg": np.full(n, 0.1),
            "fe_h": np.linspace(-2.5, 0.5, n),
            "e_fe_h": np.full(n, 0.08),
            "alpha": np.linspace(-0.1, 0.5, n),
            "e_alpha": np.full(n, 0.05),
            "age": np.linspace(1.0, 12.0, n),
            "e_age": np.full(n, 0.5),
            "PARALLAX": np.linspace(0.5, 5.0, n),
            "e_parallax": np.full(n, 0.1),
            "G": np.linspace(8.0, 18.0, n),
            "EBV": np.linspace(0.0, 0.4, n),
            "unused": np.array(["x" * 1000] * n),
        }
    )
    path = tmp_path / "labels.fits"
    source.write(path)
    config = {
        "data": {
            "ft_datafile": str(path),
            "classes": [
                "teff",
                "e_teff",
                "logg",
                "e_logg",
                "fe_h",
                "e_fe_h",
                "alpha",
                "e_alpha",
                "age",
                "e_age",
            ],
            "feature_cols": ["PARALLAX", "G", "EBV"],
            "error_cols": ["e_parallax", None, None],
            "recon_cols": ["PARALLAX", "G", "EBV"],
        },
        "finetuning": {"seed": 42, "metal_poor": {}},
        "preprocessing": {},
    }

    selected = prepare_finetune_arrays(config)
    required_columns = [
        *config["data"]["classes"],
        *config["data"]["feature_cols"],
        "e_parallax",
    ]
    selected_table = _read_selected_fits(str(path), required_columns)
    assert "unused" not in selected_table
    full_table = Table.read(path).to_pandas()
    with patch(
        "masked_stellar_autoencoder.training.finetune_data._read_selected_fits",
        return_value=full_table,
    ):
        baseline = prepare_finetune_arrays(config)

    for key in (
        "trainset",
        "etrainset",
        "validset",
        "evalidset",
        "testset",
        "etestset",
        "labelled_set",
        "vlabelled_set",
        "target_set",
    ):
        np.testing.assert_allclose(selected[key], baseline[key])


def _sample_frames(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "teff": np.linspace(4000, 6000, n),
            "fe_h": np.linspace(-2.5, 0.5, n),
            "e_fe_h": np.full(n, 0.1),
            "f1": np.ones(n),
        }
    )
    errordata = pd.DataFrame({"e_teff": np.full(n, 50.0), "e_fe_h": data["e_fe_h"]})
    return data, errordata


def test_filter_metal_poor_drops_rows_by_thresholds():
    data, errordata = _sample_frames()
    data.loc[0, "e_fe_h"] = np.nan
    data.loc[1, "teff"] = 3500.0
    data.loc[2, "e_fe_h"] = 0.5

    mp = {"require_finite_e_fe_h": True, "max_e_fe_h": 0.2, "min_teff": 4000.0}
    out_data, out_err = _filter_metal_poor(data, errordata, mp)

    assert len(out_data) == len(data) - 3
    assert len(out_err) == len(out_data)
    assert out_data["e_fe_h"].notna().all()
    assert (out_data["e_fe_h"] <= 0.2).all()
    assert (out_data["teff"] >= 4000.0).all()


def test_split_data_falls_back_when_stratify_fails(capsys):
    data, errordata = _sample_frames(n=8)
    mp = {"stratify_feh": True, "feh_stratify_bins": [-np.inf, 0.0, np.inf]}

    with patch(
        "masked_stellar_autoencoder.training.finetune_data.train_test_split",
        side_effect=[
            ValueError("too few samples per class"),
            (
                data.iloc[:5].to_numpy(),
                data.iloc[5:].to_numpy(),
                errordata.iloc[:5].to_numpy(),
                errordata.iloc[5:].to_numpy(),
            ),
            (
                data.iloc[5:7].to_numpy(),
                data.iloc[7:].to_numpy(),
                errordata.iloc[5:7].to_numpy(),
                errordata.iloc[7:].to_numpy(),
            ),
        ],
    ):
        splits = _split_data(data, errordata, mp)

    assert len(splits) == 6
    captured = capsys.readouterr()
    assert "stratified split failed" in captured.out


def test_augment_below_feh_duplicates_metal_poor_rows():
    trainset = np.array([[5000.0, -2.5], [5200.0, -0.5], [5400.0, 0.0]])
    etrainset = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    mp = {"augment_below_feh": -1.0, "augment_fraction": 0.5}

    out_train, out_err = _augment_below_feh(trainset, etrainset, mp, feh_col=1, seed=0)

    assert out_train.shape[0] == trainset.shape[0] + 1
    assert out_err.shape[0] == out_train.shape[0]
    assert np.all(out_train[-1, 1] < -1.0)


def test_scale_paired_label_blocks_standard_scaler():
    target_train = np.array([[5000.0, 50.0, -1.0, 0.1]], dtype=np.float64)
    target_valid = np.array([[5100.0, 55.0, -0.5, 0.1]], dtype=np.float64)

    labelled, e_labelled, vlabelled, e_vlabelled, scalers = _scale_paired_label_blocks(
        target_train, target_valid, num_classes=4, scaler_cls=StandardScaler
    )

    assert labelled.shape == (1, 2)
    assert e_labelled.shape == (1, 2)
    assert vlabelled.shape == (1, 2)
    assert len(scalers) == 2


def test_augment_below_feh_noop_when_no_metal_poor_rows():
    trainset = np.array([[5000.0, 0.0], [5200.0, 0.5]])
    etrainset = np.array([[0.1, 0.1], [0.1, 0.1]])
    mp = {"augment_below_feh": -1.0, "augment_fraction": 0.5}

    out_train, out_err = _augment_below_feh(trainset, etrainset, mp, feh_col=1, seed=0)

    np.testing.assert_array_equal(out_train, trainset)
    np.testing.assert_array_equal(out_err, etrainset)


def test_scale_features_masks_nonfinite_values_and_keeps_errors_finite():
    train = np.array([[1.0, 0.0], [2.0, 1.0], [np.inf, 2.0], [4.0, np.nan]])
    valid = np.array([[np.inf, 3.0], [5.0, np.nan]])
    test = np.array([[6.0, 4.0]])
    e_train = np.array([[0.1, 0.2], [0.2, 0.3], [np.inf, 0.4], [0.4, 0.0]])
    e_valid = np.array([[np.inf, 0.5], [0.3, np.nan]])
    e_test = np.array([[0.2, 0.2]])

    scaled = _scale_features(
        train,
        valid,
        test,
        e_train,
        e_valid,
        e_test,
        ["G", "bp_1"],
        {"xp_feature_scaling": "global"},
    )
    scaled_train, scaled_valid, _, scaled_e_train, scaled_e_valid, _, scaler = scaled

    assert np.isnan(scaled_train[2, 0])
    assert np.isnan(scaled_valid[0, 0])
    assert np.isfinite(scaler.center_).all()
    assert np.isfinite(scaler.scale_).all()
    assert np.isfinite(scaled_e_train).all() and (scaled_e_train > 0).all()
    assert np.isfinite(scaled_e_valid).all() and (scaled_e_valid > 0).all()


def test_scale_features_uses_neutral_uncertainty_for_unavailable_channels():
    train = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    errors = np.array([[0.1, np.nan], [0.2, np.nan], [0.3, np.nan]])
    scaled = _scale_features(
        train,
        train[:1],
        train[:1],
        errors,
        errors[:1],
        errors[:1],
        ["measured", "unmeasured"],
        {},
        error_available=np.array([True, False]),
    )

    for errors_scaled in (scaled[3], scaled[4], scaled[5]):
        np.testing.assert_array_equal(errors_scaled[:, 1], 1.0)
