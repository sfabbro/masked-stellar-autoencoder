import os
import pytest
import h5py
import numpy as np
from data.data_validator import DataValidator

class DummyScaler:
    def __init__(self, scale_):
        self.scale_ = scale_

def test_validate_hdf5_file_success(tmp_path):
    # Create a valid HDF5 file
    filepath = tmp_path / "valid.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=np.array([1, 2, 3]))
        f.create_dataset("labels", data=np.array([0, 1, 0]))

    # Test without required keys
    assert DataValidator.validate_hdf5_file(str(filepath)) is True

    # Test with required keys
    assert DataValidator.validate_hdf5_file(str(filepath), required_keys=["data", "labels"]) is True

def test_validate_hdf5_file_missing_keys(tmp_path):
    # Create a valid HDF5 file but with missing required keys
    filepath = tmp_path / "missing_keys.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=np.array([1, 2, 3]))

    # Test missing keys
    with pytest.raises(ValueError, match="Missing required keys: \\['labels'\\]"):
        DataValidator.validate_hdf5_file(str(filepath), required_keys=["data", "labels"])

def test_validate_hdf5_file_corrupted(tmp_path):
    # Create a corrupted/invalid HDF5 file
    filepath = tmp_path / "corrupted.h5"
    with open(filepath, "w") as f:
        f.write("This is not an HDF5 file.")

    # Test corrupted file
    with pytest.raises(ValueError, match="HDF5 file validation failed"):
        DataValidator.validate_hdf5_file(str(filepath))

def test_validate_hdf5_file_not_found():
    # Test non-existent file
    with pytest.raises(ValueError, match="HDF5 file validation failed"):
        DataValidator.validate_hdf5_file("non_existent_file.h5")

def test_validate_stellar_data_empty():
    data = np.array([])
    report = DataValidator.validate_stellar_data(data, feature_names=[])
    assert report["valid"] is False
    assert "Dataset is empty" in report["errors"]

def test_validate_stellar_data_nan():
    data = np.array([[np.nan, 1.0], [np.nan, np.nan]])
    report = DataValidator.validate_stellar_data(data, feature_names=["A", "B"], max_nan_fraction=0.5)
    assert report["valid"] is True
    assert any("High NaN fraction" in warning for warning in report["warnings"])

def test_validate_stellar_data_inf():
    data = np.array([[np.inf, 1.0], [0.0, -np.inf]])
    report = DataValidator.validate_stellar_data(data, feature_names=["A", "B"])
    assert report["valid"] is True
    assert any("Found 2 infinite values" in warning for warning in report["warnings"])

def test_validate_stellar_data_g_range():
    data = np.array([[4.0, 1.0], [26.0, 0.0], [15.0, 0.0]])
    report = DataValidator.validate_stellar_data(data, feature_names=["G", "B"])
    assert report["valid"] is True
    assert any("G magnitude values outside typical range" in warning for warning in report["warnings"])

def test_validate_stellar_data_stats():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    report = DataValidator.validate_stellar_data(data, feature_names=["A", "B"])
    assert report["valid"] is True
    assert report["stats"]["shape"] == (2, 2)
    assert report["stats"]["nan_fraction"] == 0.0
    assert report["stats"]["inf_count"] == 0
    assert report["stats"]["finite_fraction"] == 1.0

def test_validate_scaling_consistency_valid():
    scaler = DummyScaler(scale_=np.array([1.0, 2.0]))
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert DataValidator.validate_scaling_consistency(scaler, data) is True

def test_validate_scaling_consistency_not_fitted():
    class UnfittedScaler:
        pass
    scaler = UnfittedScaler()
    data = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Scaler not fitted - missing scale_ attribute"):
        DataValidator.validate_scaling_consistency(scaler, data)

def test_validate_scaling_consistency_scale_none():
    scaler = DummyScaler(scale_=None)
    data = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Scaler scale_ attribute is None"):
        DataValidator.validate_scaling_consistency(scaler, data)

def test_validate_scaling_consistency_dim_mismatch():
    scaler = DummyScaler(scale_=np.array([1.0]))
    data = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Scaler dimension mismatch: 1 vs 2"):
        DataValidator.validate_scaling_consistency(scaler, data)

def test_validate_scaling_consistency_negative_scale():
    scaler = DummyScaler(scale_=np.array([1.0, -1.0]))
    data = np.array([[1.0, 2.0]])
    with pytest.warns(UserWarning, match="Zero or negative scale factors detected"):
        assert DataValidator.validate_scaling_consistency(scaler, data) is False

def test_check_data_leakage_no_leakage():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    test_ids = np.array([7, 8, 9])
    assert DataValidator.check_data_leakage(train_ids, val_ids, test_ids) is True
    assert DataValidator.check_data_leakage(train_ids, val_ids) is True

def test_check_data_leakage_train_val():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([3, 4, 5])
    with pytest.raises(ValueError, match="Data leakage: 1 samples overlap between train and validation"):
        DataValidator.check_data_leakage(train_ids, val_ids)

def test_check_data_leakage_train_test():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    test_ids = np.array([3, 7, 8])
    with pytest.raises(ValueError, match="Data leakage: 1 samples overlap between train and test"):
        DataValidator.check_data_leakage(train_ids, val_ids, test_ids)

def test_check_data_leakage_val_test():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    test_ids = np.array([6, 7, 8])
    with pytest.raises(ValueError, match="Data leakage: 1 samples overlap between validation and test"):
        DataValidator.check_data_leakage(train_ids, val_ids, test_ids)
