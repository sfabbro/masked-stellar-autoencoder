import numpy as np
import h5py
import pytest
import os
import sys
import warnings

# Add data directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
from data_validator import DataValidator

class DummyScaler:
    def __init__(self, scale_=None):
        if scale_ is not None:
            self.scale_ = scale_

def test_validate_hdf5_file_valid(tmp_path):
    filepath = tmp_path / "test.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data1", data=[1, 2, 3])
        f.create_dataset("data2", data=[4, 5, 6])

    assert DataValidator.validate_hdf5_file(str(filepath)) is True
    assert DataValidator.validate_hdf5_file(str(filepath), required_keys=["data1"]) is True

def test_validate_hdf5_file_missing_keys(tmp_path):
    filepath = tmp_path / "test.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data1", data=[1, 2, 3])

    with pytest.raises(ValueError, match="Missing required keys: \\['data2'\\]"):
        DataValidator.validate_hdf5_file(str(filepath), required_keys=["data1", "data2"])

def test_validate_hdf5_file_invalid_path():
    with pytest.raises(ValueError, match="HDF5 file validation failed"):
        DataValidator.validate_hdf5_file("nonexistent_file.h5")

def test_validate_stellar_data_empty():
    data = np.array([])
    features = ["Teff", "logg"]
    report = DataValidator.validate_stellar_data(data, features)
    assert report["valid"] is False
    assert "Dataset is empty" in report["errors"]

def test_validate_stellar_data_high_nan():
    data = np.array([[1.0, np.nan], [np.nan, np.nan]])
    features = ["Teff", "logg"]
    report = DataValidator.validate_stellar_data(data, features, max_nan_fraction=0.5)
    assert report["valid"] is True
    assert len(report["warnings"]) == 1
    assert "High NaN fraction" in report["warnings"][0]

def test_validate_stellar_data_inf():
    data = np.array([[1.0, 2.0], [3.0, np.inf]])
    features = ["Teff", "logg"]
    report = DataValidator.validate_stellar_data(data, features)
    assert report["valid"] is True
    assert any("infinite values" in w for w in report["warnings"])

def test_validate_stellar_data_g_magnitude():
    # Outside typical range
    data = np.array([[4.0], [26.0]])
    features = ["G"]
    report = DataValidator.validate_stellar_data(data, features)
    assert report["valid"] is True
    assert any("G magnitude values outside typical range" in w for w in report["warnings"])

    # Inside typical range
    data_valid = np.array([[10.0], [15.0]])
    report_valid = DataValidator.validate_stellar_data(data_valid, features)
    assert not any("G magnitude values outside typical range" in w for w in report_valid["warnings"])

def test_validate_scaling_consistency_valid():
    scaler = DummyScaler(scale_=np.array([1.0, 2.0]))
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert DataValidator.validate_scaling_consistency(scaler, data) is True

def test_validate_scaling_consistency_not_fitted():
    scaler = DummyScaler()  # No scale_ attribute
    data = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Scaler not fitted - missing scale_ attribute"):
        DataValidator.validate_scaling_consistency(scaler, data)

def test_validate_scaling_consistency_none():
    class NoneScaler:
        scale_ = None
    scaler = NoneScaler()
    data = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Scaler scale_ attribute is None"):
        DataValidator.validate_scaling_consistency(scaler, data)

def test_validate_scaling_consistency_mismatch():
    scaler = DummyScaler(scale_=np.array([1.0]))
    data = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Scaler dimension mismatch"):
        DataValidator.validate_scaling_consistency(scaler, data)

def test_validate_scaling_consistency_negative():
    scaler = DummyScaler(scale_=np.array([1.0, -1.0]))
    data = np.array([[1.0, 2.0]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = DataValidator.validate_scaling_consistency(scaler, data)
        assert result is False
        assert len(w) == 1
        assert "Zero or negative scale factors detected" in str(w[-1].message)

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
