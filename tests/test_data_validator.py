import pytest
import numpy as np
from data.data_validator import DataValidator

def test_check_data_leakage_no_overlap():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    assert DataValidator.check_data_leakage(train_ids, val_ids) is True

def test_check_data_leakage_train_val_overlap():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([3, 4, 5])
    with pytest.raises(ValueError, match="Data leakage: 1 samples overlap between train and validation"):
        DataValidator.check_data_leakage(train_ids, val_ids)

def test_check_data_leakage_with_test_no_overlap():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    test_ids = np.array([7, 8, 9])
    assert DataValidator.check_data_leakage(train_ids, val_ids, test_ids) is True

def test_check_data_leakage_train_test_overlap():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    test_ids = np.array([3, 7, 8])
    with pytest.raises(ValueError, match="Data leakage: 1 samples overlap between train and test"):
        DataValidator.check_data_leakage(train_ids, val_ids, test_ids)

def test_check_data_leakage_val_test_overlap():
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    test_ids = np.array([6, 7, 8])
    with pytest.raises(ValueError, match="Data leakage: 1 samples overlap between validation and test"):
        DataValidator.check_data_leakage(train_ids, val_ids, test_ids)

def test_check_data_leakage_empty_arrays():
    train_ids = np.array([])
    val_ids = np.array([])
    test_ids = np.array([])
    assert DataValidator.check_data_leakage(train_ids, val_ids, test_ids) is True

def test_check_data_leakage_multiple_overlaps():
    train_ids = np.array([1, 2, 3, 4])
    val_ids = np.array([3, 4, 5, 6])
    with pytest.raises(ValueError, match="Data leakage: 2 samples overlap between train and validation"):
        DataValidator.check_data_leakage(train_ids, val_ids)
