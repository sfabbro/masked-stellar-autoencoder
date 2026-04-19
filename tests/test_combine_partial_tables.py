import pytest
from unittest.mock import patch, MagicMock
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data import combine_partial_tables

@patch("data.combine_partial_tables.glob.glob")
@patch("data.combine_partial_tables.fits.open")
@patch("data.combine_partial_tables.h5py.File")
def test_process_tables_exception_handling(mock_h5py, mock_fits_open, mock_glob, capsys):
    # Setup mock files
    mock_glob.return_value = ['partialtable_good.fits', 'partialtable_bad.fits']

    # Mock fits.open to succeed for the good file and raise an exception for the bad file
    def mock_fits_open_side_effect(filename, **kwargs):
        if 'bad' in filename:
            raise ValueError("Corrupt FITS file")

        mock_hdul = MagicMock()
        mock_hdul.__len__.return_value = 2
        mock_data = np.zeros(10)
        mock_hdul[1].data = mock_data

        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_hdul
        return mock_context

    mock_fits_open.side_effect = mock_fits_open_side_effect

    # Run the function
    combine_partial_tables.process_tables()

    # Verify the output contains the expected error message for the bad file
    captured = capsys.readouterr()
    assert "Error processing partialtable_bad.fits: Corrupt FITS file, skipping" in captured.out

    # Check that h5py.File was called for the good file
    assert mock_h5py.call_count == 1


@patch("data.combine_partial_tables.glob.glob")
@patch("data.combine_partial_tables.fits.open")
def test_process_tables_no_files_found(mock_fits_open, mock_glob):
    mock_glob.return_value = []
    with pytest.raises(FileNotFoundError, match="No FITS files found matching pattern"):
        combine_partial_tables.process_tables()

@patch("data.combine_partial_tables.glob.glob")
@patch("data.combine_partial_tables.fits.open")
@patch("data.combine_partial_tables.h5py.File")
def test_process_tables_insufficient_hdus(mock_h5py, mock_fits_open, mock_glob, capsys):
    mock_glob.return_value = ['partialtable_nohdu.fits']

    def mock_fits_open_side_effect(filename, **kwargs):
        mock_hdul = MagicMock()
        mock_hdul.__len__.return_value = 1  # Less than 2
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_hdul
        return mock_context

    mock_fits_open.side_effect = mock_fits_open_side_effect

    combine_partial_tables.process_tables()

    captured = capsys.readouterr()
    assert "Warning: partialtable_nohdu.fits has insufficient HDUs, skipping" in captured.out
    assert mock_h5py.call_count == 0

@patch("data.combine_partial_tables.glob.glob")
@patch("data.combine_partial_tables.fits.open")
@patch("data.combine_partial_tables.h5py.File")
def test_process_tables_no_data(mock_h5py, mock_fits_open, mock_glob, capsys):
    mock_glob.return_value = ['partialtable_nodata.fits']

    def mock_fits_open_side_effect(filename, **kwargs):
        mock_hdul = MagicMock()
        mock_hdul.__len__.return_value = 2
        mock_hdul[1].data = None  # No data
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_hdul
        return mock_context

    mock_fits_open.side_effect = mock_fits_open_side_effect

    combine_partial_tables.process_tables()

    captured = capsys.readouterr()
    assert "Warning: partialtable_nodata.fits contains no data, skipping" in captured.out
    assert mock_h5py.call_count == 0
