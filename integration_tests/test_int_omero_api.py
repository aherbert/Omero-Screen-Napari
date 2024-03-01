import os
import shutil

import polars as pl
import pytest

os.environ["USE_LOCAL_ENV"] = "1"

from omero_screen_napari.plate_handler import (
    CsvFileManager,
    ChannelDataManager,
)  # noqa: E402, I001
from omero_screen_napari._omero_utils import omero_connect  # noqa: E402, I001
from omero_screen_napari.omero_data_singleton import omero_data  # noqa: E402, I001


@pytest.fixture
def cleanup_csv_directory(tmp_path):
    """
    Fixture to prepare a temporary directory for test and clean it up afterward.
    Yields the temporary path to be used in the test.
    """
    # Setup phase: yield the temporary directory to the test.
    yield tmp_path

    # Teardown phase: cleanup the directory after test execution.
    shutil.rmtree(tmp_path)


@omero_connect
def test_csv_download(cleanup_csv_directory, conn=None):
    """
    Test the csv download function
    """
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    assert plate is not None, "Failed to retrieve plate object."
    csv_manager = CsvFileManager(omero_data, plate)
    csv_manager.csv_path = cleanup_csv_directory
    csv_manager.handle_csv()
    df = pl.read_csv(csv_manager.csv_path / "53_cc.csv")
    assert len(df) == 1006


@omero_connect
def test_channel_data_manager(conn=None):
    """
    Test the channel data manager
    """
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    channel_manager = ChannelDataManager(omero_data, plate)
    channel_manager.get_channel_data()
    assert omero_data.channel_data == {
        "DAPI": "0",
        "Tub": "1",
        "p21": "2",
        "EdU": "3",
    }


@omero_connect
def test_no_channel_data_manager(conn=None):
    """
    Test the channel data manager on a plate without channel data
    """
    omero_data.reset()
    omero_data.plate_id = 2  # Plate 2 has no channel data
    plate = conn.getObject("Plate", 2)
    channel_manager = ChannelDataManager(omero_data, plate)
    with pytest.raises(ValueError) as exc_info:
        channel_manager.get_channel_data()
    assert "No MapAnnotations found" in str(exc_info.value)


@omero_connect
def test_noDapi_channel_data_manager(conn=None):
    """
    Test the channel data manager on a plate without channel data
    """
    omero_data.reset()
    omero_data.plate_id = 201  # plate without DAPI channel
    plate = conn.getObject("Plate", 201)
    channel_manager = ChannelDataManager(omero_data, plate)
    with pytest.raises(ValueError) as exc_info:
        channel_manager.get_channel_data()
    assert "No DAPI or Hoechst channel information found" in str(
        exc_info.value
    )
