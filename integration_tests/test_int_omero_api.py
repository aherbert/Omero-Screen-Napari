import os
import shutil

import polars as pl
import pytest

os.environ["USE_LOCAL_ENV"] = "1"

from omero_screen_napari.plate_handler import (
    CsvFileManager,
    ChannelDataManager,
    FlatfieldMaskManager,
    ScaleIntensityManager,
    PixelSizeManager
) 
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
    csv_manager._data_path = cleanup_csv_directory
    csv_manager.handle_csv()
    df = pl.read_csv(csv_manager._csv_file_path)
    assert len(df) == 1006
    assert str(omero_data.csv_path).endswith('test_csv_download0/53_cc.csv')


    
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
    print(omero_data.channel_data)
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




@omero_connect
def test_flatfield_channel_data_manager(conn=None):
    """
    Test the channel data manager on a plate without channel data
    """
    omero_data.reset()
    omero_data.channel_data = {
        "DAPI": "0",
        "Tub": "1",
        "p21": "2",
        "EdU": "3",
    }
    omero_data.plate_id = 53
    manager = FlatfieldMaskManager(omero_data, conn)
    manager.get_flatfieldmask()
    assert omero_data.flatfield_mask.shape == (1, 1, 1080, 1080, 4), "Failed to retrieve flatfield mask"
@omero_connect
def test_scale_intensity_manager(cleanup_csv_directory, conn=None):
    """
    Test the scale intensity manager
    """
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    csv_manager = CsvFileManager(omero_data, plate)
    csv_manager.csv_path = cleanup_csv_directory
    csv_manager.handle_csv()
    omero_data.channel_data = {
        "DAPI": "0",
        "Tub": "1",
        "p21": "2",
        "EdU": "3",
    }
    manager = ScaleIntensityManager(omero_data)
    manager.get_intensities()
    assert {'DAPI': (280, 15378), 'Tub': (3696, 20275), 'p21': (2006, 4373), 'EdU': (236, 4728)} == omero_data.intensities

@omero_connect
def test_pixel_size_manager(conn=None):
    """
    Test the pixel size manager
    """
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    pixel_manager = PixelSizeManager(omero_data, plate)
    pixel_manager.get_pixel_size_values()
    assert omero_data.pixel_size == (1.2, 1.2), "Failed to retrieve pixel size"