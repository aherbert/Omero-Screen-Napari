import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

os.environ["USE_LOCAL_ENV"] = "1"

from omero_screen_napari.utils import omero_connect  # noqa: E402, I001
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import (
    UserInput,
    ChannelDataParser,
    CsvFileParser,
    FlatfieldMaskParser,
    PixelSizeParser,
    ScaleIntensityParser,
    WellDataParser,
    ImageParser,
    parse_plate_data,
    parse_omero_data,
)


@omero_connect
def test_user_input(plate_id=53, conn=None):
    omero_data = OmeroData()
    plate_id = 53
    user_input = UserInput(omero_data, plate_id, "C2", "All", conn)
    user_input.parse_data()
    assert omero_data.well_pos_list == ["C2"]
    assert omero_data.image_index == [0, 1, 2]


@pytest.fixture
def cleanup_csv_directory():
    """
    Fixture to prepare a temporary directory for test and clean it up afterward.
    Yields the temporary path to be used in the test.
    """
    tmp_path = Path.home() / "omero-napari-data_tests"
    # Setup phase: yield the temporary directory to the test.
    yield tmp_path
    if tmp_path.exists():
        print(f"Cleaning up {tmp_path}")
        shutil.rmtree(tmp_path)


@omero_connect
def test_csv_download(cleanup_csv_directory, conn=None):
    """
    Test the csv download function
    """
    omero_data = OmeroData()
    omero_data.plate_id = 53
    omero_data.data_path = cleanup_csv_directory
    plate = conn.getObject("Plate", 53)
    assert plate is not None, "Failed to retrieve plate object."
    csv_manager = CsvFileParser(omero_data)
    csv_manager._plate = plate
    csv_manager.parse_csv()
    df = pl.read_csv(csv_manager._csv_file_path)
    assert len(df) == 1006
    assert str(omero_data.csv_path).endswith("53_cc.csv")


@omero_connect
def test_channel_data_manager(conn=None):
    """
    Test the channel data manager
    """
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    channel_manager = ChannelDataParser(omero_data)
    channel_manager._plate = plate
    channel_manager.parse_channel_data()
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
    channel_manager = ChannelDataParser(omero_data)
    channel_manager._plate = plate
    with pytest.raises(ValueError) as exc_info:
        channel_manager.parse_channel_data()
    assert "No MapAnnotations found" in str(exc_info.value)


@omero_connect
def test_noDapi_channel_data_manager(conn=None):
    """
    Test the channel data manager on a plate without channel data
    """
    omero_data.reset()
    omero_data.plate_id = 201  # plate without DAPI channel
    plate = conn.getObject("Plate", 201)
    channel_manager = ChannelDataParser(omero_data)
    channel_manager._plate = plate
    with pytest.raises(ValueError) as exc_info:
        channel_manager.parse_channel_data()
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
    manager = FlatfieldMaskParser(omero_data, conn)
    manager.parse_flatfieldmask()
    assert omero_data.flatfield_masks.shape == (
        1,
        1,
        1080,
        1080,
        4,
    ), "Failed to retrieve flatfield mask"


@omero_connect
def test_scale_intensity_manager(cleanup_csv_directory, conn=None):
    """
    Test the scale intensity manager
    """
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    csv_manager = CsvFileParser(omero_data)
    csv_manager._plate = plate
    csv_manager.csv_path = cleanup_csv_directory
    csv_manager.parse_csv()
    omero_data.channel_data = {
        "DAPI": "0",
        "Tub": "1",
        "p21": "2",
        "EdU": "3",
    }
    manager = ScaleIntensityParser(omero_data)
    manager.parse_intensities()
    assert {
        0: (280, 15378),
        1: (3696, 20275),
        2: (2006, 4373),
        3: (236, 4728),
    } == omero_data.intensities


@omero_connect
def test_pixel_size_manager(conn=None):
    """
    Test the pixel size manager
    """
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    pixel_manager = PixelSizeParser(omero_data)
    pixel_manager._plate = plate
    pixel_manager.parse_pixel_size_values()
    assert omero_data.pixel_size == (1.2, 1.2), "Failed to retrieve pixel size"


@omero_connect
def test_parse_plate_data_success(cleanup_csv_directory, conn=None):
    """
    Test the parse plate data function
    """
    omero_data.reset()
    omero_data.data_path = cleanup_csv_directory
    parse_plate_data(
        omero_data, plate_id=53, well_pos="C2", image_input="All", conn=conn
    )
    # print(f"Data Path is {omero_data.data_path}")
    # print(f"CSV Path is {omero_data.csv_path}")
    # print(f"Project is {omero_data.project_id}")
    assert omero_data.plate_id == 53, "Failed to retrieve plate id"
    assert omero_data.channel_data == {
        "DAPI": "0",
        "Tub": "1",
        "p21": "2",
        "EdU": "3",
    }
    assert omero_data.flatfield_masks.shape == (
        1,
        1,
        1080,
        1080,
        4,
    ), "Failed to retrieve flatfield mask"
    assert {
        0: (280, 15378),
        1: (3696, 20275),
        2: (2006, 4373),
        3: (236, 4728),
    } == omero_data.intensities, "Failed to retrieve intensities"
    assert omero_data.pixel_size == (1.2, 1.2), "Failed to retrieve pixel size"


@omero_connect
def test_well_data_parser(conn=None):
    omero_data.reset()
    omero_data.plate_id = 53
    plate = conn.getObject("Plate", 53)
    well_data = WellDataParser(omero_data, "C2")
    well_data._plate = plate
    well_data._parse_well_object()
    well_data._get_well_metadata()
    assert well_data._metadata == {"cell_line": "RPE-1", "condition": "SCR"}


@omero_connect
def test_image_parser(conn=None):
    omero_data.reset()
    parse_plate_data(
        omero_data, plate_id=53, well_pos="C2", image_input="All", conn=conn
    )
    well_data = WellDataParser(omero_data, "C2")
    well_data.parse_well()
    image_data = ImageParser(omero_data, MagicMock(), conn)
    image_data._well = conn.getObject("Well", 105)
    image_data._parse_images()
    assert omero_data.images.shape == (3, 1080, 1080, 4)


@omero_connect
def test_label_parser(conn=None):
    omero_data.reset()
    parse_plate_data(
        omero_data, plate_id=53, well_pos="C2", image_input="All", conn=conn
    )
    well_data = WellDataParser(omero_data, "C2")
    well_data.parse_well()
    image_data = ImageParser(omero_data, MagicMock(), conn)
    image_data._well = conn.getObject("Well", 105)
    image_data._parse_images()
    image_data._parse_labels()


@omero_connect
def test_parse_omero_data(conn=None):
    omero_data.reset()
    plate_id = 53
    well_pos = "C2, C5"
    image_input = "All"
    parse_omero_data(omero_data, plate_id, well_pos, image_input, conn)
    assert omero_data.images.shape == (6, 1080, 1080, 4)
    assert len(omero_data.image_ids) == 6


@omero_connect
def test_parse_omero_data_twice(conn=None):
    omero_data.reset()
    plate_id = 53
    well_pos = "C2"
    image_input = "1"
    parse_omero_data(omero_data, plate_id, well_pos, image_input, conn)
    assert omero_data.images.shape == (1, 1080, 1080, 4)
    assert omero_data.labels.shape == (1, 1080, 1080, 2)

    assert len(omero_data.image_ids) == 1
    well_pos = "C2, C5"
    image_input = "0-1"
    parse_omero_data(omero_data, plate_id, well_pos, image_input, conn)
    assert omero_data.images.shape == (4, 1080, 1080, 4)
    assert omero_data.labels.shape == (4, 1080, 1080, 2)
