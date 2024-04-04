from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omero_screen_napari.gallery_api import CroppedImageParser, UserData
from omero_screen_napari.omero_data_singleton import omero_data


# ----------------------------------TESTS for USERDATA--------------------------
def test_user_data_valid():
    valid_data = {
        "well": "A1",
        "segmentation": "method",
        "replacement": "strategy",
        "crop_size": 100,
        "cellcycle": "G1",
        "columns": 5,
        "rows": 5,
        "contour": True,
        "channels": ["DAPI"],
    }
    omero_data.channel_data = {"DAPI": 1, "EdU": 2}
    UserData.set_omero_data_channel_keys(list(omero_data.channel_data.keys()))
    try:
        user_data = UserData(**valid_data)
        assert user_data is not None
    except ValueError:
        pytest.fail("Unexpected ValueError with valid input data")


@pytest.mark.parametrize(
    "channels, value",
    [
        ("channels", ["DAPI", "EdU"]),
        ("channels", ["DAPI", "P21"]),
        ("channels", ["DAPI", "p21", "Tub"]),
    ],
)
def test_user_data_invalid(channels, value):
    invalid_data = {
        "segmentation": "method",
        "replacement": "strategy",
        "crop_size": 100,
        "cellcycle": "G1",
        "columns": 5,
        "rows": 5,
        "contour": True,
        channels: value,
    }
    omero_data.channel_data = {"DAPI": 1, "p21": 2}
    UserData.set_omero_data_channel_keys(list(omero_data.channel_data.keys()))
    with pytest.raises(ValueError) as excinfo:
        UserData(**invalid_data)
        assert f"{value} is not a valid key for {channels}" in str(
            excinfo.value
        )


# ----------------------------------TESTS for CROPPEDIMAGEPARSER--------------------------


@pytest.fixture
def mock_user_data(mock_omero_data):
    # Return an instance of UserData with the required fields for your tests
    UserData.set_omero_data_channel_keys(["DAPI"])
    return UserData(
        well="A1",
        segmentation="default_segmentation",
        replacement="default_replacement",
        crop_size=100,
        cellcycle="G1",
        columns=4,
        rows=4,
        contour=True,
        channels=[
            "DAPI"
        ],  # Adjust according to the actual model and test requirements
    )


def test_select_wells_success(mock_user_data):
    # Mocking the omero_data
    mock_omero_data = MagicMock()
    mock_omero_data.well_list = [
        MagicMock(getWellPos=lambda: "A1"),
        MagicMock(getWellPos=lambda: "A2"),
    ]
    mock_omero_data.images = np.random.rand(
        6, 1080, 1080, 2
    )  # Assuming 3 images per well for simplicity

    mock_user_data.well = "A1"

    # Instance of your parser
    parser = CroppedImageParser(
        omero_data=mock_omero_data, user_data=mock_user_data
    )

    # Execute
    selected_images = parser.select_wells()

    # Assert
    assert selected_images.shape == (
        3,
        1080,
        1080,
        2,
    ), "Should select 3 images for well A1"


def test_select_wells_unknown_well(mock_user_data):
    mock_omero_data = MagicMock()
    mock_omero_data.well_list = [
        MagicMock(getWellPos=lambda: "A1"),
        MagicMock(getWellPos=lambda: "A2"),
    ]
    mock_user_data.well = "A3"  # Non-existent well

    parser = CroppedImageParser(
        omero_data=mock_omero_data, user_data=mock_user_data
    )

    with pytest.raises(ValueError) as exc_info:
        parser.select_wells()

    assert "The selected well A3 has not been loaded from the plate" in str(
        exc_info.value
    ), "Should raise ValueError for unknown well"
