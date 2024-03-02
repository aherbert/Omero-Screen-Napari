from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.plate_handler import (
    ChannelDataManager,
    FlatfieldMaskManager,
)


# __________________________TESTING CSV FILE MANAGER________________________________
def test_csv_available_when_file_exists(tmp_path, csv_manager):
    csv_path = tmp_path / "omero_screen_data"
    csv_path.mkdir(exist_ok=True)
    csv_manager.csv_path = csv_path
    (csv_path / "123.csv").touch()
    assert csv_manager._csv_available() is True, "The csv file was not found."
    print(f"omero_data after execution: {csv_manager.omero_data.csv_path}")
    assert (
        csv_manager.omero_data.csv_path == csv_path / "123.csv"
    ), "The csv path was not set correctly."


def test_csv_unvailable_when_file_exists(tmp_path, csv_manager):
    csv_path = tmp_path / "omero_screen_data"
    csv_path.mkdir(exist_ok=True)
    csv_manager.csv_path = csv_path
    (csv_path / "456.csv").touch()
    assert not csv_manager._csv_available()


@pytest.mark.parametrize(
    "file_names,expected_file_name",
    [
        (
            ["not_relevant_file.csv", "final_data.csv", "final_data_cc.csv"],
            "final_data_cc.csv",
        ),
        (
            ["final_data_cc.csv", "final_data.csv", "not_relevant_file.csv"],
            "final_data_cc.csv",
        ),
        (
            ["final_data.csv", "not_relevant_file.csv", "final_data_cc.csv"],
            "final_data_cc.csv",
        ),
        (
            ["final_data.csv", "not_relevant_file.csv"],
            "final_data.csv",
        ),
    ],
)
def test_get_csv_file_success(
    csv_manager, mock_plate, file_names, expected_file_name
):
    csv_manager.plate = mock_plate(file_names)
    csv_manager._get_csv_file()
    actual_file_name = csv_manager.original_file.getName()
    assert (
        actual_file_name == expected_file_name
    ), f"Expected file '{expected_file_name}', but got '{actual_file_name}'."


def test_get_csv_file_failure(csv_manager, mock_plate, mock_omero_data):
    file_names = ["not_relevant_file1.csv", "not_relevant_file2.csv"]
    csv_manager.plate = mock_plate(file_names)
    with pytest.raises(ValueError) as exc_info:
        csv_manager._get_csv_file()
    assert "No suitable csv file found for the plate" in str(
        exc_info.value
    ), "Expected ValueError was not raised or message did not match."


def test_download_csv(csv_manager_with_mocked_file):
    # Given
    expected_file_path = csv_manager_with_mocked_file.csv_path / "123_data.csv"
    # When
    csv_manager_with_mocked_file._download_csv()
    # Then
    assert (
        expected_file_path.exists()
    ), f"The expected file {expected_file_path} does not exist."
    # Verify the file contents
    with open(expected_file_path, "rb") as file_on_disk:
        content = file_on_disk.read()
        assert (
            content == b"chunk1chunk2chunk3"
        ), "The file content does not match the expected content."


# __________________________TESTING CHANNEL DATA MANAGER________________________________


def test_error_when_no_map_annotations(mock_plate):
    plate = mock_plate(file_names=["file1.txt", "file2.txt"])
    channel_manager = ChannelDataManager(omero_data, plate)

    with pytest.raises(ValueError) as exc_info:
        channel_manager._get_map_ann()

    assert "No MapAnnotations found" in str(exc_info.value)


def test_error_when_wrong_map_annotations(mock_plate):
    map_annotations = [("key1", "value1"), ("key2", "value2")]
    plate = mock_plate(
        file_names=["file1.txt"], map_annotations=[map_annotations]
    )
    channel_manager = ChannelDataManager(
        omero_data, plate
    )  # Ensure omero_data is appropriately defined

    with pytest.raises(ValueError) as exc_info:
        channel_manager._get_map_ann()

    assert "No DAPI or Hoechst channel information found" in str(
        exc_info.value
    )


def test_found_map_annotations(mock_plate):
    map_annotations = [("DAPI", "1"), ("key2", "value2")]
    plate = mock_plate(
        file_names=["file1.txt"], map_annotations=[map_annotations]
    )
    channel_manager = ChannelDataManager(
        omero_data, plate
    )  # Ensure omero_data is appropriately defined
    channel_manager._get_map_ann()

    assert channel_manager.map_annotations == map_annotations


def test_filter_channel_data_with_space(mock_plate):
    map_annotations = [
        ("DAPI ", "1"),
        ("Tub", "2"),
        (" p21", "3"),
        ("EdU", "4"),
    ]
    plate = mock_plate(
        file_names=["file1.txt"], map_annotations=[map_annotations]
    )
    channel_manager = ChannelDataManager(omero_data, plate)
    channel_manager.map_annotations = map_annotations
    channel_manager._tidy_up_channel_data()
    assert channel_manager.channel_data == {
        "DAPI": "1",
        "Tub": "2",
        "p21": "3",
        "EdU": "4",
    }


def test_filter_channel_data_with_Hoechst(mock_plate):
    map_annotations = [
        ("Hoechst", "1"),
        ("Tub", "2"),
        (" p21", "3"),
        ("EdU", "4"),
    ]
    plate = mock_plate(
        file_names=["file1.txt"], map_annotations=[map_annotations]
    )
    channel_manager = ChannelDataManager(omero_data, plate)
    channel_manager.map_annotations = map_annotations
    channel_manager._tidy_up_channel_data()
    assert channel_manager.channel_data == {
        "DAPI": "1",
        "Tub": "2",
        "p21": "3",
        "EdU": "4",
    }


# __________________________TESTING FLATFIELD CORRECTION MASK MANAGER________________________________


def test_load_dataset_success(mock_conn):
    # Setup mock connection with a project and a dataset
    project_id = omero_data.project_id
    omero_data.plate_id = 123
    datasets = {"124": 457, "123": 456}  # Dataset name and its ID
    mock_connection = mock_conn(project_id, datasets)
    manager = FlatfieldMaskManager(omero_data, mock_connection)

    # Perform the test
    manager._load_dataset()

    # Verify the dataset is correctly assigned to omero_data.screen_dataset
    assert (
        omero_data.screen_dataset.getId() == 456
    ), "The dataset was not assigned."


def test_load_dataset_failure(mock_conn):
    # Setup mock connection with a project and a dataset
    project_id = omero_data.project_id
    omero_data.plate_id = 122
    datasets = {"124": 457, "123": 458}
    mock_connection = mock_conn(project_id, datasets)
    manager = FlatfieldMaskManager(omero_data, mock_connection)
    # Perform the test
    with pytest.raises(ValueError) as exc_info:
        manager._load_dataset()
    # Verify the error message
    assert "The plate  has not been assigned a dataset" in str(exc_info.value)


class MockImage:
    """
    Mock to supply images for testing the listChidlren method of the Omero DatasetWrapper.
    """

    def __init__(self, name):
        self._name = name

    def getName(self):
        return self._name

    def getId(self):
        return (
            "mock_id"  # Return a mock ID or vary this as needed for your tests
        )


def test_get_flatfieldmask_found(
    mock_image, mock_screen_dataset_factory, mock_conn
):
    omero_data.plate_id = 123
    mock_images = [
        MockImage("123_flatfield_masks"),
        MockImage("some_other_image"),
    ]
    omero_data.screen_dataset = mock_screen_dataset_factory(mock_images)
    mock_flatfield_obj = (
        MagicMock()
    )  # Simulate the image object returned by get_image
    mock_flatfield_array = np.random.rand(
        10, 10
    )  # Simulate a 10x10 flatfield array
    manager = FlatfieldMaskManager(omero_data, mock_conn)

    with patch(
        "omero_screen_napari.plate_handler.get_image",
        return_value=(mock_flatfield_obj, mock_flatfield_array),
    ) as mock_get_image:
        manager._get_flatfieldmask()
        mock_get_image.assert_called_once()

        assert manager.flatfield_array is not None
        assert manager.flatfield_array.shape == (10, 10)


def test_get_flatfieldmask_not_found(
    mock_image, mock_screen_dataset_factory, mock_conn
):
    omero_data.plate_id = 123
    mock_images = [
        MockImage("122_flatfield_masks"),
        MockImage("some_other_image"),
    ]
    omero_data.screen_dataset = mock_screen_dataset_factory(mock_images)
    manager = FlatfieldMaskManager(omero_data, mock_conn)

    with pytest.raises(ValueError) as exc_info:
        manager._get_flatfieldmask()
    # Verify the error message

    assert "No flatfieldmasks found in dataset" in str(exc_info.value)


@pytest.mark.parametrize(
    "key_variation",
    ["Dapi", "dapi", " HOECHST", "hoechst ", " dApI ", "HOECHST"],
)
def test_get_map_ann_with_annotations(
    mock_flatfield_obj, mock_flatfield_map_annotation, mock_conn, key_variation
):
    omero_data = MagicMock()
    manager = FlatfieldMaskManager(omero_data, mock_conn)
    manager.flatfield_obj = mock_flatfield_obj

    annotation_values = [( "some_value", key_variation), ("key2", "value2")]
    mock_flatfield_obj.listAnnotations.return_value = [
        mock_flatfield_map_annotation(annotation_values),
    ]

    manager._get_map_ann()

    normalized_values = [
    value.strip().lower()
    for _keykey, value in manager.flatfield_channels.items()]

    assert (
        "dapi" in normalized_values or "hoechst" in normalized_values
    ), f"Neither 'dapi' nor 'hoechst' found in processed keys: {normalized_values}"
    mock_flatfield_obj.listAnnotations.assert_called_once()


def test_get_map_ann_without_annotations(
    mock_flatfield_obj, mock_flatfield_map_annotation, mock_conn
):
    omero_data = MagicMock()
    manager = FlatfieldMaskManager(omero_data, mock_conn)
    manager.flatfield_obj = mock_flatfield_obj

    annotation_values = [("some_key", "some_value"), ("key2", "value2")]
    mock_flatfield_obj.listAnnotations.return_value = [
        mock_flatfield_map_annotation(annotation_values),
    ]

    with pytest.raises(ValueError) as excinfo:
        manager._get_map_ann()
    assert (
        "No DAPI or Hoechst channel information found for flatfieldmasks."
        in str(excinfo.value)
    )


def test_get_map_ann_without_appropriate_annotations(
    mock_flatfield_obj, mock_conn
):
    # Setup for a scenario with no appropriate map annotations
    manager = FlatfieldMaskManager(omero_data, mock_conn)
    manager.flatfield_obj = mock_flatfield_obj
    mock_flatfield_obj.listAnnotations.return_value = [
        MagicMock()
    ]  # No MapAnnotationWrapper instances

    with pytest.raises(ValueError) as excinfo:
        manager._get_map_ann()

    assert "No Flatfield Mask Channel info found in datset" in str(
        excinfo.value
    )


# tests for reverse channel function

# Test data for happy path scenarios
happy_path_data = [
    ("single_channel", {"channel_1": 1}, {"1": "channel_1"}),
    (
        "multiple_channels",
        {"channel_1": 1, "channel_2": 2},
        {"1": "channel_1", "2": "channel_2"},
    ),
    (
        "non_sequential",
        {"channel_1": 3, "channel_5": 1},
        {"3": "channel_1", "1": "channel_5"},
    ),
    (
        "with_extra_underscores",
        {"channel_extra_1": 1},
        {"1": "channel_extra_1"},
    ),
]

# Test data for edge cases
edge_cases_data = [
    ("empty_dict", {}, {}),
    (
        "non_numeric_values",
        {"channel_a": "a", "channel_b": "b"},
        {"a": "channel_a", "b": "channel_b"},
    ),
]

# Test data for error cases
error_cases_data = [
    ("non_dict", "not_a_dict", ValueError),
    ("list_instead_of_dict", ["channel_1", "channel_2"], ValueError),
]


@pytest.mark.parametrize(
    "test_id, flatfield_channels, expected_output", happy_path_data
)
def test_reverse_flatfield_channels_happy_path(
    test_id, flatfield_channels, mock_conn, expected_output
):
    # Arrange
    manager = FlatfieldMaskManager(omero_data, mock_conn)
    manager.flatfield_channels = flatfield_channels

    # Act
    result = manager.reverse_flatfield_channels()

    # Assert
    assert (
        result == expected_output
    ), f"Test {test_id} failed: {result} != {expected_output}"


@pytest.mark.parametrize("test_id, flatfield_channels, expected_output", edge_cases_data)
def test_reverse_flatfield_channels_edge_cases(test_id, flatfield_channels, mock_conn, expected_output):
    # Arrange
    manager = FlatfieldMaskManager(omero_data, mock_conn)
    manager.flatfield_channels = flatfield_channels

    # Act
    result = manager.reverse_flatfield_channels()

    # Assert
    assert result == expected_output, f"Test {test_id} failed: {result} != {expected_output}"

@pytest.mark.parametrize("test_id, flatfield_channels, expected_exception", error_cases_data)
def test_reverse_flatfield_channels_error_cases(test_id, flatfield_channels, mock_conn, expected_exception):
    # Arrange
    manager = FlatfieldMaskManager(omero_data, mock_conn)

    # Act / Assert
    with pytest.raises(expected_exception) as exc_info:
        manager.flatfield_channels = flatfield_channels
        manager.reverse_flatfield_channels()

    # Optionally, assert on the exception message if you want to ensure it's specific enough
    assert "flatfield_channels must be a dictionary" in str(exc_info.value)


def test_flatfieldmask_succes(mock_conn):
    manager = FlatfieldMaskManager(omero_data, mock_conn)
    manager.flatfield_channels = {'channel_1': 'DAPI', 'channel_2': 'Tub'}
    omero_data.channel_data = {'DAPI': '1', 'Tub': '2'}
    manager._check_flatfieldmask()

def test_flatfieldmask_failue(mock_conn):
    manager = FlatfieldMaskManager(omero_data, mock_conn)
    manager.flatfield_channels = {'channel_1': 'DAPI', 'channel_2': 'EdU'}
    omero_data.channel_data = {'DAPI': '1', 'Tub': '2'}
    with pytest.raises(ValueError) as exc_info:
        manager._check_flatfieldmask()
    assert "Inconsistency found: flatfield_mask and plate_map have different channels" in str(exc_info.value)