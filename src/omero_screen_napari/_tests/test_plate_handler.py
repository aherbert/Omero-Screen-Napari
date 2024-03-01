from unittest.mock import MagicMock

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
    datasets = {"123": 456}  # Dataset name and its ID
    mock_connection = mock_conn(project_id, datasets)
    manager = FlatfieldMaskManager(omero_data, mock_connection)

    # Perform the test
    manager._load_dataset()

    # Verify the dataset is correctly assigned to omero_data.screen_dataset
    assert omero_data.screen_dataset is not None
    assert omero_data.screen_dataset.getId() == 456


def test_load_dataset_failure(mock_conn):
    # Setup mock connection with a project and a dataset
    project_id = omero_data.project_id
    omero_data.plate_id = 122
    datasets = {"123": 456}
    mock_connection = mock_conn(project_id, datasets)
    manager = FlatfieldMaskManager(omero_data, mock_connection)
    # Perform the test
    with pytest.raises(ValueError) as exc_info:
        manager._load_dataset()
    # Verify the error message
    assert "The plate  has not been assigned a dataset" in str(exc_info.value)
