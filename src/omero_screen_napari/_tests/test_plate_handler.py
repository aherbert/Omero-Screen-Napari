from unittest.mock import MagicMock, patch
import logging
import numpy as np
import polars as pl
import pytest

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.plate_handler import (
    UserInput,
    ChannelDataParser,
    CsvFileParser,
    FlatfieldMaskParser,
    PixelSizeParser,
    ScaleIntensityParser,
    WellDataParser,
)
# __________________________TESTING INPUT DATA PARSER________________________________
def test_check_plate_exist(mock_conn, caplog):
    plate_id = 123
    caplog.set_level(logging.INFO)
    mock_connection = mock_conn(plate_id, MagicMock())
    user_input = UserInput(MagicMock(), plate_id, MagicMock(), MagicMock(), mock_connection)
    user_input.check_plate_id()
    assert 'Processing plate with ID 123' in caplog.text

def test_check_plate_doesnotexist(mock_conn):
    plate_id = 124
    mock_connection = mock_conn(plate_id, MagicMock(), none_plate_id=124)
    user_input = UserInput(MagicMock(), plate_id, MagicMock(), MagicMock(), mock_connection)
    with pytest.raises(ValueError) as exc_info:
        user_input.check_plate_id()
    assert "Plate with ID 124 does not exist" in str(exc_info.value)
def test_parse_image_number(mock_plate_for_input_test):
    user_input = UserInput(MagicMock(), 123, MagicMock(), MagicMock(), MagicMock())
    user_input._plate = mock_plate_for_input_test
    user_input.parse_image_number()
    assert user_input._image_number == 5

@pytest.mark.parametrize("well_pos,expected_exception", [
    ("A1,B2,C3", None),  # valid input, no exception expected
    ("A12,H12", None),  # valid edge cases
    ("A0,B13,Z1", ValueError),  # invalid: A0 and B13 out of range, Z1 invalid row
    ("", ValueError),  # invalid: empty string
    ("A1, B 2, C3", ValueError)  # invalid: space in well position
])

def test_well_data_parser(well_pos, expected_exception):
# sourcery skip: no-conditionals-in-tests
    if expected_exception:
        with pytest.raises(expected_exception):
            user_input = UserInput(MagicMock(), 1, well_pos, MagicMock(), MagicMock())
            user_input.well_data_parser()
    else:
        user_input = UserInput(MagicMock(), 1, well_pos, MagicMock(), MagicMock())
        user_input.well_data_parser()
        # Ensure the well positions are correctly parsed and stored in _well_pos_list
        expected_well_pos_list = [item.strip() for item in well_pos.split(",")]
        assert user_input._well_pos_list == expected_well_pos_list, "The well positions were not parsed correctly"

@pytest.mark.parametrize("images_input,expected_output,image_number", [
    ("all", list(range(10)), 10),  # Assuming `_image_number` is 10
    ("1-3", [1, 2, 3], 5),  # Range input
    ("1, 3, 5", [1, 3, 5], 5),  # List input
    ("1", [1], 5),  # Single number input
    ("1-3, 5", ValueError, 5),  # Invalid format (mixed range and list without proper handling in the method)
    ("abc", ValueError, 5),  # Completely invalid input
])
def test_image_index_parser(images_input, expected_output, image_number, mock_omero_data, mock_conn):
    user_input = UserInput(mock_omero_data, 1, "A1,B2,C3", images_input, mock_conn)
    user_input._image_number = image_number  # Set the `_image_number` attribute for the test

# sourcery skip: no-conditionals-in-tests
    if expected_output is ValueError:
        with pytest.raises(ValueError):
            user_input.image_index_parser()
    else:
        user_input.image_index_parser()
        assert user_input._image_index == expected_output, f"Expected {expected_output}, got {user_input._image_index}"


# __________________________TESTING CSV FILE MANAGER________________________________
def test_csv_available_when_file_exists(tmp_path, mock_omero_data):
    data_path = tmp_path / "omero_screen_data"
    mock_omero_data.data_path = data_path
    mock_omero_data.data_path.mkdir(exist_ok=True)
    csv_manager = CsvFileParser(mock_omero_data)
    tmp_file = data_path / "123.csv"
    tmp_file.touch()
    assert csv_manager._csv_available() is True, "The csv file was not found."
    assert (
        csv_manager._csv_file_path == tmp_file
    ), "The csv path was not set correctly."


def test_csv_unvailable_when_file_exists(tmp_path, mock_omero_data):
    data_path = tmp_path / "omero_screen_data"
    mock_omero_data.data_path = data_path
    mock_omero_data.data_path.mkdir(exist_ok=True)
    csv_manager = CsvFileParser(mock_omero_data)
    tmp_file = data_path / "124.csv"
    tmp_file.touch()
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
    mock_omero_data, mock_plate, file_names, expected_file_name
):
    csv_manager = CsvFileParser(mock_omero_data)
    csv_manager._plate = mock_plate(file_names)
    csv_manager._get_csv_file()
    actual_file_name = csv_manager._original_file.getName()
    assert (
        actual_file_name == expected_file_name
    ), f"Expected file {expected_file_name}, but got '{actual_file_name}'."


def test_get_csv_file_failure(mock_plate, mock_omero_data):
    file_names = ["not_relevant_file1.csv", "not_relevant_file2.csv"]
    csv_manager = CsvFileParser(mock_omero_data)
    csv_manager._plate = mock_plate(file_names)
    with pytest.raises(ValueError) as exc_info:
        csv_manager._get_csv_file()
    assert "No suitable csv file found for the plate" in str(
        exc_info.value
    ), "Expected ValueError was not raised or message did not match."


def test_download_csv(csv_manager_with_mocked_file):
    # Given
    expected_file_path = (
        csv_manager_with_mocked_file._data_path / "123_data.csv"
    )
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
    channel_manager = ChannelDataParser(omero_data)
    channel_manager._plate = plate
    with pytest.raises(ValueError) as exc_info:
        channel_manager._get_map_ann()

    assert "No MapAnnotations found" in str(exc_info.value)


def test_error_when_wrong_map_annotations(mock_plate):
    map_annotations = [("key1", "value1"), ("key2", "value2")]
    plate = mock_plate(
        file_names=["file1.txt"], map_annotations=[map_annotations]
    )
    channel_manager = ChannelDataParser(
        omero_data
    )  # Ensure omero_data is appropriately defined
    channel_manager._plate = plate
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
    channel_manager = ChannelDataParser(
        omero_data
    )  # Ensure omero_data is appropriately defined
    channel_manager._plate = plate
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
    channel_manager = ChannelDataParser(omero_data)
    channel_manager._plate = plate
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
    channel_manager = ChannelDataParser(omero_data)
    channel_manager._plate = plate
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
    manager = FlatfieldMaskParser(omero_data, mock_connection)

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
    manager = FlatfieldMaskParser(omero_data, mock_connection)
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
    manager = FlatfieldMaskParser(omero_data, mock_conn)

    with patch(
        "omero_screen_napari.plate_handler.get_image",
        return_value=(mock_flatfield_obj, mock_flatfield_array),
    ) as mock_get_image:
        manager._get_flatfieldmask()
        mock_get_image.assert_called_once()

        assert manager._flatfield_array is not None
        assert manager._flatfield_array.shape == (10, 10)


def test_get_flatfieldmask_not_found(
    mock_image, mock_screen_dataset_factory, mock_conn
):
    omero_data.plate_id = 123
    mock_images = [
        MockImage("122_flatfield_masks"),
        MockImage("some_other_image"),
    ]
    omero_data.screen_dataset = mock_screen_dataset_factory(mock_images)
    manager = FlatfieldMaskParser(omero_data, mock_conn)

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
    manager = FlatfieldMaskParser(omero_data, mock_conn)
    manager._flatfield_obj = mock_flatfield_obj

    annotation_values = [("some_value", key_variation), ("key2", "value2")]
    mock_flatfield_obj.listAnnotations.return_value = [
        mock_flatfield_map_annotation(annotation_values),
    ]

    manager._get_map_ann()

    normalized_values = [
        value.strip().lower()
        for _keykey, value in manager._flatfield_channels.items()
    ]

    assert (
        "dapi" in normalized_values or "hoechst" in normalized_values
    ), f"Neither 'dapi' nor 'hoechst' found in processed keys: {normalized_values}"
    mock_flatfield_obj.listAnnotations.assert_called_once()


def test_get_map_ann_without_annotations(
    mock_flatfield_obj, mock_flatfield_map_annotation, mock_conn
):
    omero_data = MagicMock()
    manager = FlatfieldMaskParser(omero_data, mock_conn)
    manager._flatfield_obj = mock_flatfield_obj

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
    manager = FlatfieldMaskParser(omero_data, mock_conn)
    manager._flatfield_obj = mock_flatfield_obj
    mock_flatfield_obj.listAnnotations.return_value = [
        MagicMock()
    ]  # No MapAnnotationWrapper instances

    with pytest.raises(ValueError) as excinfo:
        manager._get_map_ann()

    assert "No Flatfield Mask Channel info found in datset" in str(
        excinfo.value
    )


def test_flatfieldmask_succes(mock_conn):
    manager = FlatfieldMaskParser(omero_data, mock_conn)
    manager._flatfield_channels = {"channel_1": "DAPI", "channel_2": "Tub"}
    omero_data.channel_data = {"DAPI": "1", "Tub": "2"}
    manager._check_flatfieldmask()


def test_flatfieldmask_failue(mock_conn):
    manager = FlatfieldMaskParser(omero_data, mock_conn)
    manager._flatfield_channels = {"channel_1": "DAPI", "channel_2": "EdU"}
    omero_data.channel_data = {"DAPI": "1", "Tub": "2"}
    with pytest.raises(ValueError) as exc_info:
        manager._check_flatfieldmask()
    assert (
        "Inconsistency found: flatfield_mask and plate_map have different channels"
        in str(exc_info.value)
    )


# __________________________TESTING INTENSITY SCALE MANAGER________________________________


# test _extract_and_assign_values
@pytest.fixture
def lf_nucleus():
    return pl.LazyFrame(
        {
            "Image": ["Image1", "Image2", "Image3"],
            "intensity_max_DAPI_nucleus": [100, 200, 300],
            "intensity_min_DAPI_nucleus": [10, 20, 30],
            "intensity_max_p21_nucleus": [200, 300, 400],
            "intensity_min_p21_nucleus": [20, 30, 30],
        }
    )


@pytest.fixture
def lf_cell():
    return pl.LazyFrame(
        {
            "Image": ["Image1", "Image2", "Image3"],
            "intensity_max_DAPI_nucleus": [100, 200, 300],
            "intensity_min_DAPI_nucleus": [10, 20, 30],
            "intensity_max_DAPI_cell": [200, 300, 400],
            "intensity_min_DAPI_cell": [20, 30, 40],
            "intensity_max_p21_cell": [300, 400, 500],
            "intensity_min_p21_cell": [30, 40, 50],
            "intensity_max_p21_nucleus": [200, 300, 300],
            "intensity_min_p21_nucleus": [20, 30, 30],
        }
    )


@pytest.fixture
def lf_wron_col():
    return pl.LazyFrame(
        {
            "Image": ["Image1", "Image2", "Image3"],
            "intensity_nucleus": [100, 200, 300],
            "intensity_min_nucleus": [10, 20, 30],
        }
    )


@pytest.fixture
def lf_none():
    return pl.LazyFrame(
        {
            "Image": ["Image1", "Image2", "Image3"],
        }
    )


def test_set_keyword_nucleus(lf_nucleus):
    manager = ScaleIntensityParser(MagicMock())
    manager._plate_data = lf_nucleus
    manager._set_keyword()
    assert (
        manager._keyword == "_nucleus"
    ), "The keyword has not been set to '_nucleus'"


def test_set_keyword_cell(lf_cell):
    manager = ScaleIntensityParser(MagicMock())
    manager._plate_data = lf_cell
    manager._set_keyword()
    assert (
        manager._keyword == "_cell"
    ), "The keyword has not been set to '_cell'"


def test_set_keyword_none(lf_none):
    manager = ScaleIntensityParser(MagicMock())
    manager._plate_data = lf_none
    with pytest.raises(ValueError) as exc_info:
        manager._set_keyword()
    assert (
        "Neither '_cell' nor '_nucleus' is present in the dataframe columns."
        in str(exc_info.value)
    )


def test_set_keyword_noframe(lf_none):
    manager = ScaleIntensityParser(MagicMock())
    manager._plate_data = None
    with pytest.raises(ValueError) as exc_info:
        manager._set_keyword()
    assert "Dataframe 'plate_data' does not exist." in str(exc_info.value)


def test_get_values_nucleus(lf_nucleus):
    manager = ScaleIntensityParser(MagicMock())
    manager._omero_data.channel_data = {"DAPI": "1", "p21": "2"}
    manager._plate_data = lf_nucleus
    manager._keyword = "_nucleus"
    manager._get_values()
    assert manager._intensities == {
        1 : (10, 200),
        2 : (20, 300),
    }, "The max_values dictionary does not match the expected values."


def test_get_values_cell(lf_cell):
    manager = ScaleIntensityParser(MagicMock())
    manager._omero_data.channel_data = {"DAPI": "1", "p21": "2"}
    manager._plate_data = lf_cell
    manager._keyword = "_cell"
    manager._get_values()
    assert manager._intensities == {
         1 : (20, 300),
         2 : (30, 400),
    }, "The max_values dictionary does not match the expected values."


def test_get_values_wrong_col(lf_wron_col):
    manager = ScaleIntensityParser(MagicMock())
    manager._omero_data.channel_data = {"DAPI": "1", "p21": "2"}
    manager._plate_data = lf_wron_col
    manager._keyword = "_nucleus"
    with pytest.raises(ValueError) as exc_info:
        manager._get_values()
    assert (
        "Column 'intensity_max_DAPI_nucleus' not found in DataFrame."
        in str(exc_info.value)
    )


# __________________________TESTING PIXEL SIZE MANAGER________________________________


def test_check_wells_and_images_with_wells(mock_plate):
    mock_plate = mock_plate(["mock1"], wells_count=4, img_number=5)
    manager = PixelSizeParser(MagicMock())
    manager._plate = mock_plate
    manager._check_wells_and_images()
    assert len(manager._random_wells) == 2
    assert len(manager._random_images) == 2


def test_check_wells_and_images_without_wells(mock_plate):
    mock_plate = mock_plate(["mock1"], wells_count=0, img_number=5)
    manager = PixelSizeParser(MagicMock())
    manager._plate = mock_plate
    with pytest.raises(ValueError) as exc_info:
        manager._check_wells_and_images()
    assert "No wells found in the plate." in str(exc_info.value)


def test_check_wells_and_images_without_images(mock_plate):
    mock_plate = mock_plate(["mock1"], wells_count=5, img_number=0)
    manager = PixelSizeParser(MagicMock())
    manager._plate = mock_plate
    with pytest.raises(Exception) as exc_info:
        manager._check_wells_and_images()
    assert "Unable to retrieve image from well" in str(exc_info.value)


def test_get_pixel_values():
    # Setup a mock image object with getPrimaryPixels returning None
    mock_image = MagicMock()
    mock_image.getPixelSizeX.return_value = 1.19
    mock_image.getPixelSizeY.return_value = 1.19
    manager = PixelSizeParser(MagicMock())
    (x, y) = manager._get_pixel_values(mock_image)
    assert (x, y) == (1.2, 1.2)


def test_get_pixel_values_raises_value_error_on_none():
    # Setup a mock image object with getPrimaryPixels returning None
    mock_image = MagicMock()
    mock_image.getPixelSizeX.return_value = None
    mock_image.getPixelSizeY.return_value = None
    manager = PixelSizeParser(MagicMock())
    # Assert that ValueError is raised when pixels are None
    with pytest.raises(ValueError) as exc_info:
        manager._get_pixel_values(mock_image)
    assert "No pixel data found for the image." in str(exc_info.value)


@patch("omero_screen_napari.plate_handler.PixelSizeParser._get_pixel_values")
def test_check_pixel_values(mock_get_pixel_values, mock_omero_data):
    # Setup a mock image object with getPrimaryPixels returning None
    mock_get_pixel_values.return_value = (1, 1)
    manager = PixelSizeParser(mock_omero_data)
    manager._random_images = [MagicMock(), MagicMock()]
    manager._check_pixel_values()
    assert manager._pixel_size == (1, 1)


@patch("omero_screen_napari.plate_handler.PixelSizeParser._get_pixel_values")
def test_check_pixel_values_zero(mock_get_pixel_values, mock_omero_data):
    # Setup a mock image object with getPrimaryPixels returning None
    mock_get_pixel_values.side_effect = [(0, 1), (1, 1)]
    manager = PixelSizeParser(mock_omero_data)
    manager._random_images = [MagicMock(), MagicMock()]
    with pytest.raises(ValueError) as exc_info:
        manager._check_pixel_values()
    assert "One of the pixel sizes is 0" in str(exc_info.value)


@patch("omero_screen_napari.plate_handler.PixelSizeParser._get_pixel_values")
def test_check_pixel_values_unequal(mock_get_pixel_values, mock_omero_data):
    # Setup a mock image object with getPrimaryPixels returning None
    mock_get_pixel_values.side_effect = [(2, 1), (1, 1)]
    manager = PixelSizeParser(mock_omero_data)
    manager._random_images = [MagicMock(), MagicMock()]
    with pytest.raises(ValueError) as exc_info:
        manager._check_pixel_values()
    assert "Pixel sizes are not identical between wells" in str(exc_info.value)


# __________________________TESTING Well Parser________________________________


def test_parse_well_object_found(mock_plate_well):
    test_well_pos = "A2"
    mock_plate = mock_plate_well(["A1", "A2", "B1"])
    manager = WellDataParser(
        MagicMock(), test_well_pos
    )
    manager._plate = mock_plate
    manager._parse_well_object()
    assert manager._well_id == "id_A2"


def test_parse_well_object_notfound(mock_plate_well):
    test_well_pos = "A5"
    mock_plate = mock_plate_well(["A1", "A2", "B1"])
    manager = WellDataParser(
        MagicMock(), test_well_pos
    )
    manager._plate = mock_plate
    with pytest.raises(ValueError) as exc_info:
        manager._parse_well_object()
    assert "Well with position A5 does not exist." in str(exc_info.value)


def test_get_well_metadata_with_valid_annotations(
    monkeypatch, well_with_annotations
):
    manager = WellDataParser(MagicMock(), "A1")
    monkeypatch.setattr(manager, "_well", well_with_annotations)
    monkeypatch.setattr(manager, "_well_pos", "A1")

    manager._get_well_metadata()
    assert manager._metadata == {"key": "value"}


def test_get_well_metadata_with_no_valid_annotations(
    monkeypatch, well_without_annotations
):
    manager = WellDataParser(MagicMock(), "A1")
    monkeypatch.setattr(manager, "_well", well_without_annotations)
    monkeypatch.setattr(manager, "_well_pos", "A1")

    with pytest.raises(ValueError) as e:
        manager._get_well_metadata()
    assert "No map annotation found for well A1" in str(e.value)


def test_load_well_csvdata(omero_data_lazyframe_mock):
    # Initialize your class with the mocked omero_data and a specific well position
    well_pos = "A1"
    manager = WellDataParser(
        omero_data_lazyframe_mock, well_pos
    )

    # Execute the method under test
    manager._load_well_csvdata()

    # Verify the results
    expected_df = (
        pl.DataFrame({"well": ["A1"], "values": [10]}).lazy().collect()
    )
    assert manager._well_ifdata.equals(
        expected_df
    ), "The dataframes do not match"




