import pytest
from unittest.mock import MagicMock, create_autospec
from omero_screen_napari.plate_handler import (
    CsvFileManager,
    ChannelDataManager,
)
from omero_screen_napari.plate_handler import OmeroData
from omero.gateway import FileAnnotationWrapper, MapAnnotationWrapper, ProjectWrapper, DatasetWrapper


class MockPlate:
    def __init__(self, file_names, map_annotations=None):
        """
        Initializes the mock plate with a list of file names to simulate file annotations
        and an optional list of tuples to simulate map annotations.
        :param file_names: List of strings representing file names.
        :param map_annotations: Optional list of tuples representing map annotations.
        """
        self.file_names = file_names
        self.map_annotations = map_annotations or []

    def listAnnotations(self):
        """
        Simulates the listAnnotations method of a Plate, yielding mock objects.
        """
        for name in self.file_names:
            file_ann_wrapper = create_autospec(
                FileAnnotationWrapper, instance=True
            )
            mock_file = MagicMock()
            mock_file.getName.return_value = name
            file_ann_wrapper.getFile.return_value = mock_file
            yield file_ann_wrapper

        for map_ann in self.map_annotations:
            map_ann_wrapper = create_autospec(
                MapAnnotationWrapper, instance=True
            )
            map_ann_wrapper.getValue.return_value = map_ann
            yield map_ann_wrapper


@pytest.fixture
def mock_plate():
    def _mock_plate(file_names, map_annotations=None):
        return MockPlate(file_names, map_annotations)

    return _mock_plate


@pytest.fixture
def mock_omero_data():
    mock_data = MagicMock(spec=OmeroData)
    mock_data.plate_id = 123
    mock_data.csv_path = "/path/to/csv"
    return mock_data

@pytest.fixture
def mock_conn():
    def _create_mock_conn(project_id, datasets):
        mock_project = create_autospec(ProjectWrapper, instance=True)
        mock_project.getId.return_value = project_id

        # Adjusted mock datasets creation to explicitly set getId return value
        mock_datasets = {}
        for name, ds_id in datasets.items():
            mock_ds = create_autospec(DatasetWrapper, instance=True)
            mock_ds.getId.return_value = ds_id  # This ensures getId() returns the integer ID
            mock_datasets[name] = mock_ds

        def get_object(_type, *args, **kwargs):
            if _type == "Project":
                return mock_project
            elif _type == "Dataset" and "attributes" in kwargs:
                dataset_name = kwargs["attributes"].get("name")
                if dataset_name in mock_datasets and "opts" in kwargs and kwargs["opts"].get("project") == project_id:
                    return mock_datasets[dataset_name]
            return None

        mock_conn = MagicMock()
        mock_conn.getObject.side_effect = get_object

        return mock_conn

    return _create_mock_conn



@pytest.fixture
def csv_manager(mock_omero_data, mock_plate):
    # This fixture now requires an additional parameter to specify file names
    return CsvFileManager(omero_data=mock_omero_data, plate=mock_plate)

@pytest.fixture
def channel_manager(mock_omero_data, mock_plate):
    # This fixture now requires an additional parameter to specify file names
    return ChannelDataManager(omero_data=mock_omero_data, plate=mock_plate)

@pytest.fixture
def csv_manager_with_mocked_file(mock_omero_data, tmp_path):
    # Create a mock for the original_file with getFileInChunks
    mock_original_file = MagicMock()
    mock_original_file.getFileInChunks.return_value = [
        b"chunk1",
        b"chunk2",
        b"chunk3",
    ]
    file_names = ["not_relevant_file.csv", "final_data.csv"]
    mock_plate = MockPlate(file_names)
    # Instantiate your class
    handler = CsvFileManager(mock_omero_data, mock_plate)

    # Mock the original_file and csv_path attributes
    handler.original_file = mock_original_file
    handler.csv_path = tmp_path
    handler.file_name = "example_final_data.csv"

    return handler


# from unittest.mock import MagicMock

# import pytest
# from omero.gateway import BlitzGateway


# @pytest.fixture
# def mock_omero_conn(mocker):
#     """Mock the BlitzGateway connection."""
#     return mocker.MagicMock(spec=BlitzGateway)

# # Mocking conn.getPlate and its connected objects:

# @pytest.fixture
# def mock_plate():
#     # Create mock objects
#     mock_plate = MagicMock()


#     mock_well = MagicMock()
#     mock_image = MagicMock()
#     mock_pixels = MagicMock()
#     mock_plate.listChildren.return_value = [mock_well]

#     # Setup mock methods and return values
#     mock_pixels.getPhysicalSizeX.return_value = 0.123
#     mock_pixels.getPhysicalSizeY.return_value = 0.456
#     mock_image.getPrimaryPixels.return_value = mock_pixels
#     mock_well.getImage.return_value = mock_image

#     return mock_plate
