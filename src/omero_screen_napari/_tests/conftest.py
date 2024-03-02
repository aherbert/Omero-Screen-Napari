from unittest.mock import MagicMock, create_autospec

import pytest
from omero.gateway import (
    DatasetWrapper,
    FileAnnotationWrapper,
    MapAnnotationWrapper,
    ProjectWrapper,
)

from omero_screen_napari.plate_handler import (
    ChannelDataManager,
    CsvFileManager,
    OmeroData,
)


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
    """
    Create a mock connection object.

    Args:
        mock_project_id: The ID of the mock project.
        datasets: A dictionary mapping dataset names to their project_ids.

    Returns:
        MagicMock: A mock connection object.
    """
    def _create_mock_conn(mock_project_id, datasets):
        # Create a mock project with autospec for more realistic behavior
        mock_project = create_autospec(ProjectWrapper, instance=True)
        mock_project.getId.return_value = mock_project_id

        # Create mock datasets, keyed by name, with autospec for realistic behavior
        mock_datasets = {}
        for name, ds_id in datasets.items():
            mock_ds = create_autospec(DatasetWrapper, instance=True)
            mock_ds.getId.return_value = ds_id  # Ensures getId() returns the specified integer ID
            mock_datasets[name] = mock_ds

        # Function to simulate getObject behavior
        def get_object(_type, attributes=None, opts=None):
            attributes = attributes or {}
            opts = opts or {}

            if _type == "Project":
                return mock_project
            elif _type == "Dataset":
                dataset_name = attributes.get("name")
                project_id = opts.get("project")
                if dataset_name and project_id == mock_project.getId.return_value:
                    return mock_datasets.get(dataset_name)
            return None

        # Create a MagicMock for the connection, with getObject using our custom logic
        mock_conn = MagicMock()
        mock_conn.getObject.side_effect = get_object

        return mock_conn

    return _create_mock_conn

class MockImage:
    """
    Mock to supply images for testing the listChidlren method of the Omero DatasetWrapper.
    """
    def __init__(self, name):
        self._name = name

    def getName(self):
        return self._name

    def getId(self):
        return "mock_id"  # Return a mock ID or vary this as needed for your tests
@pytest.fixture
def mock_image():
    # This fixture creates a single MockImage instance
    name = "default_name"
    return MockImage(name)

@pytest.fixture
def mock_screen_dataset_factory():
    """Fixture factory to create a mock screen dataset with a dynamic list of children."""
    def _factory(mock_images):
        class MockScreenDataset:
            def listChildren(self):
                return iter(mock_images)
        return MockScreenDataset()
    return _factory

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


@pytest.fixture
def mock_flatfield_obj():
    # Mock the flatfield object with listAnnotations method
    mock = MagicMock()
    mock.listAnnotations = MagicMock()
    return mock

@pytest.fixture
def mock_flatfield_map_annotation():
    def _mock_map_annotation(values):
        mock = MagicMock(spec=MapAnnotationWrapper)
        mock.getValue.return_value = values
        return mock
    return _mock_map_annotation