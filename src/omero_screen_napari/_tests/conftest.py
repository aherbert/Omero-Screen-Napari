from unittest.mock import MagicMock, create_autospec
import polars as pl
import pytest
from omero.gateway import (
    DatasetWrapper,
    FileAnnotationWrapper,
    MapAnnotationWrapper,
    ProjectWrapper,
    PlateWrapper
)

from omero_screen_napari.plate_handler import (
    ChannelDataParser,
    CsvFileParser,
    OmeroData,
)
@pytest.fixture
def mock_plate_for_input_test():
    # Create a mock for the Well object with listChildren method
    mock_well = MagicMock()
    mock_well.listChildren.return_value = iter(range(5))

    # Create a mock for the Plate object with listChildren method returning a generator with one Well
    mock_plate = MagicMock()
    mock_plate.listChildren.return_value = iter([mock_well])

    return mock_plate

class MockPlate:
    def __init__(
        self, file_names, map_annotations=None, wells_count=0, img_number=5
    ):
        """
        Initializes the mock plate with a list of file names to simulate file annotations
        and an optional list of tuples to simulate map annotations.
        :param file_names: List of strings representing file names.
        :param map_annotations: Optional list of tuples representing map annotations.
        """
        self.file_names = file_names
        self.map_annotations = map_annotations or []
        self.wells_count = wells_count
        self.image_count = img_number

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

    def listChildren(self):
        """
        Simulates the listChildren method of a Plate, yielding mock well objects.
        """
        for _ in range(self.wells_count):
            yield MockWell(self.image_count)


@pytest.fixture
def mock_plate():
    def _mock_plate(
        file_names, map_annotations=None, wells_count=0, img_number=5
    ):
        return MockPlate(file_names, map_annotations, wells_count, img_number)

    return _mock_plate


class MockWell:
    def __init__(self, img_number):
        """
        Initializes a mock well object.
        """
        self.img_number = img_number
        self.images = [
            MagicMock(name=f"MockImage{i}") for i in range(img_number)
        ]  # Assuming 5 mock images per well for example

    def getImage(self, index):
        """
        Returns a mock object simulating an image based on the index.
        """
        try:
            return self.images[index]
        except Exception as e:
            raise e


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


@pytest.fixture
def mock_image():
    # This fixture creates a single MockImage instance
    name = "default_name"
    return MockImage(name)


@pytest.fixture
def mock_omero_data():
    mock_data = MagicMock(spec=OmeroData)
    mock_data.plate = MagicMock()
    mock_data.plate_id = 123
    mock_data.data_path = "/path/to/data"
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

    def _create_mock_conn(mock_project_id, datasets, none_plate_id=None):
        # Create a mock project with autospec for more realistic behavior
        mock_project = create_autospec(ProjectWrapper, instance=True)
        mock_project.getId.return_value = mock_project_id
        mock_plate = create_autospec(PlateWrapper, instance=True)
        mock_plate.getId.return_value = mock_project_id
        # Create mock datasets, keyed by name, with autospec for realistic behavior
        mock_datasets = {}
        for name, ds_id in datasets.items():
            mock_ds = create_autospec(DatasetWrapper, instance=True)
            mock_ds.getId.return_value = (
                ds_id  # Ensures getId() returns the specified integer ID
            )
            mock_datasets[name] = mock_ds

        # Function to simulate getObject behavior
        def get_object(_type, attributes=None, opts=None):
            attributes = attributes or {}
            opts = opts or {}

            if _type == "Project":
                return mock_project
            elif _type == "Plate":
                plate_id = attributes
                # Check if the requested plate_id matches the one that should return None
                return None if plate_id == none_plate_id else mock_plate
            elif _type == "Dataset":
                dataset_name = attributes.get("name")
                project_id = opts.get("project")
                if (
                    dataset_name
                    and project_id == mock_project.getId.return_value
                ):
                    return mock_datasets.get(dataset_name)
            return None

        # Create a MagicMock for the connection, with getObject using our custom logic
        mock_conn = MagicMock()
        mock_conn.getObject.side_effect = get_object

        return mock_conn

    return _create_mock_conn


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
    return CsvFileParser(omero_data=mock_omero_data, plate=mock_plate)


@pytest.fixture
def channel_manager(mock_omero_data, mock_plate):
    # This fixture now requires an additional parameter to specify file names
    return ChannelDataParser(omero_data=mock_omero_data, plate=mock_plate)


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
    handler = CsvFileParser(mock_omero_data)

    # Mock the original_file and csv_path attributes
    handler._original_file = mock_original_file
    handler._data_path = tmp_path
    handler._file_name = "example_final_data.csv"

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


@pytest.fixture
def mock_plate_well():
    """
    Creates a simple mock plate with wells that have mock implementations of getWellPos and getId.
    """
    def _mock_plate_well(well_positions):
        # Create a list of mock wells, each configured with specific getWellPos and getId behaviors.
        mock_wells = [
            MagicMock(
                getWellPos=MagicMock(return_value=pos), 
                getId=MagicMock(return_value=f"id_{pos}")
            ) for pos in well_positions
        ]
        
        # Create a mock plate that returns the mock wells from listChildren.
        mock_plate = MagicMock()
        mock_plate.listChildren.return_value = mock_wells
        return mock_plate

    return _mock_plate_well


@pytest.fixture
def mock_annotation_with_value():
    annotation = MagicMock()
    annotation.getValue.return_value = {'key': 'value'}
    return annotation

@pytest.fixture
def mock_annotation_without_value():
    annotation = MagicMock()
    annotation.getValue.return_value = None
    return annotation

@pytest.fixture
def well_with_annotations(mock_annotation_with_value):
    well = MagicMock()
    well.listAnnotations.return_value = [mock_annotation_with_value]
    return well

@pytest.fixture
def well_without_annotations(mock_annotation_without_value):
    well = MagicMock()
    well.listAnnotations.return_value = [mock_annotation_without_value]
    return well

@pytest.fixture
def omero_data_lazyframe_mock():
    # Create a LazyFrame mock
    lazy_frame = pl.DataFrame({"well": ["A1", "A2", "B1"], "values": [10, 20, 30]}).lazy()
    # Create a mock for the omero_data object
    omero_data = MagicMock()
    omero_data.plate_data = lazy_frame
    
    return omero_data

