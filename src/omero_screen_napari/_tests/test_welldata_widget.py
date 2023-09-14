from unittest.mock import Mock
from omero_screen_napari._welldata_widget import (
    _get_plate_data,
    _get_channel_data,
)


def test_get_channel_data(omero_conn):
    plate = omero_conn.getObject("Plate", 3)
    channel_data = _get_channel_data(plate)
    assert channel_data == {"DAPI": "0", "Tub": "1", "p21": "2", "EdU": "3"}


# Mock class for MapAnnotationWrapper
class MapAnnotationWrapper:
    def getValue(self):
        pass


def test_get_channel_data_hoechst(monkeypatch):
    # Create a mock for MapAnnotationWrapper
    map_annotation_mock = Mock(spec=MapAnnotationWrapper)
    map_annotation_mock.getValue.return_value = [("Hoechst", "value")]

    # Create a mock for _PlateWrapper with listAnnotations method
    plate_mock = Mock()
    plate_mock.listAnnotations.return_value = [map_annotation_mock]

    # Run the function and get the result
    channel_data = _get_channel_data(plate_mock)

    # Validate the result
    assert "Hoechst" not in channel_data
    assert "DAPI" in channel_data
    assert channel_data["DAPI"] == "value"
