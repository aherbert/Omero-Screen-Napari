import pytest

from omero_screen_napari.omero_api import _get_pixel_size


# def test_retrieve_data_handles_errors(mock_omero_conn, mock_get_image, mock_viewer_data, mocker, caplog):
#     mock_omero_conn.connect.side_effect = Exception("Test Error")

#     retrieve_data("plate_id", "well_pos", "images")

#     # Check if the error log message was recorded
#     assert "Test Error" in caplog.text
#     # # Or more specifically, you can check for log messages at the ERROR level
#     # assert any("Test Error" in message for message in caplog.messages)


## 1) Testing the plate retrieval functions:

## A) Testing the pixel size retrieval function:

def test_get_pixel_size(mock_plate):
    result = _get_pixel_size(mock_plate)

    assert result is not None, "The function returned None instead of a tuple."
    assert isinstance(result, tuple), "The function should return a tuple."
    pixel_x, pixel_y = result

    # Assertions to verify correct behavior
    assert pixel_x == 0.123, "Pixel size X is incorrect."
    assert pixel_y == 0.456, "Pixel size Y is incorrect."
    mock_plate.listChildren.assert_called_once()
    mock_plate.listChildren()[0].getImage.assert_called_once_with(0)
    mock_plate.listChildren()[
        0
    ].getImage().getPrimaryPixels.assert_called_once()


def test_get_pixel_size_no_wells_found(mock_plate):
    mock_plate.listChildren.return_value = []

    with pytest.raises(ValueError) as excinfo:
        _get_pixel_size(mock_plate)
    assert "No wells found in the plate" in str(excinfo.value)

    # Verify that listChildren was called once
    mock_plate.listChildren.assert_called_once()


def test_get_pixel_size_no_image_found(mock_plate):
    mock_well = mock_plate.listChildren()[0]
    mock_well.getImage.return_value = None

    with pytest.raises(ValueError) as excinfo:
        _get_pixel_size(mock_plate)
    assert "No images found in the first well" in str(excinfo.value)

    # Verify that getImage was called correctly
    mock_well.getImage.assert_called_once_with(0)


def test_get_pixel_size_no_pixel_found(mock_plate):
    mock_well = mock_plate.listChildren()[0]
    mock_image = mock_well.getImage.return_value
    mock_image.getPrimaryPixels.return_value = None

    with pytest.raises(ValueError) as excinfo:
        _get_pixel_size(mock_plate)
    assert "No pixel data found for the image" in str(excinfo.value)

    # Verify that getImage was called correctly
    mock_well.getImage.assert_called_once_with(0)
