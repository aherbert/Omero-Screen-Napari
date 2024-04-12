from unittest.mock import MagicMock, patch
import numpy as np

import pytest

from omero_screen_napari.utils import omero_connect, scale_image


# Mock the logger to avoid side effects during testing

@omero_connect
def dummy_function(*args, **kwargs):
    # This function might interact with the OMERO server using the connection
    return "Function Executed"

def set_env_vars_mock():
    return "/path/to/.env"

def test_omero_connect_success():
    with patch('omero_screen_napari.utils.BlitzGateway') as mock_gateway, \
         patch('omero_screen_napari.utils.load_dotenv') as mock_load_dotenv, \
         patch('os.getenv') as mock_getenv, \
         patch('omero_screen_napari.utils.logger') as mock_logger:
        
        # Set up mock behavior
        mock_gateway_instance = MagicMock()
        mock_gateway_instance.connect.return_value = True
        mock_gateway.return_value = mock_gateway_instance

        mock_load_dotenv.return_value = None
        mock_getenv.side_effect = lambda x: {"USERNAME": "user", "PASSWORD": "pass", "HOST": "ome2.hpc.sussex.ac.uk"}.get(x)
        # Call the decorated function
        result = dummy_function()

        # Assert the function executed and logged connection messages
        assert result == "Function Executed"
        mock_logger.info.assert_any_call('Connecting to Omero at host: ome2.hpc.sussex.ac.uk')
        mock_logger.info.assert_any_call("Disconnecting from Omero")

def test_omero_connect_failure():
    with patch('omero_screen_napari.utils.BlitzGateway') as mock_gateway, \
         patch('omero_screen_napari.utils.load_dotenv'), \
         patch('os.getenv') as mock_getenv, \
         patch('omero_screen_napari.utils.logger') as mock_logger:
        
        # Set up mock behavior for connection failure
        mock_gateway_instance = MagicMock()
        mock_gateway_instance.connect.return_value = False
        mock_gateway_instance.getLastError.return_value = "Connection Error"
        mock_gateway.return_value = mock_gateway_instance
        
        mock_getenv.side_effect = ["user", "pass", "host"]

        # Expect the function to raise an Exception due to connection failure
        with pytest.raises(Exception) as exc_info:
            dummy_function()

        assert "Failed to connect to Omero: Connection Error" in str(exc_info.value)
        mock_logger.error.assert_called_once_with("Failed to connect to Omero: Connection Error")


@pytest.mark.parametrize("input_image, intensities, expected_shape", [
    (np.zeros((100, 100, 3), dtype=np.uint8), {0: (0, 255), 1: (0, 255), 2: (0, 255)}, (100, 100, 3)),
    # Add more test cases as needed
])
def test_scale_image(input_image, intensities, expected_shape):
    # Call the scale_image function
    output_image = scale_image(input_image, intensities)
    
    # Check that the output has the correct shape
    assert output_image.shape == expected_shape, f"Expected shape {expected_shape}, got {output_image.shape}"
    
    # Check that the output type is correct (optional, based on your requirements)
    assert output_image.dtype == np.uint8, "Output image dtype is not np.uint8"