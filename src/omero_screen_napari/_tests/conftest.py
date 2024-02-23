from unittest.mock import MagicMock

import pytest
from omero.gateway import BlitzGateway


@pytest.fixture
def mock_omero_conn(mocker):
    """Mock the BlitzGateway connection."""
    return mocker.MagicMock(spec=BlitzGateway)

# Mocking conn.getPlate and its connected objects:

@pytest.fixture
def mock_plate():
    # Create mock objects
    mock_well = MagicMock()
    mock_image = MagicMock()
    mock_pixels = MagicMock()

    # Setup mock methods and return values
    mock_pixels.getPhysicalSizeX.return_value = 0.123
    mock_pixels.getPhysicalSizeY.return_value = 0.456
    mock_image.getPrimaryPixels.return_value = mock_pixels
    mock_well.getImage.return_value = mock_image
    mock_plate = MagicMock()
    mock_plate.listChildren.return_value = [mock_well]

    return mock_plate
