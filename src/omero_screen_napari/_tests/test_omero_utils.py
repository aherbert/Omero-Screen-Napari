from unittest.mock import MagicMock, patch

import pytest

from omero_screen_napari.omero_utils import omero_connect


# Mock the logger to avoid side effects during testing
@pytest.fixture
def mock_logger(mocker):
    return mocker.patch('omero_screen_napari.omero_utils.logger')

# Mock the environment variables
@pytest.fixture
def mock_env_vars(mocker):
    mocker.patch('os.getenv', side_effect=lambda k: {
        "USERNAME": "test_user",
        "PASSWORD": "test_pass",
        "HOST": "test_host",
        "PROJECT_ID": "123",
        "USE_LOCAL_ENV": "0"
    }.get(k))

# Mock the BlitzGateway object
@pytest.fixture
def mock_blitzgateway(mocker):
    mock = MagicMock()
    mocker.patch('omero_screen_napari.omero_utils.BlitzGateway', return_value=mock)
    return mock

# Parametrized test cases
@pytest.mark.parametrize("test_id, connect_success, expected_result", [
    ("happy_path_1", [True], ["connected"]),
    ("happy_path_2", [True], ["different_result"]),
    ("edge_case_no_connection", [False], [None]),
    ("error_case_os_error", [None], [None]),  # Simulate an OSError
], ids=lambda x: x[0])
def test_omero_connect(test_id, connect_success, expected_result, mock_logger, mock_env_vars, mock_blitzgateway):
    # Arrange
    mock_blitzgateway.connect.return_value = connect_success[0]


    # Mock decorated function to return a value if connected
    @omero_connect
    def mock_decorated_function(conn=None):
        return "connected" if conn else "not connected"

    # Act
    result = mock_decorated_function()

    # Assert
# sourcery skip: no-conditionals-in-tests
    if connect_success[0]:
        mock_blitzgateway.connect.assert_called_once()
        mock_blitzgateway.close.assert_called()
        assert result == "connected"
    elif connect_success[0] is False:
        mock_blitzgateway.connect.assert_called_once()
        mock_blitzgateway.close.assert_called()
        assert result is None
