import pytest
from omero.gateway import BlitzGateway

PLATE_ID = 3


@pytest.fixture(scope="function")
def omero_conn():
    # Create a connection to the omero server
    conn = BlitzGateway("root", "omero", host="localhost", port=4064)
    conn.connect()
    print("Setting up connection to OMERO server")
    yield conn
    # After the tests are done, disconnect
    conn.close()
    print("Closed connection to OMERO server")
