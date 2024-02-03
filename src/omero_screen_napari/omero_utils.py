import functools
from omero.gateway import BlitzGateway
import os
from pathlib import Path
from omero_screen_napari.viewer_data_module import viewer_data
from dotenv import load_dotenv


def omero_connect(func):
    """
    decorator to log in and generate omero connection
    :param func: function to be decorated
    :return: wrapper function: passes conn to function and closes it after execution
    """
    # Check if the environment variable is set
    dotenv_path = ".localenv" if os.getenv("USE_LOCAL_ENV") == "1" else ".env"

    # Load the environment variables
    load_dotenv(dotenv_path=dotenv_path)

    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    project_id = os.getenv("PROJECT_ID")
    viewer_data.project_id = project_id
    print(username, password, host)

    @functools.wraps(func)
    def wrapper_omero_connect(*args, **kwargs):
        try:
            conn = BlitzGateway(username, password, host=host)
        except IOError:
            print("could not get login data")
        value = None
        try:
            print("Connecting to Omero")
            if conn.connect():
                value = func(*args, **kwargs, conn=conn)
                conn.close()
                print("Disconnecting from Omero")
            else:
                print(f"Failed to connect to Omero: {conn.getLastError()}")
        finally:
            # No side effects if called without a connection
            conn.close()
        return value

    return wrapper_omero_connect


def attach_file_to_well(well, file_path, conn):
    """
    Attach the .npz file to the specified OMERO well.
    """
    namespace = f"Training Data for Well {well.getId()}"
    description = None  # Optionally, add a description
    print(f"\nUploading data to Well ID {well.getId()}")

    # Create the file annotation and link it to the well
    file_ann = conn.createFileAnnfromLocalFile(
        file_path,
        mimetype="application/octet-stream",
        ns=namespace,
        desc=description,
    )
    well.linkAnnotation(file_ann)
