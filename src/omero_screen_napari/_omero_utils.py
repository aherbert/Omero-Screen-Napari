import functools
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from omero.gateway import BlitzGateway

from omero_screen_napari import set_env_vars

logger = logging.getLogger("omero-screen-napari")

def omero_connect(func):
    """
    decorator to log in and generate omero connection
    :param func: function to be decorated
    :return: wrapper function: passes conn to function and closes it after execution
    """
    # Load environment variables
    dotenv_path = set_env_vars()
    load_dotenv(dotenv_path=dotenv_path)

    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")

    @functools.wraps(func)
    def wrapper_omero_connect(*args, **kwargs):
        value = None
        try:
            conn = BlitzGateway(username, password, host=host)
            logger.info(f"Connecting to Omero at host: {host}")
            if conn.connect():
                value = func(*args, **kwargs, conn=conn)
                conn.close()
                logger.info("Disconnecting from Omero")
            else:
                error_msg = f"Failed to connect to Omero: {conn.getLastError()}"
                logger.error(error_msg) 
# sourcery skip: raise-specific-error
                raise Exception(error_msg)
                
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
