from omero_screen_napari.omero_data import OmeroData
import os
from unittest.mock import patch


def test_omero_data_initialization():
    os.environ["USE_LOCAL_ENV"] = "0"
    expected_project_id = 5313
    data = OmeroData()
    assert data.project_id == expected_project_id


def test_omero_data_initialization_env1():
    os.environ["USE_LOCAL_ENV"] = "1"
    expected_project_id = 151
    data = OmeroData()
    assert data.project_id == expected_project_id

def test_reset_method():
    os.environ["USE_LOCAL_ENV"] = "0"
    data = OmeroData()
    data.project_id = 100 
    assert data.project_id == 100
    data.reset()
    expected_project_id = 5313
    assert data.project_id == expected_project_id 
