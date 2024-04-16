
from omero_screen_napari.omero_data_singleton import omero_data

omero_data.channel_data = {
    "DAPI": "1",
    "Tub": "2",
}

from omero_screen_napari.gallery_userdata_singleton import userdata
def test_set_userdata():
    print(userdata)
