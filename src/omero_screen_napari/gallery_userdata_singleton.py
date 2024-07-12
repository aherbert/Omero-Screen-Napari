from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data_singleton import omero_data

if channel_keys := list(omero_data.channel_data.keys()):
    UserData.set_omero_data_channel_keys(omero_data.channel_data.keys())
    userdata = UserData.set_defaults(channel_keys)
else:
    userdata = None
