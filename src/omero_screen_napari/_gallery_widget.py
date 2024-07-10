import logging
from pathlib import Path
import datetime

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container


from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.gallery_api import show_gallery, run_gallery_parser
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.gallery_userdata_singleton import userdata


logger = logging.getLogger("omero-screen-napari")


def gallery_gui_widget():
    # Call the magic factories to get the widget instances
    gallery_widget_instance = gallery_widget()
    reset_widget_instance = reset_widget()
    analysis_widget_instance = run_analysis_widget()
    return Container(widgets=[gallery_widget_instance, reset_widget_instance, analysis_widget_instance])


@magic_factory(
    call_button="Enter",
)
def reset_widget():
    omero_data.cropped_images = []
    omero_data.cropped_labels = []
@magic_factory(
    call_button="Enter",
)
def run_analysis_widget(wells: str, galleries: int):
    well_list = wells.split(", ")
    run_gallery_parser(omero_data, userdata, well_list, galleries)


@magic_factory(
    call_button="Enter",
    segmentation={"choices": ["nucleus", "cell"]},
    crop_size={"choices": [20, 30, 50, 100]},
    cellcycle={"choices": ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]},
)
def gallery_widget(
    #viewer: "napari.viewer.Viewer",
    well: str,
    segmentation: str,
    crop_size: int,
    cellcycle: str,
    columns: int = 4,
    rows: int = 4,
    reload: bool = True,
    contour: bool = True,
    blue_channel: str = "DAPI",
    green_channel: str = "Tub",
    red_channel: str = "EdU",
):
    channels = [blue_channel, green_channel, red_channel]  # to match rgb order
    channels = [channel for channel in channels if channel != ""]
    user_data_dict = {
        "well": well,
        "segmentation": segmentation,
        "reload": reload,
        "crop_size": crop_size,
        "cellcycle": cellcycle,
        "columns": columns,
        "rows": rows,
        "contour": contour,
        "channels": channels,
    }
    UserData.set_omero_data_channel_keys(omero_data.channel_data.keys())
    UserData.reset_with_input(userdata, **user_data_dict)
    show_gallery(omero_data, userdata)