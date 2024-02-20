"""
This module handles the widget to call Omero and load flatfield corrected well images
as well as segmentation masks (if avaliable) into napari.
The plugin can be run from napari as Welldata Widget under Plugins.
"""


# Logging
import atexit
import logging
import os
import re
import tempfile
from typing import List, Optional, Tuple

import napari
import numpy as np
import omero
import pandas as pd
from ezomero import get_image
from magicgui import magic_factory
from napari.viewer import Viewer
from omero.gateway import BlitzGateway, FileAnnotationWrapper
from qtpy.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget
from skimage import exposure
from tqdm import tqdm

from omero_screen_napari.omero_utils import omero_connect
from omero_screen_napari.viewer_data_module import viewer_data
from omero_screen_napari.omero_api import retrieve_data

# Looging

logger = logging.getLogger("omero-screen-napari")


# Global variable to keep track of the existing metadata widget
metadata_widget: Optional[QWidget] = None

# Widget to call Omero and load well images


@magic_factory(call_button="Enter")
def welldata_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos: str = "Well Position",
    images: str = "All",
    image_id: int = 0,
) -> None:
    """
    This function is a widget for handling well data in a napari viewer.
    It retrieves data based on the provided plate ID and well position,
    and then adds the images and labels to the viewer. It also handles metadata,
    sets color maps, and adds label layers to the viewer.
    """
    global metadata_widget

    clear_viewer_layers(viewer)
    retrieve_data(plate_id, well_pos, images)
    add_image_to_viewer(viewer)
    handle_metadata_widget(viewer)
    set_color_maps(viewer)
    add_label_layers(viewer)


# accessory functions for _welldata_widget


def clear_viewer_layers(viewer: Viewer) -> None:
    viewer.layers.select_all()
    viewer.layers.remove_selected()


def add_image_to_viewer(viewer: Viewer) -> None:
    viewer.add_image(
        viewer_data.images, channel_axis=-1, scale=viewer_data.pixel_size
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"


def handle_metadata_widget(viewer: Viewer) -> None:
    global metadata_widget
    if metadata_widget is not None:
        viewer.window.remove_dock_widget(metadata_widget)
    metadata_widget = MetadataWidget(viewer_data.metadata)
    viewer.window.add_dock_widget(metadata_widget)


def set_color_maps(viewer: Viewer) -> None:
    channel_names: dict = {
        key: int(value) for key, value in viewer_data.channel_data.items()
    }
    color_maps: dict[str, str] = _generate_color_map(channel_names)
    for name, index in channel_names.items():
        layer = viewer.layers[index]
        layer.name = name
        layer.colormap = color_maps[name]


def add_label_layers(viewer: Viewer) -> None:
    scale = viewer_data.pixel_size
    if len(viewer_data.labels.shape) == 3:
        viewer.add_labels(
            viewer_data.labels.astype(int), name="Nuclei Masks", scale=scale
        )
    elif viewer_data.labels.shape[3] == 2:
        channel_1_masks = viewer_data.labels[..., 0].astype(int)
        channel_2_masks = viewer_data.labels[..., 1].astype(int)
        viewer.add_labels(channel_1_masks, name="Nuclei Masks", scale=scale)
        viewer.add_labels(channel_2_masks, name="Cell Masks", scale=scale)
    else:
        raise ValueError("Invalid segmentation label shape")


def _generate_color_map(channel_names: dict) -> dict[str, str]:
    """
    Generate a color map dictionary for the channels
    :param channel_names: dictionary of channel names and indices
    :return: a dictionary with channel names as keys and color names as values
    """
    # Initialize the color map dictionary
    color_map_dict = {}

    # Determine the number of channels
    num_channels = len(channel_names)

    # If there is only one channel, set it to gray
    if num_channels == 1:
        single_key = list(channel_names.keys())[0]
        color_map_dict[single_key] = "gray"
        return color_map_dict

    # Default color assignments for known channels
    known_channel_colors = {"DAPI": "blue", "Tub": "green", "EdU": "gray"}

    # Remaining colors for other channels
    remaining_colors = ["red"]

    # Assign known colors to known channels if they exist
    for known_channel, known_color in known_channel_colors.items():
        if known_channel in channel_names:
            color_map_dict[known_channel] = known_color

    # Assign remaining colors to any remaining channels
    remaining_channels = set(channel_names.keys()) - set(color_map_dict.keys())
    for remaining_channel, remaining_color in zip(
        remaining_channels, remaining_colors
    ):
        color_map_dict[remaining_channel] = remaining_color
    return color_map_dict


# Add small widget for metadata to get info pn cell line and condition
class MetadataWidget(QWidget):
    """
    A custom QWidget that displays metadata in a QLabel. The metadata is displayed as key-value pairs.

    Inherits from:
    QWidget: Base class for all user interface objects in PyQt5.

    Attributes:
    layout (QVBoxLayout): Layout for the widget.
    label (QLabel): Label where the metadata is displayed.

    Args:
    metadata (dict): The metadata to be displayed. It should be a dictionary where the keys are the metadata
    fields and the values are the metadata values.
    """

    def __init__(self, metadata):
        super().__init__()
        self.layout = QVBoxLayout()

        self.label = QLabel()
        self.label.setText(
            "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        )

        self.layout.addWidget(self.label)
        self.setLayout(self.layout)


