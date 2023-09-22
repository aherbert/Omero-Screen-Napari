"""
This module handles the widget to call Omero and load well images
"""
import re
import tempfile
from typing import Optional

import napari
import numpy as np
import omero
import pandas as pd
from ezomero import get_image
from magicgui import magic_factory
from omero.gateway import BlitzGateway
from omero.gateway import FileAnnotationWrapper
from qtpy.QtWidgets import QMessageBox, QLabel, QVBoxLayout, QWidget
from skimage import exposure
from tqdm import tqdm

from omero_screen_napari.omero_utils import omero_connect
from omero_screen_napari.viewer_data_module import viewer_data

# Global variable to keep track of the existing metadata widget
metadata_widget: Optional[QWidget] = None


@magic_factory(call_button="Enter")
def welldata_widget(
        viewer: "napari.viewer.Viewer",
        plate_id: str = "Plate ID",
        well_pos: str = "Well Position",
        images: str = "All",
):
    global metadata_widget
    # Clear all layers from the viewer
    viewer.layers.select_all()
    viewer.layers.remove_selected()

    # Get the data from Omero
    _get_data(plate_id, well_pos, images)
    channel_names = {
        key: int(value) for key, value in viewer_data.channel_data.items()
    }

    viewer.add_image(viewer_data.images, channel_axis=-1)
    # Remove existing metadata widget if it exists
    if metadata_widget is not None:
        viewer.window.remove_dock_widget(metadata_widget)

    metadata_widget = MetadataWidget(viewer_data.metadata)
    viewer.window.add_dock_widget(metadata_widget)

    # Dictionary of color maps
    color_maps = _generate_color_map(channel_names)

    # Setting the names and color maps of the layers
    for name, index in channel_names.items():
        layer = viewer.layers[index]
        layer.name = name
        layer.colormap = color_maps[name]
        # # Calculate 1st and 99th percentiles for the contrast limits
        # lower_limit = np.percentile(layer.data, 0.1)
        # upper_limit = np.percentile(layer.data, 99.9)
        # layer.contrast_limits = (lower_limit, upper_limit)

    if viewer_data.labels.shape[3] == 1:
        viewer.add_labels(viewer_data.labels.astype(int), name="Nuclei Masks")
    else:
        # Split the last dimension to get two arrays of shape (3, 1020, 1020)
        channel_1_masks = viewer_data.labels[..., 0].astype(int)
        channel_2_masks = viewer_data.labels[..., 1].astype(int)

        # Add these to the viewer as Labels layers
        viewer.add_labels(channel_1_masks, name="Nuclei Masks")
        viewer.add_labels(channel_2_masks, name="Cell Masks")


class MetadataWidget(QWidget):
    def __init__(self, metadata):
        super().__init__()
        self.layout = QVBoxLayout()

        self.label = QLabel()
        self.label.setText(
            "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        )

        self.layout.addWidget(self.label)
        self.setLayout(self.layout)


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


## backend handling the omero connection and adding data to viwer_data


@omero_connect
def _get_data(plate_id: str, well_pos: str, images: str, conn=None):
    """
    Get data from Omero and add it to the viewer_data dataclass object
    :param plate_id: number of the omero screen plate
    :param well_pos: postion of the well in teh screen, example 'C2
    :param conn: omero.gateway connection
    :return: None
    """

    _get_omero_objects(conn, plate_id, well_pos)
    _get_well_data()
    _get_channel_data()
    _get_well_metadata()
    _get_images(images, conn)
    _get_segmentation_masks(conn)

    # except Exception as e:
    #     print(f"the error is {e}")
        # # Show a message box with the error message
        # msg_box = QMessageBox()
        # msg_box.setIcon(QMessageBox.Warning)
        # msg_box.setText(str(e))
        # msg_box.setWindowTitle("Error")
        # msg_box.setStandardButtons(QMessageBox.Ok)
        # msg_box.exec_()


# plate and channel_data
def _get_omero_objects(conn, plate_id: str, well_pos: str):
    """
    Get plate data from Omero
    """
    # get plate object
    viewer_data.plate_id = int(plate_id)
    viewer_data.plate = conn.getObject("Plate", viewer_data.plate_id)
    if viewer_data.plate is None:
        raise ValueError(f"Plate with ID {plate_id} does not exist")
    viewer_data.plate_name = viewer_data.plate.getName()
    viewer_data.well_name = well_pos

    well_found = False  # Initialize a flag to check if the well is found
    # get well object
    for well in viewer_data.plate.listChildren():
        if well.getWellPos() == well_pos:
            viewer_data.well = well
            viewer_data.well_id = well.getId()
            well_found = True  # Set the flag to True when the well is found
            break  # Exit the loop as the well is found
    if not well_found:  # Raise an error if the well was not found
        raise ValueError(f"Well with position {well_pos} does not exist.")
    owner = conn.getUser().getId()
    project = conn.getObject("Project",
                             opts={'owner': owner},
                             attributes={"name": "Screens"})
    if dataset := conn.getObject(
            "Dataset",
            attributes={"name": plate_id},
            opts={"project": project.getId()},
    ):
        viewer_data.screen_dataset = dataset
    else:
        raise ValueError(
            f"Well with position {well_pos} has not been assigned a dataset."
        )


def _get_well_data():
    file_anns = viewer_data.plate.listAnnotations()
    for ann in file_anns:
        if isinstance(
                ann, FileAnnotationWrapper
        ) and ann.getFile().getName().endswith("final_data.csv"):
            original_file = ann.getFile()
            with tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False
            ) as tmp:
                _download_file_to_tmp(original_file, tmp)
                tmp.flush()  # Ensure all data is written to the temp file
                data = pd.read_csv(tmp.name, index_col=0)
                viewer_data.plate_data = data[
                    data["well"] == viewer_data.well.getWellPos()
                    ]


def _download_file_to_tmp(original_file, tmp):
    with open(tmp.name, "wb") as f:
        for chunk in original_file.asFileObj():
            f.write(chunk)


def _get_channel_data():
    map_ann = _get_map_ann(viewer_data.plate)
    channel_data = dict(map_ann)
    cleaned_channel_data = {key.strip(): value for key, value in channel_data.items()}
    if "Hoechst" in cleaned_channel_data:
        cleaned_channel_data["DAPI"] = cleaned_channel_data.pop("Hoechst")
    viewer_data.channel_data = cleaned_channel_data


def _get_map_ann(omero_object):
    annotations = omero_object.listAnnotations()
    if not (
            map_annotations := [
                ann
                for ann in annotations
                if isinstance(ann, omero.gateway.MapAnnotationWrapper)
            ]
    ):
        raise ValueError(
            f"Map annotation not found for {viewer_data.plate.getName()}."
        )
    return map_annotations[0].getValue()


def _get_well_metadata():
    map_ann = None
    for ann in viewer_data.well.listAnnotations():
        if ann.getValue():
            map_ann = dict(ann.getValue())
    if map_ann:
        viewer_data.metadata = map_ann
    else:
        raise ValueError(
            f"No map annotation found for well {viewer_data.well.getWellPos()}"
        )


# image data


def _get_images(images, conn):
    # Lists to store the individual image arrays and ids
    flatfield_array, flatfield_channels = _get_flatfieldmask(conn)
    _check_flatfieldmask(flatfield_channels)
    print(
        f"Gathering {images} images from well {viewer_data.well.getWellPos()}"
    )
    image_arrays: list = []
    image_ids: list = []
    pattern = r'^\d+-\d+$'
    if images == "All" or re.match(pattern, images):
        if images == "All":
            img_range = range(viewer_data.well.countWellSample())
        else:
            start, end = map(int, images.split("-"))
            img_range = range(start, end + 1)
        for index in tqdm(img_range):
            image_id, image_array = _process_omero_image(index, flatfield_array, conn)
            image_arrays.append(image_array)
            image_ids.append(image_id)
    elif images.isdigit():
        index: int = int(images)
        image_id, image_array = _process_omero_image(index, flatfield_array, conn)
        image_arrays.append(image_array)
        image_ids.append(image_id)
    else:
        raise ValueError(f"Invalid image number: {images}")
    viewer_data.images = np.stack(image_arrays, axis=0)
    viewer_data.image_ids = image_ids


def _process_omero_image(index, flatfield_array, conn):
    image, image_array = get_image(
        conn, viewer_data.well.getImage(index).getId()
    )
    image_array = image_array.squeeze()
    corrected_array = image_array / flatfield_array

    # Iterate through each channel to scale it
    scaled_channels = []
    for i in range(corrected_array.shape[-1]):
        scaled_channel = scale_img(corrected_array[..., i])
        scaled_channels.append(scaled_channel)
    return image.getId(), np.stack(scaled_channels, axis=-1)


def scale_img(img: np.array, percentile: tuple = (0.1, 99.9)) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))


def _get_flatfieldmask(conn):
    """Gets flatfieldmasks from project linked to screen"""
    flatfield_mask_name = f"{viewer_data.plate_id}_flatfield_masks"
    for image in viewer_data.screen_dataset.listChildren():
        image_name = image.getName()
        if flatfield_mask_name == image_name:
            flatfield_mask_id = image.getId()
            flatfield_obj, flatfield_array = get_image(conn, flatfield_mask_id)
            flatfield_channels = dict(_get_map_ann(flatfield_obj))
            for key, value in flatfield_channels.items():
                flatfield_channels[key] = value.strip()
            return flatfield_array.squeeze(), flatfield_channels


def _check_flatfieldmask(flatfield_channels):
    """Checks if flatfieldmask is correct"""
    # check if the channels in plate and flatfield_mask are the same
    reverse_flatfield_mask = {
        v: k.split("_")[-1] for k, v in flatfield_channels.items()
    }
    # Check if the mappings are consistent
    print(reverse_flatfield_mask)
    for channel, index in viewer_data.channel_data.items():
        print(reverse_flatfield_mask[channel])
        print(index)
        try:
            assert reverse_flatfield_mask[channel] == index
        except AssertionError:
            print(f"Inconsistency found: {channel} is mapped to {index} in plate_map but {reverse_flatfield_mask[channel]} in flatfield_mask")
        else:
            print("Flatfield mask is consistent with images")

# get the image labels
def _get_segmentation_masks(conn):
    """
    Get segmentation masks as mask list
    """
    mask_list = []
    name_to_id = {
        image.getName(): image.getId()
        for image in viewer_data.screen_dataset.listChildren()
    }
    for image_name in viewer_data.image_ids:
        image_id = name_to_id.get(f"{image_name}_segmentation")
        if image_id is not None:
            mask, mask_array = get_image(conn, image_id)
            mask_array = mask_array.squeeze()
            mask_list.append(mask_array)
    viewer_data.labels = np.stack(mask_list, axis=0)
    # assuming that segmnetation is done either only on nuclei or on cell channel and nuclei channel:
    assert viewer_data.labels.shape[3] in [
        1,
        2,
    ], "Segmentation masks must have either 1 or 2 channels"


if __name__ == "__main__":
    @omero_connect
    def show_image(image_id, conn=None):
        image, image_array = get_image(conn, image_id)
        viewer = napari.Viewer()
        viewer.add_image(image_array.squeeze(), channel_axis=-1)
        print(image_array.squeeze().shape)
        napari.run()


    show_image(571808)
    # plate_id = "1421"
    # well_pos = "B2"
    # image = "0"
    # print(viewer_data.channel_data)
    #
    # _get_data(plate_id, well_pos, image)
    # print(viewer_data.channel_data)
    # channel_names = {
    #     key: int(value) for key, value in viewer_data.channel_data.items()
    # }
    # print(channel_names)
    # color_maps = _generate_color_map(channel_names)
    # print(color_maps)
    # print(viewer_data.metadata)
    # print(viewer_data.images.shape)
    # print(viewer_data.labels.shape)
    # print(viewer_data.image_ids)