"""
This module handles the widget to call Omero and combine all images for a single well
together to generate a stiched flatfield corrected image.
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

from omero_screen_napari._omero_utils import omero_connect
from omero_screen_napari.viewer_data_module import viewer_data

# Looging

logger = logging.getLogger("omero-screen-napari")


# Global variable to keep track of the existing metadata widget
metadata_widget: Optional[QWidget] = None

# Widget to call Omero and load well images


@magic_factory(call_button="Enter")
def stitching_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos_list: str = "Well Position",
) -> None:
    """
    This function is a widget for handling well data in a napari viewer.
    It retrieves data based on the provided plate ID and well position,
    and then adds the images and labels to the viewer. It also handles metadata,
    sets color maps, and adds label layers to the viewer.
    """
    global metadata_widget
    well_pos = [item.strip() for item in well_pos_list.split(",")]
    logger.debug("Well positions: %s", well_pos)
    clear_viewer_layers(viewer)
    retrieve_and_stitch_data(plate_id, well_pos)
    add_image_to_viewer(viewer)
    handle_metadata_widget(viewer)
    set_color_maps(viewer)


# accessory functions for _welldata_widget


def clear_viewer_layers(viewer: Viewer) -> None:
    viewer.layers.select_all()
    viewer.layers.remove_selected()


def add_image_to_viewer(viewer: Viewer) -> None:
    viewer.add_image(
    viewer_data.stitched_images,
    contrast_limits=list(viewer_data.intensities[0]),
    gamma=1,
    channel_axis=-1,
    scale=viewer_data.pixel_size,
    name='Stitched Image'
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
    viewer.scale_bar.color = "white"


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
    for layer in viewer.layers:
        if layer.name in color_maps:
            layer.colormap = color_maps[layer.name]


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


# backend handling the omero connection and adding data to viwer_data
@omero_connect
def retrieve_and_stitch_data(plate_id: str, well_positions: List[str], conn: Optional[BlitzGateway] = None) -> None:
    stitched_images = []  # To hold stitched images for each well
    for well_pos in well_positions:
        # Process each well
        retrieve_data(plate_id, well_pos, conn)
        stitched_image = _stitch_images()  # Ensure this returns the stitched image for the current well
        stitched_images.append(stitched_image)
    
    # Combine all stitched images into a single 4D array
    viewer_data.stitched_images = np.stack(stitched_images, axis=0)


def retrieve_data(
    plate_id: str,
    well_pos: str,
    conn,
) -> None:
    """
    Get data from Omero and add it to the viewer_data dataclass object
    :param plate_id: number of the omero screen plate
    :param well_pos: postion of the well in teh screen, example 'C2
    :param conn: omero.gateway connection
    :return: None
    """
    try:
        _get_omero_objects(conn, plate_id, well_pos)
        _get_channel_data()
        _process_plate_data()
        _get_well_metadata()
        _get_images(conn)
        _stitch_images()

    except Exception as e:
        logging.exception("The following error occurred:")  # noqa: G004
        # Show a message box with the error message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(str(e))
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


# plate and channel_data
def _get_omero_objects(
    conn: BlitzGateway, plate_id: str, well_pos: str
) -> None:
    """
    Get plate data from Omero.

    Args:
        conn: The Omero connection object.
        plate_id: The ID of the plate.
        well_pos: The position of the well.

    Raises:
        ValueError: If the plate or well does not exist.

    Returns:
        None.
    """
    # get plate object
    viewer_data.plate_id = int(plate_id)
    viewer_data.plate = conn.getObject("Plate", viewer_data.plate_id)
    if viewer_data.plate is None:
        raise ValueError(f"Plate with ID {plate_id} does not exist")
    viewer_data.plate_name = viewer_data.plate.getName()
    viewer_data.well_name = well_pos
    pixel_x, pixel_y = _get_pixel_size(viewer_data.plate)
    viewer_data.pixel_size = (
        round(pixel_x.getValue(), 1),
        round(pixel_y.getValue(), 1),
    )
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
    project = conn.getObject("Project", viewer_data.project_id)
    if dataset := conn.getObject(
        "Dataset",
        attributes={"name": plate_id},
        opts={"project": project.getId()},
    ):
        viewer_data.screen_dataset = dataset
    else:
        raise ValueError(
            f"The plate {viewer_data.plate_name} has not been assigned a dataset."
        )


def _get_pixel_size(plate: omero.gateway.PlateWrapper) -> float:
    """
    Get the pixel size from the plate metadata.
    :param plate: The plate object.
    :return: The pixel size.
    """
    well = list(plate.listChildren())[0]
    image = well.getImage(0)
    pixels = image.getPrimaryPixels()
    return pixels.getPhysicalSizeX(), pixels.getPhysicalSizeY()


def _get_channel_data():
    """Process channel data to dictionary {channel_name: channel_index}
    and add it to viewer_data; raise exception if no channel data found."""
    map_ann = _get_map_ann(viewer_data.plate)
    channel_data = dict(map_ann)
    logger.debug(f"channel data is: {channel_data}")
    # convert Hoechst key to DAPI if present
    cleaned_channel_data = {
        key.strip(): value for key, value in channel_data.items()
    }
    if "Hoechst" in cleaned_channel_data:
        cleaned_channel_data["DAPI"] = cleaned_channel_data.pop("Hoechst")
    # check if one of the channels is DAPI otherwise raise exception
    if "DAPI" not in cleaned_channel_data:
        raise ValueError(
            f"Channel specifications not found for {viewer_data.plate.getName()}."
        )
    # add the channel data to viewer_data
    viewer_data.channel_data = cleaned_channel_data


def _get_map_ann(
    omero_object: omero.gateway.BlitzObjectWrapper,
) -> List[Tuple[str, str]]:
    """Get channel information from Omero map annotations
    or raise exception
    """
    annotations = omero_object.listAnnotations()
    if not (
        map_annotations := [
            ann
            for ann in annotations
            if isinstance(ann, omero.gateway.MapAnnotationWrapper)
        ]
    ):
        raise ValueError(
            f"Channel specifications not found for {viewer_data.plate.getName()}."
        )
    logger.debug(f"mapann is: {map_annotations[0].getValue()}")  # noqa: G004
    return map_annotations[0].getValue()


# experimental data retrieval and metadata organisation


def _get_annotations():
    try:
        annotations = list(
            viewer_data.plate.listAnnotations()
        )  # Convert generator to a list
        if not annotations:
            raise ValueError("No annotations found for the plate.")
        return annotations
    except ValueError as e:
        logger.warning(str(e))
        raise


def _find_relevant_files(annotations):
    relevant_files = [
        ann for ann in annotations if isinstance(ann, FileAnnotationWrapper)
    ]
    if not relevant_files:
        raise ValueError("No files attached to the current plate.")
    return relevant_files


def _select_csv_file(relevant_files):
    file_name_list = [file.getFile().getName() for file in relevant_files]
    index = find_relevant_file_index(file_name_list)
    return relevant_files[index].getFile()


def find_relevant_file_index(file_names):
    # First, try to find a file ending with 'final_data_cc.csv'
    for index, file_name in enumerate(file_names):
        if file_name.endswith("final_data_cc.csv"):
            return index

    # If not found, look for a file ending with 'final_data.csv'
    for index, file_name in enumerate(file_names):
        if file_name.endswith("final_data.csv"):
            return index

    # If neither type of file is found, raise an exception
    raise ValueError("No final analysis CSV data found for the current plate.")


def _download_and_process_csv(original_file):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        _download_file_to_tmp(original_file, tmp)
        tmp.flush()  # Ensure all data is written to the temp file
        try:
            data = pd.read_csv(tmp.name, index_col=0)
            _process_data(data)
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty or all data is NA")
        except pd.errors.ParserError:
            logger.error("Error parsing CSV file")
        except Exception as e:  # Catch-all for unexpected exceptions
            logger.error("Unexpected error processing 'final_data.csv': %s", e)
            raise  # Raise exception after logging for further handling up the stack
        atexit.register(_cleanup_temp_file, tmp.name)


def _process_data(data):
    filtered_data = data[data["well"] == viewer_data.well.getWellPos()]
    if filtered_data.empty:
        logger.warning(
            "No data found for well %s in 'final_data.csv'",
            viewer_data.well.getWellPos(),
        )
    else:
        viewer_data.plate_data = filtered_data
        viewer_data.intensities = _get_intensity_values(data)
        logger.info("'final_data.csv' processed successfully")


def _process_plate_data():
    try:
        annotations = _get_annotations()
        relevant_files = _find_relevant_files(annotations)
        original_file = _select_csv_file(relevant_files)
        _download_and_process_csv(original_file)
    except ValueError as e:
        logger.error(str(e))
        raise


def _get_intensity_values(df: pd.DataFrame) -> dict[str, Tuple]:
    intensity_dict = {}
    for key, value in viewer_data.channel_data.items():
        max_value = int(df[f"intensity_max_{key}_nucleus"].max())
        logger.debug("Max value for %s: %s", key, max_value)
        min_value = int(df[f"intensity_min_{key}_nucleus"].min())
        logger.debug("Min value for %s: %s", key, min_value)
        intensity_dict[int(value)] = (min_value, max_value)
    return intensity_dict


def _download_file_to_tmp(original_file, tmp):
    try:
        with open(tmp.name, "wb") as f:
            for chunk in original_file.asFileObj():
                f.write(chunk)
            logger.info("Successfully downloaded file to %s", tmp.name)
    except OSError as e:
        logger.error("OS error while downloading file to %s: %s", tmp.name, e)
    except Exception as e:  # For custom or unexpected exceptions
        logger.error("Error downloading file to %s: %s", tmp.name, e)
        raise e


def _cleanup_temp_file(filepath):
    try:
        os.remove(filepath)
        logger.info("Deleted temporary file: %s", filepath)
    except FileNotFoundError:
        logger.warning("Temporary file not found: %s", filepath)
    except PermissionError:
        logger.error(
            "Permission denied when deleting temporary file: %s", filepath
        )
    except IsADirectoryError:
        logger.error("Expected a file but found a directory: %s", filepath)
    except OSError as e:
        logger.error(
            "Error deleting temporary file: %s. Error: %s", filepath, e
        )


# TODO this function needs to check if the well mapp ann is actually the right data
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


def _get_images(conn):
    assert (
        len(list(viewer_data.well.listChildren())) == 21
    ), "Stiching requires 21 images per well"
    # Lists to store the individual image arrays and ids
    flatfield_array, flatfield_channels = _get_flatfieldmask(conn)
    _check_flatfieldmask(flatfield_channels)
    image_arrays: list = []
    image_ids: list = []
    img_range = range(viewer_data.well.countWellSample())

    for index in tqdm(img_range):
        image_id, image_array = _process_omero_image(
            index, flatfield_array, conn
        )
        image_arrays.append(image_array)
        image_ids.append(image_id)

    viewer_data.images = np.stack(image_arrays, axis=0)
    viewer_data.image_ids = image_ids


def _process_omero_image(
    index: int, flatfield_array: np.array, conn: BlitzGateway
) -> tuple[int, np.array]:
    """
    Process an Omero image by applying flatfield correction and scaling.
    """
    image, image_array = get_image(
        conn, viewer_data.well.getImage(index).getId()
    )
    image_array = image_array.squeeze()
    logger.debug("Image dtype: %s", image_array.dtype)
    logger.debug("Image max: %s", image_array.max())
    corrected_array = image_array / flatfield_array
    logger.debug("Corr Image dtype: %s", corrected_array.dtype)
    logger.debug("Corr Image max: %s", corrected_array.max())
    logger.debug("Corr Image min: %s", corrected_array.min())
    if len(corrected_array.shape) == 2:
        corrected_array = corrected_array[..., np.newaxis]
    return image.getId(), corrected_array #np.stack(scaled_channels, axis=-1)


def scale_img(img: np.array) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    percentiles = np.percentile(img, (1, 99))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))


def _get_flatfieldmask(conn):
    """Gets flatfieldmasks from project linked to screen"""
    flatfield_mask_name = f"{viewer_data.plate_id}_flatfield_masks"
    for image in viewer_data.screen_dataset.listChildren():
        image_name = image.getName()
        if flatfield_mask_name == image_name:
            flatfield_mask_id = image.getId()
            flatfield_obj, flatfield_array = get_image(conn, flatfield_mask_id)
            logger.debug("Flatfield mask dtype: %s", flatfield_array.dtype)
            logger.debug("Flatfield mask range: %s, %s", flatfield_array.min(), flatfield_array.max())
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
    for channel, index in viewer_data.channel_data.items():
        try:
            assert reverse_flatfield_mask[channel] == index
        except AssertionError:
            print(
                f"Inconsistency found: {channel} is mapped to {index} in plate_map but {reverse_flatfield_mask[channel]} in flatfield_mask"
            )


def _stitch_images() -> np.ndarray:
    """Stitch the images in the array according to the specified pattern
    This is only allowed when 21 single channel 10x images from an Operetta microscope
    are provided)
    returns: [np.ndarray] stitched image of shape (5*1080, 5*1080, 1)
    """
    logger.debug("Stitching images %s", viewer_data.images.shape)
    assert viewer_data.images.shape == (
        21,
        1080,
        1080,
        1,
    ), "The input array should be 21x1080x1080x1"


    # Creating an empty array to add as spacer
    empty_array = np.full((1080, 1080, 1), fill_value=viewer_data.intensities[0][0])

    indices_pattern = [
        [
            -1,
            1,
            2,
            3,
            -1,
        ],  # Adjusted for zero-based indexing but preserved -1 for empty
        [8, 7, 6, 5, 4],  # Adjusted for zero-based indexing
        [9, 10, 0, 11, 12],  # The first image is now 0 (zero-based)
        [17, 16, 15, 14, 13],  # Adjusted for zero-based indexing
        [-1, 18, 19, 20, -1],  # Preserved -1 for empty
    ]

    # No need to adjust indices by subtracting 1 since we've directly used zero-based indices above

    # Create the stitched image
    stitched_shape = (5 * 1080, 5 * 1080, 1)  # 25x25 of 1080x1080 images
    stitched_image = np.zeros(stitched_shape)

    # Fill in the stitched image
    for i, row in enumerate(indices_pattern):
        for j, idx in enumerate(row):
            # Calculate the position where the current image should be placed
            x_pos = i * 1080
            y_pos = j * 1080

            # Place either the empty array or the corresponding image
            if idx == -1:
                stitched_image[
                    x_pos : x_pos + 1080, y_pos : y_pos + 1080, :
                ] = empty_array
            else:
                stitched_image[
                    x_pos : x_pos + 1080, y_pos : y_pos + 1080, :
                ] = viewer_data.images[idx]
    logger.debug("Stitched image shape: %s", stitched_image.shape)
    return stitched_image
    
