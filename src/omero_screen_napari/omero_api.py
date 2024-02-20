"""
This module handles the logic of the data base interactions with the omero server.
"""


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
from omero.gateway import BlitzGateway, FileAnnotationWrapper
from qtpy.QtWidgets import QMessageBox
from skimage import exposure
from tqdm import tqdm

from omero_screen_napari.omero_utils import omero_connect
from omero_screen_napari.viewer_data_module import viewer_data

# Looging

logger = logging.getLogger("omero-screen-napari")


@omero_connect
def retrieve_data(
    plate_id: str,
    well_pos: str,
    images: str,
    conn: Optional[BlitzGateway] = None,
) -> None:
    """
    Get data from Omero and add it to the viewer_data dataclass object
    :param plate_id: number of the omero screen plate
    :param well_pos: postion of the well in teh screen, example 'C2
    :param conn: omero.gateway connection
    :return: None
    """
    try:
        _get_plate_data(conn, plate_id)
        _get_well_object(well_pos)
        _get_well_metadata()
        _get_images(images, conn)
        _get_segmentation_masks(conn)

    except Exception as e:
        logging.exception("The following error occurred:")  # noqa: G004
        # Show a message box with the error message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(str(e))
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def _get_plate_data(conn, plate_id):
    if plate_id != viewer_data.plate_id:
        _get_omero_objects(conn, plate_id)
        _get_channel_data()
        _process_plate_data()
    else:
        logger.info("Plate data already retrieved")


# plate and channel_data
def _get_omero_objects(conn: BlitzGateway, plate_id: str) -> None:
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
    pixel_x, pixel_y = _get_pixel_size(viewer_data.plate)
    viewer_data.pixel_size = (
        round(pixel_x.getValue(), 1),
        round(pixel_y.getValue(), 1),
    )
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
        if annotations := list(viewer_data.plate.listAnnotations()):
            return annotations
        else:
            raise ValueError("No annotations found for the plate.")
    except ValueError as e:
        logger.warning(str(e))
        raise


def _find_relevant_files(annotations):
    if relevant_files := [
        ann for ann in annotations if isinstance(ann, FileAnnotationWrapper)
    ]:
        return relevant_files
    else:
        raise ValueError("No files attached to the current plate.")


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


def _download_csv(original_file):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        _download_file_to_tmp(original_file, tmp)
        tmp.flush()  # Ensure all data is written to the temp file
        viewer_data.csv_path = tmp.name
        logger.info("CSV file path: %s", viewer_data.csv_path)
        atexit.register(_cleanup_temp_file, tmp.name)

def _process_plate_data():
    try:
        annotations = _get_annotations()
        relevant_files = _find_relevant_files(annotations)
        original_file = _select_csv_file(relevant_files)
        _download_csv(original_file)
    except ValueError as e:
        logger.error(str(e))
        raise


def _get_intensity_values(df: pd.DataFrame) -> dict[str, Tuple]:
    intensity_dict = {}
    for key, value in viewer_data.channel_data.items():
        max_value = int(df[f"intensity_max_{key}_nucleus"].mean())
        min_value = int(df[f"intensity_min_{key}_nucleus"].min())
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


# well data
def _load_well_specific_data(well_pos: str):
    """
    Load and process data for a specific well from the CSV file.
    """
    if not viewer_data.csv_path:
        logger.error("CSV path is not set in viewer_data.")
        return

    try:
        data = pd.read_csv(viewer_data.csv_path, index_col=0)
        _filter_csv_data(data, well_pos)
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or all data is NA")
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file")
    except Exception as e:
        logger.error(f"Unexpected error processing CSV: {e}")
        raise

def _filter_csv_data(data, wellpos):
    filtered_data = data[data["well"] == wellpos]
    if filtered_data.empty:
        logger.warning(
            "No data found for well %s in 'final_data.csv'",
            viewer_data.well.getWellPos(),
        )
    else:
        viewer_data.plate_data = filtered_data
        viewer_data.intensities = _get_intensity_values(data)
        logger.info("'final_data.csv' processed successfully")


def _get_well_object(well_pos: str):
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
    _load_well_specific_data(well_pos)


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


def _get_images(images, conn):
    image_number = f"image_{images} of {viewer_data.well.getWellPos()}"
    # Lists to store the individual image arrays and ids
    flatfield_array, flatfield_channels = _get_flatfieldmask(conn)
    _check_flatfieldmask(flatfield_channels)
    logger.info(
        f"Gathering {image_number}"  # noqa: G004
    )
    image_arrays: list = []
    image_ids: list = []
    pattern = r"^\d+-\d+$"
    if images == "All" or re.match(pattern, images):
        if images == "All":
            img_range = range(viewer_data.well.countWellSample())
        else:
            start, end = map(int, images.split("-"))
            img_range = range(start, end + 1)
        for index in tqdm(img_range):
            image_id, image_array = _process_omero_image(
                index, flatfield_array, conn
            )
            image_arrays.append(image_array)
            image_ids.append(image_id)
    elif images.isdigit():
        index: int = int(images)
        image_id, image_array = _process_omero_image(
            index, flatfield_array, conn
        )
        image_arrays.append(image_array)
        image_ids.append(image_id)
    else:
        raise ValueError(f"Invalid image number: {images}")
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
    corrected_array = image_array / flatfield_array
    if len(corrected_array.shape) == 2:
        corrected_array = corrected_array[..., np.newaxis]
    # Iterate through each channel to scale it
    scaled_channels = []
    for i in range(corrected_array.shape[-1]):
        scaled_channel = scale_img(
            corrected_array[..., i], viewer_data.intensities[i]
        )
        scaled_channels.append(scaled_channel)
    return image.getId(), np.stack(scaled_channels, axis=-1)


def scale_img(img: np.array, intensities: tuple) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    return exposure.rescale_intensity(img, in_range=intensities)


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
    for channel, index in viewer_data.channel_data.items():
        try:
            assert reverse_flatfield_mask[channel] == index
        except AssertionError:
            print(
                f"Inconsistency found: {channel} is mapped to {index} in plate_map but {reverse_flatfield_mask[channel]} in flatfield_mask"
            )


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
