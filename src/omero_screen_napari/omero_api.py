"""
This module handles the logic of the data base interactions with the omero server.
"""


import atexit
import logging
import os
import re
import tempfile
from typing import List, Optional, Tuple
from pathlib import Path

import napari
import numpy as np
import omero
import pandas as pd
from ezomero import get_image
from omero.gateway import BlitzGateway, FileAnnotationWrapper
from qtpy.QtWidgets import QMessageBox
from skimage import exposure
from tqdm import tqdm

from omero_screen_napari.omero_utils import omero_connect, load_dotenv

# Looging

logger = logging.getLogger("omero-screen-napari")


@omero_connect
def retrieve_wells(
    plate_id: str,
    well_positions: List[str],
    images: str,
    viewer_data,
    conn: Optional[BlitzGateway] = None,
) -> None:
    for index, well_pos in enumerate(well_positions):
        retrieve_well_data(viewer_data, plate_id, well_pos, index, images, conn)


def retrieve_well_data(
    viewer_data,
    plate_id: str,
    well_pos: str,
    index: int,
    images: str,
    conn: Optional[BlitzGateway],
    stitch: bool = False,
) -> None:
    """
    Get data from Omero and add it to the viewer_data dataclass object.
    :param plate_id: number of the omero screen plate
    :param well_pos: postion of the well in teh screen, example 'C2
    :param conn: omero.gateway connection
    :return: None
    """
    try:
        _get_plate(viewer_data, conn, plate_id)
        _get_well_object(viewer_data, conn, well_pos, index, images)
        _get_segmentation_masks(viewer_data, conn)
        if stitch:
            _stitch_images(viewer_data)

    except Exception as e:
        logging.exception("The following error occurred:")  # noqa: G004
        # Show a message box with the error message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(str(e))
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def _get_plate(viewer_data, conn: BlitzGateway, plate_id: str) -> None:
    """Highlevel function that combines various data transactions from Omero
    at plate level. Sets up relevant metadata in viewer_data class via helper functions.
    Args:
        conn (_type_): _description_
        plate_id (_type_): _description_
    """
    if plate_id != viewer_data.plate_id:
        _get_plate_metadata(viewer_data, conn, plate_id)
        _get_channel_data(viewer_data)
        _process_plate_data(viewer_data)
        logger.info("Downloading plate data")
    logger.info("Reusing plate data")    


# plate and channel_data
def _get_plate_metadata(viewer_data, conn: BlitzGateway, plate_id: str) -> None:
    """
    Links to Omero via plate_id and sets up basic screen info in viewer_data.
    This includes viewer_data.plate, viewer_data.plate_name, viewer_data.pixel_size,
    and viewer_data.screen_dataset to access flatfield correction and segmentation masks.
    Args:
        conn: The Omero connection object.
        plate_id: The ID of the plate.
    Raises:
        ValueError: If the plate does not exist.

    Returns:
        None.
    """
    # get plate object
    viewer_data.plate_id = int(plate_id)
    logger.info(f"Retrieveing data for plate: {viewer_data.plate_id} from Omero")
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
    _get_flatfieldmask(viewer_data, conn)


def _get_pixel_size(plate: omero.gateway.PlateWrapper) -> tuple:
    """
    Get the pixel size from the plate metadata.

    Args:
        plate: The plate object.

    Returns:
        A tuple containing the physical size in X and Y dimensions.

    Raises:
        ValueError: If no wells or images are found in the plate.
    """
    try:
        wells = list(plate.listChildren())
        if not wells:
            logger.debug("No wells found in the plate, raising ValueError.")
            raise ValueError("No wells found in the plate.")

        well = wells[0]
        image = well.getImage(0)
        if image is None:
            logger.debug(
                "No images found in the first well, raising ValueError."
            )
            raise ValueError("No images found in the first well.")

        pixels = image.getPrimaryPixels()
        if pixels is None:
            logger.debug(
                "No pixel data found for the image, raising ValueError."
            )
            raise ValueError("No pixel data found for the image.")

        pixel_size_x, pixel_size_y = (
            pixels.getPhysicalSizeX(),
            pixels.getPhysicalSizeY(),
        )
        logger.info(
            f"Retrieved pixel sizes: X={pixel_size_x}, Y={pixel_size_y}"
        )  # noqa: G004
        return pixel_size_x, pixel_size_y
    except Exception as e:
        # Log with exception details at retrieve data to show message as widget
        raise


def _get_flatfieldmask(viewer_data, conn):
    """Gets flatfieldmasks from project linked to screen"""
    flatfield_mask_name = f"{viewer_data.plate_id}_flatfield_masks"
    flatfield_mask_found = False
    for image in viewer_data.screen_dataset.listChildren():
        image_name = image.getName()
        if flatfield_mask_name == image_name:
            flatfield_mask_id = image.getId()
            flatfield_obj, flatfield_array = get_image(conn, flatfield_mask_id)
            flatfield_channels = dict(_get_map_ann(flatfield_obj))
            for key, value in flatfield_channels.items():
                flatfield_channels[key] = value.strip()
            _check_flatfieldmask(viewer_data, flatfield_channels)
            viewer_data.flatfield_masks = flatfield_array
            flatfield_mask_found = (
                True  # Set flag to true since flatfield mask is found
            )
            break  # Exit loop after finding the flatfield mask
        if not flatfield_mask_found:  # Check if the flatfield mask was not found after looping through all images
            logger.warning("No flatfield mask found for the plate")
            raise ValueError("No flatfield mask found for the plate")


def _check_flatfieldmask(viewer_data, flatfield_channels):
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


def _get_channel_data(viewer_data):
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


def _get_annotations(viewer_data):
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


def _download_csv(viewer_data, original_file):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        _download_file_to_tmp(original_file, tmp)
        tmp.flush()  # Ensure all data is written to the temp file
        viewer_data.csv_path = tmp.name
        logger.info("CSV file path: %s", viewer_data.csv_path)
        atexit.register(_cleanup_temp_file, tmp.name)


def _process_plate_data(viewer_data):
    try:
        annotations = _get_annotations(viewer_data)
        relevant_files = _find_relevant_files(annotations)
        original_file = _select_csv_file(relevant_files)
        _download_csv(viewer_data, original_file)
    except ValueError as e:
        logger.error(str(e))
        raise


def _get_intensity_values(viewer_data, df: pd.DataFrame) -> dict[str, Tuple]:
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


def _get_well_object(viewer_data, conn, well_pos: str, index: int, images: str):
    if well_pos not in viewer_data.well_name:
        viewer_data.well_name.append(well_pos)
        well_found = False  # Initialize a flag to check if the well is found
        # get well object
        for well in viewer_data.plate.listChildren():
            if well.getWellPos() == well_pos:
                viewer_data.well.append(well)
                viewer_data.well_id.append(well.getId())
                _load_well_specific_data(viewer_data, well_pos)
                _get_well_metadata(viewer_data, index)
                _get_images(viewer_data, images, index, conn)
                well_found = (
                    True  # Set the flag to True when the well is found
                )
                break  # Exit the loop as the well is found
        if not well_found:  # Raise an error if the well was not found
            raise ValueError(f"Well with position {well_pos} does not exist.")
       
    else:
        logger.info(f"Well {well_pos} already retrieved")


def _load_well_specific_data(viewer_data, well_pos: str):
    """
    Load and process data for a specific well from the CSV file.
    """
    if not viewer_data.csv_path:
        logger.error("CSV path is not set in viewer_data.")
        return

    try:
        data = pd.read_csv(viewer_data.csv_path, index_col=0)
        _filter_csv_data(viewer_data, data, well_pos)
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or all data is NA")
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file")
    except Exception as e:
        logger.error(f"Unexpected error processing CSV: {e}")
        raise


def _filter_csv_data(viewer_data, data, wellpos):
    filtered_data = data[data["well"] == wellpos]
    if filtered_data.empty:
        logger.warning(
            "No data found for well %s in 'final_data.csv'",
            viewer_data.well.getWellPos(),
        )
    else:
        viewer_data.plate_data = pd.concat(
            [viewer_data.plate_data, filtered_data]
        )
        viewer_data.intensities = _get_intensity_values(viewer_data, data)
        logger.info("'final_data.csv' processed successfully")


# TODO this function needs to check if the well mapp ann is actually the right data
def _get_well_metadata(viewer_data, index):
    map_ann = None
    for ann in viewer_data.well[index].listAnnotations():
        if ann.getValue():
            map_ann = dict(ann.getValue())
    if map_ann:
        viewer_data.metadata.append(map_ann)
    else:
        raise ValueError(
            f"No map annotation found for well {viewer_data.well[index].getWellPos()}"
        )


# image data


def _get_images(viewer_data, images, index, conn):
    image_number = f"image_{images} of {viewer_data.well_name[index]}"
    # Lists to store the individual image arrays and ids
    logger.info(
        f"Gathering {image_number}"  # noqa: G004
    )
    image_arrays: list = []
    image_ids: list = []
    pattern = r"^\d+-\d+$"
    if images == "All" or re.match(pattern, images):
        if images == "All":
            img_range = range(viewer_data.well[index].countWellSample())
        else:
            start, end = map(int, images.split("-"))
            img_range = range(start, end + 1)
        for image_index in tqdm(img_range):
            image_id, image_array = _process_omero_image(
                viewer_data, image_index, index, conn
            )
            image_arrays.append(image_array)
            image_ids.append(image_id)
    elif images.isdigit():
        image_index: int = int(images)
        image_id, image_array = _process_omero_image(viewer_data, image_index, index, conn)
        image_arrays.append(image_array)
        image_ids.append(image_id)
    else:
        raise ValueError(f"Invalid image number: {images}")
    if viewer_data.images.size == 0:
        viewer_data.images = np.squeeze(np.stack(image_arrays, axis=0), axis=(1, 2))
        viewer_data.image_ids = image_ids

    else:
        logger.debug(f"the new image's shape is {np.squeeze(np.stack(image_arrays, axis=0), axis=(1, 2))}")
        logger.debug(f"the old image's shape is {viewer_data.images.shape})")
        viewer_data.images = np.concatenate(
            (viewer_data.images, np.squeeze(np.stack(image_arrays, axis=0), axis=(1, 2))), axis=0
        )
        viewer_data.image_ids.extend(image_ids)
    logger.debug(f"Images retrieved successfully: {viewer_data.image_ids}")
    

def _process_omero_image(
    viewer_data,
    image_index: int, index, conn: BlitzGateway
) -> tuple[int, np.array]:
    """
    Process an Omero image by applying flatfield correction and scaling.
    """
    logger.debug(
        f"Processing image {viewer_data.well[index].getImage(image_index).getId()}"
    )
    image, image_array = get_image(
        conn, viewer_data.well[index].getImage(image_index).getId()
    )
    image_array = image_array.squeeze()
    corrected_array = image_array / viewer_data.flatfield_masks
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


# get the image labels
def _get_segmentation_masks(viewer_data, conn):
    """
    Get segmentation masks as mask list
    """
    mask_list = []
    id_list = []
    name_to_id = {
        image.getName(): image.getId()
        for image in viewer_data.screen_dataset.listChildren()
    }
    for image_name in viewer_data.image_ids:
        image_id = name_to_id.get(f"{image_name}_segmentation")
        if image_id not in viewer_data.label_ids:
            try:
                mask, mask_array = get_image(conn, image_id)
                mask_array = mask_array.squeeze()
                mask_list.append(mask_array)
                id_list.append(image_id)
            except Exception as e:
                logger.error(f"Error retrieving segmentation mask: {e}")
                raise
    if viewer_data.labels.size == 0 and mask_list:
        viewer_data.labels = np.stack(mask_list, axis=0)
        viewer_data.label_ids = id_list
        logger.info("Segmentation masks retrieved successfully")
    elif mask_list:
        # If not empty, stack the new images along the index axis
        viewer_data.labels = np.concatenate(
            (viewer_data.labels, np.stack(mask_list, axis=0)), axis=0
        )
        viewer_data.label_ids.extend(id_list)
        logger.info("Segmentation masks added successfully")
    else:
        logger.warning("No segmentation masks found for the plate")


# stitch images


def _stitch_images(viewer_data) -> np.ndarray:
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
    empty_array = np.full(
        (1080, 1080, 1), fill_value=viewer_data.intensities[0][0]
    )

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
