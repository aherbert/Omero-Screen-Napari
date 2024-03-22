import logging
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import omero
import polars as pl
from ezomero import get_image
from omero.gateway import (
    BlitzGateway,
    FileAnnotationWrapper,
    MapAnnotationWrapper,
    PlateWrapper,
)
from qtpy.QtWidgets import QMessageBox
from skimage import exposure
from tqdm import tqdm

from omero_screen_napari._omero_utils import omero_connect
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import (
    omero_data,
    reset_omero_data,
)

logger = logging.getLogger("omero-screen-napari")


def parse_omero_data(
    omero_data: OmeroData,
    plate_id: int,
    well_pos: str,
    image_input: str,
    conn: BlitzGateway,
) -> None:
    """
    This functions controls the flow of the data
    via the seperate parser classes to the omero_data class that stores omero plate metadata
    to be supplied to the napari viewer.
    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        plate_id (int): Omero Screen Plate ID number provided by welldata_widget
        well_list (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        image_input (str): String Describing the combinations of images to be supplied for each well
        The images options are 'All'; 0-3; 1, 3, 4.
    """
    try:
        parse_plate_data(omero_data, plate_id, well_pos, image_input, conn)
        # clear well and image data to avoid appending to existing data
        omero_data.reset_well_and_image_data() 
        for well_pos in omero_data.well_pos_list:
            logger.info(f"Processing well {well_pos}")  # noqa: G004
            well_image_parser(omero_data, well_pos, conn)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error parsing plate data: {e}")
        # # Show a message box with the error message
        # msg_box = QMessageBox()
        # msg_box.setIcon(QMessageBox.Warning)
        # msg_box.setText(str(e))
        # msg_box.setWindowTitle("Error")
        # msg_box.setStandardButtons(QMessageBox.Ok)
        # msg_box.exec_()


def parse_plate_data(
    omero_data,
    plate_id: int,
    well_pos: str,
    image_input: str,
    conn: BlitzGateway,
) -> None:
    # check if plate-ID already supplied, if it is then skip PlateHandler
    if plate_id != omero_data.plate_id:
        logger.info(f"Loading new plate data for plate {plate_id}")  # noqa: G004
        reset_omero_data()

        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn
        )
        user_input.parse_data()
        omero_data.plate_id = plate_id
        plate = conn.getObject("Plate", plate_id)
        omero_data.plate = plate
        csv_parser = CsvFileParser(omero_data)
        csv_parser.parse_csv()
        channel_parser = ChannelDataParser(omero_data)
        channel_parser.parse_channel_data()
        flatfield_parser = FlatfieldMaskParser(omero_data, conn)
        flatfield_parser.parse_flatfieldmask()
        scale_intensity_parser = ScaleIntensityParser(omero_data)
        scale_intensity_parser.parse_intensities()
        pixel_size_parser = PixelSizeParser(omero_data)
        pixel_size_parser.parse_pixel_size_values()

    else:
        logger.info(f"Plate data for plate {plate_id} already exists.")
        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn
        )
        user_input.parse_data()


def well_image_parser(omero_data, well_pos: str, conn):
    well_data_parser = WellDataParser(omero_data, well_pos)
    well_data_parser.parse_well()
    well = well_data_parser._well
    image_parser = ImageParser(omero_data, well, conn)
    image_parser.parse_images()


# -----------------------------------------------USER INPUT -----------------------------------------------------
class UserInput:
    def __init__(
        self,
        omero_data: OmeroData,
        plate_id: int,
        well_pos: str,
        image_input: str,
        conn,
    ):
        self._omero_data = omero_data
        self._plate_id = plate_id
        self._well_pos = well_pos
        self._images = image_input
        self._conn = conn
        self._plate: Optional[omero.gateway.PlateWrapper] = None
        self._image_number: Optional[int] = None
        self._well_pos_list: Optional[List[str]] = None

    def parse_data(self):
        self._check_plate_id()
        self._parse_image_number()
        self._well_data_parser()
        self._omero_data.well_pos_list = self._well_pos_list
        self._image_index_parser()
        self._omero_data.image_index = self._image_index

    def _check_plate_id(self):
        self._plate = self._conn.getObject("Plate", self._plate_id)
        if not self._plate:
            logger.error(f"Plate with ID {self._plate_id} does not exist.")
            raise ValueError(f"Plate with ID {self._plate_id} does not exist.")
        else:
            logger.info(f"Found plate with ID {self._plate_id} in omero.")

    def _parse_image_number(self):
        first_well = list(self._plate.listChildren())[0]
        self._image_number = len(list(first_well.listChildren()))

    def _well_data_parser(self):
        pattern = re.compile(r"^[A-H][1-9]|1[0-2]$")
        self._well_pos_list = [
            item.strip() for item in self._well_pos.split(",")
        ]
        for item in self._well_pos_list:
            if not pattern.match(item):
                logger.error(
                    f"Invalid well input format: '{item}'. Must be A-H followed by 1-12."
                )
                raise ValueError(
                    f"Invalid well input format: '{item}'. Must be A-H followed by 1-12."
                )

    def _image_index_parser(self):
        index = self._images

        if not (
            index.lower() == "all"
            or re.match(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$", index)
        ):
            logger.error(
                f"Image input '{index}' doesn't match any of the expected patterns 'All, 1-3, 1'."
            )
            raise ValueError(
                f"Image input '{index}' doesn't match any of the expected patterns 'All, 1-3, 1'."
            )

        if index.lower() == "all":
            self._image_index = list(
                range(self._image_number)
            )  # Assuming well count is inclusive and starts at 1
        elif "-" in index:
            # Handle range, e.g., '1-3'
            start, end = map(int, index.split("-"))
            self._image_index = list(range(start, end + 1))
        elif "," in index:
            # Handle list, e.g., '1, 4, 5'
            self._image_index = list(map(int, index.split(", ")))
        else:
            # Handle single number, e.g., '1'
            self._image_index = [int(index)]


# -----------------------------------------------PLATE DATA -----------------------------------------------------

# classes to parse the plate data from the Omero Plate: csv file, channel data, flatfield mask, pixel size,
# image intensity and pixels size

# -----------------------------------------------CSV FILE -----------------------------------------------------


class CsvFileParser:
    """
    Class to handle csv file retrieval and processing from Omero Plate
    Class methods:
    handle_csv: Orchestrate the csv file handling. Check if csv file is available, if not download it.
    and make it available to the omero_data class as a polars LazyFrame object.
    _csv_available: Helper; checks if any csv file in the target directory contains the string self.plate_id in its name.
    _get_csv_file: Helper; gets the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
    _download_csv: Helper; downloads the csv file from the Omero Plate. If _cc.csv the file name will be

    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_id: int = omero_data.plate_id
        self._plate = omero_data.plate
        self._data_path: Path = omero_data.data_path
        self._csv_file_path: Optional[Path] = None

    def parse_csv(self):
        """Orchestrate the csv file handling. Check if csv file is available, if not download it.
        and make it available to the omero_data class as a polars LazyFrame object.

        """
        if not self._csv_available():
            logger.info("Downloading csv file from Omero")  # noqa: G004
            self._get_csv_file()
            self._download_csv()
        else:  # csv file already exists
            logger.info(
                "CSV file already exists in local directory. Skipping download."
            )
        omero_data.plate_data = pl.scan_csv(self._csv_file_path)
        self._omero_data.csv_path = self._csv_file_path

    # helper functions for csv handling

    def _csv_available(self):
        """
        Check if any csv file in the directory contains the string self.plate_id in its name.
        """
        # make the directory to store the csv file if it does not exist
        self._data_path.mkdir(exist_ok=True)
        for file in self._data_path.iterdir():
            if file.is_file() and str(self._plate_id) in file.name:
                self._csv_file_path = file
                return True

    def _get_csv_file(self):
        """
        Get the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
        """
        self._original_file = None
        file_anns = self._plate.listAnnotations()
        for ann in file_anns:
            if isinstance(ann, FileAnnotationWrapper):
                name = ann.getFile().getName()
                if name.endswith("_cc.csv"):
                    self._file_name = name
                    self._original_file = ann.getFile()
                    break  # Prioritize and stop searching if _cc.csv is found
                elif (
                    name.endswith("final_data.csv")
                    and self._original_file is None
                ):
                    self._file_name = name
                    self._original_file = ann.getFile()
                    # Don't break to continue searching for a _cc.csv file

        if self._original_file is None:
            logger.error("No suitable csv file found for the plate.")
            raise ValueError("No suitable csv file found for the plate.")

    def _download_csv(self):
        """
        Download the csv file from the Omero Plate. If _cc.csv the file name will be
        {plate_id}_cc.csv, otherwise {plate_id}_data.csv
        """
        saved_name = f"{self._plate_id}_{self._file_name.split('_')[-1]}"
        logger.info(f"Downloading csv file {self._file_name} to {saved_name}")  # noqa: G004
        self._csv_file_path = self._data_path / saved_name
        file_content = self._original_file.getFileInChunks()
        with open(self._csv_file_path, "wb") as file_on_disk:
            for chunk in file_content:
                file_on_disk.write(chunk)


# -----------------------------------------------CHANNEL DATA -----------------------------------------------------


class ChannelDataParser:
    """
    Class to handle channel data retrieval and processing from Omero Plate.
    class methods:
    _get_channel_data: Coordinates the retrieval of channel data from Omero Plate
    _get_map_ann: Hhlper method that scans plate map annotations for channel data
    and raises an exceptions if no suitable annotation is found
    _tidy_up_channel_data: Helper method that processes the channel data to a dictionary
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate = omero_data.plate
        self._plate_id: int = omero_data.plate_id

    def parse_channel_data(self):
        """Process channel data to dictionary {channel_name: channel_index}
        and add it to viewer_data; raise exception if no channel data found."""
        self._get_map_ann()
        self._tidy_up_channel_data()
        self._omero_data.channel_data = self.channel_data
        logger.info(
            f"Channel data: {self.channel_data} loaded to omero_data.channel_data"
        )  # noqa: G004

    def _get_map_ann(self):
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """

        annotations = self._plate.listAnnotations()

        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]

        if not map_annotations:
            logger.error(
                f"No MapAnnotations found for plate {self._plate_id}."
            )  # noqa: G004
            raise ValueError(
                f"No MapAnnotations found for plate {self._plate_id}."
            )

        # Search for a map annotation with "dapi" or "hoechst" in its value (case-insensitive)
        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Assuming the value is a list of tuples or similar structure
            for key, _value in ann_value:
                if key.lower() in ["dapi", "hoechst"]:
                    self.map_annotations = ann_value
                    return  # Return the first matching map annotation's value

        # If no matching annotation is found, raise a specific exception
        logger.error(
            f"No DAPI or Hoechst channel information found for plate {self._plate_id}."  # noqa: G004
        )  # noqa: G004
        raise ValueError(
            f"No DAPI or Hoechst channel information found for platplate {self._omero_data.plate_id}."
        )

    def _tidy_up_channel_data(self):
        """
        Convert channel map annotation from list if tuples to dictionary.
        Eliminate spaces and swap Hoechst to DAPI if necessary.
        """
        channel_data = dict(self.map_annotations)
        self.channel_data = {
            key.strip(): value for key, value in channel_data.items()
        }
        if "Hoechst" in self.channel_data:
            self.channel_data["DAPI"] = self.channel_data.pop("Hoechst")
        # check if one of the channels is DAPI otherwise raise exception


# -----------------------------------------------FLATFIELD CORRECTION-----------------------------------------------------


class FlatfieldMaskParser:
    def __init__(self, omero_data: OmeroData, conn: BlitzGateway):
        self._omero_data: OmeroData = omero_data
        self._conn: BlitzGateway = conn

    def parse_flatfieldmask(self):
        self._load_dataset()
        self._get_flatfieldmask()
        self._get_map_ann()
        self._check_flatfieldmask()
        omero_data.flatfield_mask = self._flatfield_array

    def _load_dataset(self):
        """
        Looks for a data set that matches the plate name in the screen project
        with id supplied by omero_data.project_id via the env variable.
        if it finds the data set it assigns it to the omero_data object
        otherwise it will throw an error becuase the program cant proceed without
        flatfielmasks and segmentationlabels stored in the data set.

        Raises:
            ValueError: If the plate has not been assigned a dataset.
        """
        project = self._conn.getObject("Project", self._omero_data.project_id)
        if dataset := self._conn.getObject(
            "Dataset",
            attributes={"name": str(self._omero_data.plate_id)},
            opts={"project": project.getId()},
        ):
            self._omero_data.screen_dataset = dataset
        else:
            logger.error(
                f"The plate {omero_data.plate_name} has not been assigned a dataset."
            )
            raise ValueError(
                f"The plate {omero_data.plate_name} has not been assigned a dataset."
            )

    def _get_flatfieldmask(self):
        """Gets flatfieldmasks from project linked to screen"""
        flatfield_mask_name = f"{self._omero_data.plate_id}_flatfield_masks"
        logger.debug(f"Flatfield mask name: {flatfield_mask_name}")  # noqa: G004
        flatfield_mask_found = False
        for image in self._omero_data.screen_dataset.listChildren():
            image_name = image.getName()
            logger.debug(f"Image name: {image_name}")  # noqa: G004
            if flatfield_mask_name == image_name:
                flatfield_mask_found = True
                flatfield_mask_id = image.getId()
                self._flatfield_obj, self._flatfield_array = get_image(
                    self._conn, flatfield_mask_id
                )
                break  # Exit the loop once the flatfield mask is found
        if not flatfield_mask_found:
            logger.error(
                f"No flatfieldmasks found in dataset {self._omero_data.screen_dataset}."
            )  # noqa: G004
            raise ValueError(
                f"No flatfieldmasks found in dataset {self._omero_data.screen_dataset}."
            )

    def _get_map_ann(self):
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """
        annotations = self._flatfield_obj.listAnnotations()
        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]
        if not map_annotations:
            logger.error(
                f"No Flatfield Mask Channel info found in datset {self._omero_data.screen_dataset}."
            )  # noqa: G004
            raise ValueError(
                f"No Flatfield Mask Channel info found in datset {self._omero_data.screen_dataset}."
            )
        # Search for a map annotation with "dapi" or "hoechst" in its value (case-insensitive)
        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Assuming the value is a list of tuples or similar structure
            for _key, value in ann_value:
                if value.lower().strip() in ["dapi", "hoechst"]:
                    self._flatfield_channels = dict(ann_value)
                    logger.debug(
                        f"Flatfield mask channels: {self._flatfield_channels}"  # noqa: G004
                    )  # noqa: G004
                    return
        # If no matching annotation is found, raise a specific exception
        logger.error(
            "No DAPI or Hoechst channel information found for flatfieldmasks."
        )  # noqa: G004
        raise ValueError(
            "No DAPI or Hoechst channel information found for flatfieldmasks."
        )

    def _check_flatfieldmask(self):
        """
        Checks if the mappings with the plate channel date are consistent.
        self.flatfield_channels are {'channel_0' : 'DAPI}
        omero_data.channel_data are {'DAPI' : 0}

        Raises :
            InconsistencyError: If the mappings are inconsistent.
        """

        class InconsistencyError(ValueError):
            pass

        logger.info(
            f"Flatfield channels: {self._flatfield_channels.values()}, channel data {omero_data.channel_data.keys()}"  # noqa: G004
        )
        if list(self._flatfield_channels.values()) != list(
            omero_data.channel_data.keys()
        ):
            error_message = "Inconsistency found: flatfield_mask and plate_map have different channels"
            logger.error(error_message)
            raise InconsistencyError(error_message)


# -----------------------------------------------SCALE INTENSITY -----------------------------------------------------


class ScaleIntensityParser:
    """
    The class extracts the caling values for image contrasts to display the different channels in napari.
    To compare intesities across different wells a single global contrasting value is set. This is based on the
    the min and max values for each channel. These data are extracted from the polars LazyFrame object stored in
    omero_data.plate_data.
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_data: pl.LazyFrame = omero_data.plate_data
        self._keyword: Optional[str] = None
        self._intensities: Tuple[dict] = ({}, {})

    def parse_intensities(self):
        self._set_keyword()
        self._get_values()
        self._omero_data.intensities = self._intensities

    def _set_keyword(self):
        if not hasattr(self, "_plate_data") or self._plate_data is None:
            raise ValueError("Dataframe 'plate_data' does not exist.")

        # Initialize keyword to None
        self._keyword = None

        # Check for presence of "_cell" and "_nucleus" in the list of strings
        has_cell = any("_cell" in s for s in self._plate_data.columns)
        has_nucleus = any("_nucleus" in s for s in self._plate_data.columns)

        # Set keyword based on presence of "_cell" or "_nucleus"
        if has_cell:
            self._keyword = "_cell"
        elif has_nucleus:
            self._keyword = "_nucleus"
        else:
            # If neither is found, raise an exception
            raise ValueError(
                "Neither '_cell' nor '_nucleus' is present in the dataframe columns."
            )

    def _get_values(self):
        """
        Filter the dataframe to only include columns with the keyword.
        """
        intensity_dict = {}
        for channel, channel_value in self._omero_data.channel_data.items():
            cols = (
                f"intensity_max_{channel}{self._keyword}",
                f"intensity_min_{channel}{self._keyword}",
            )
            for col in cols:
                if col not in self._plate_data.columns:
                    logger.error(f"Column '{col}' not found in DataFrame.")
                    raise ValueError(f"Column '{col}' not found in DataFrame.")

            max_value = (
                self._plate_data.select(pl.col(cols[0])).mean().collect()
            )
            min_value = (
                self._plate_data.select(pl.col(cols[1])).min().collect()
            )

            intensity_dict[int(channel_value)] = (
                int(min_value[0, 0]),
                int(max_value[0, 0]),
            )

        self._intensities = intensity_dict


# -----------------------------------------------PIXEL SIZE -----------------------------------------------------


class PixelSizeParser:
    """
    The class extracts the pixel size from the plate metadata. For this purpose it uses the first well and image
    to extract the pixel size in X and Y dimensions. The pixel size is then stored in the omero_data class.
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate = omero_data.plate
        self._random_wells: Optional[List[omero.gateway.WellWrapper]] = None
        self._random_images: Optional[List[omero.gateway.ImageWrapper]] = None
        self._pixel_size: Optional[Tuple[int]] = None

    def parse_pixel_size_values(self):
        self._check_wells_and_images()
        self._check_pixel_values()
        self._omero_data.pixel_size = self._pixel_size

    def _check_wells_and_images(self):
        """
        Check if any wells are found in the plate and if each well has images.
        Extract the first image of two random wells.
        """
        wells = list(self._plate.listChildren())
        if not wells:
            logger.debug("No wells found in the plate, raising ValueError.")
            raise ValueError("No wells found in the plate.")
        self._random_wells = random.sample(wells, 2)
        image_list = []
        for i, well in enumerate(self._random_wells):
            try:
                image = well.getImage(0)  # Attempt to get the first image
                image_list.append(image)
            except Exception as e:  # Catches any exception raised by getImage(0)  # noqa: BLE001
                logger.error(f"Unable to retrieve image from well {i}: {e}")
                raise ValueError(
                    f"Unable to retrieve image from well {i}: {e}"
                ) from e

        self._random_images = image_list

    def _get_pixel_values(self, image) -> tuple:
        """
        Get pixel values from a single OMERO image object.
        Returns a tuple of floats for x, y values.
        """
        x_size = image.getPixelSizeX()
        y_size = image.getPixelSizeY()

        # Check if either x_size or y_size is None before rounding
        if x_size is None or y_size is None:
            logger.error(
                "No pixel data found for the image, raising ValueError."
            )
            raise ValueError("No pixel data found for the image.")

        # Since both values are not None, proceed to round them
        return (round(x_size, 1), round(y_size, 1))

    def _check_pixel_values(self):
        """
        Check if pixel values from the two random images are identical and not 0
        and load the pixel size tuple to the omero_data class.
        """
        pixel_well_1 = self._get_pixel_values(self._random_images[0])
        pixel_well_2 = self._get_pixel_values(self._random_images[1])

        if 0 in pixel_well_1 + pixel_well_2:
            logger.error("One of the pixel sizes is 0")
            raise ValueError("One of the pixel sizes is 0")
        elif pixel_well_1 == pixel_well_2:
            self._pixel_size = pixel_well_1
        else:
            logger.error("Pixel sizes are not identical between wells")
            raise ValueError("Pixel sizes are not identical between wells")


# -----------------------------------------------Well Data-----------------------------------------------------


class WellDataParser:
    """
    Class to handle well data retrieval and processing from Omero Plate.
    Well metadata, csv data and imaged are checked and retrieved for each well
    and added to the omaro_data data class for further usage by the Napari Viewer.
    """

    def __init__(
        self,
        omero_data: OmeroData,
        well_pos: str,
    ):
        self._omero_data = omero_data
        self._plate = omero_data.plate
        self._well_pos = well_pos
        self._well: Optional[omero.gateway.WellWrapper] = None
        self._well_id: Optional[str] = None
        self._metadata: Optional[dict] = None
        self._well_ifdata: Optional[pl.DataFrame] = None

    def parse_well(self):
        self._parse_well_object()
        self._omero_data.well_list.append(self._well)
        self._omero_data.well_id_list.append(self._well_id)
        self._get_well_metadata()
        self._omero_data.well_metadata_list.append(self._metadata)
        self._load_well_csvdata()
        self._omero_data.well_ifdata = pl.concat(
            [self._omero_data.well_ifdata, self._well_ifdata]
        )

    def _parse_well_object(self):
        well_found = False
        for well in self._plate.listChildren():
            if well.getWellPos() == self._well_pos:
                self._well = well
                self._well_id = well.getId()
                logger.info(f"Well {self._well_pos} retrieved")
                well_found = True
                break
        if not well_found:
            logger.error(
                f"Well with position {self._well_pos} does not exist."
            )  # Raise an error if the well was not found
            raise ValueError(
                f"Well with position {self._well_pos} does not exist."
            )

    def _get_well_metadata(self):
        map_ann = None
        for ann in self._well.listAnnotations():
            if ann.getValue():
                map_ann = dict(ann.getValue())
        if map_ann:
            self._metadata = map_ann
        else:
            raise ValueError(
                f"No map annotation found for well {self._well_pos}"
            )

    def _load_well_csvdata(self):
        df = self._omero_data.plate_data
        self._well_ifdata = df.filter(
            pl.col("well") == self._well_pos
        ).collect()


# -----------------------------------------------Image Data-----------------------------------------------------
class ImageParser:
    """
    Class to handle image data retrieval and processing for a given well.
    Each image is loaded using the ezomero get_image function and then flatfield corrected and scaled.
    The indivudauk images and image names are stored in two seperat lists. The image list
    with the numpy array is finally convereted to a numpy array of the dimension
    number of images, 1080, 1080, number of channels and stored in the omero_data class as omero_data.images.
    The image ids are added to the image_ids list in the omero_data class
    """

    def __init__(
        self,
        omero_data: OmeroData,
        well: omero.gateway.WellWrapper,
        conn: BlitzGateway,
    ):
        self._omero_data: OmeroData = omero_data
        self._well: omero.gateway.WellWrapper = well
        self._conn: BlitzGateway = conn
        self._image_index: list[int] = self._omero_data.image_index
        self._images: Optional[np.array] = None

    def parse_images(self):  
        self._collect_images()
        if self._omero_data.images.shape == (0,):
            self._omero_data.images = np.squeeze(
                np.stack(self._image_arrays, axis=0), axis=(1, 2)
            )
            self._omero_data.image_ids = self._image_ids
            logger.info(f"Images loaded to empty omero_data.images, new shape: {omero_data.images.shape}")  # noqa: G004
        else:
            self._omero_data.images = np.concatenate(
                (
                    self._omero_data.images,
                    np.squeeze(
                        np.stack(self._image_arrays, axis=0), axis=(1, 2)
                    ),
                ),
                axis=0,
            )
            self._omero_data.image_ids.extend(self._image_ids)
            logger.info(f"Images added to omero_data.images, new shape: {omero_data.images.shape}")  # noqa: G004

    def _collect_images(self):
        self._image_arrays: list = []
        self._image_ids: list = []
        for index in self._image_index:
            image_id, image_array = get_image(
                self._conn, self._well.getImage(index).getId()
            )
            flatfield_corrected_image = self._flatfield_correct_image(
                image_array
            )
            self._image_arrays.append(
                self._scale_images(flatfield_corrected_image)
            )
            self._image_ids.append(image_id)

    def _flatfield_correct_image(self, image_array):
        corrected_array = image_array / self._omero_data.flatfield_mask
        if len(corrected_array.shape) == 2:
            corrected_array = corrected_array[..., np.newaxis]
        logger.debug(f"Corrected image shape: {corrected_array.shape}")
        return corrected_array

    def _scale_images(self, corrected_array):
        scaled_channels = []
        logger.debug(f"omero_data.intensities: {self._omero_data.intensities}")
        for i in range(corrected_array.shape[-1]):
            scaled_channel = self._scale_img(
                corrected_array[..., i], self._omero_data.intensities[i]
            )
            scaled_channels.append(scaled_channel)
        return np.stack(scaled_channels, axis=-1)

    def _scale_img(self, img: np.array, intensities: tuple) -> np.array:
        """Increase contrast by scaling image to exclude lowest and highest intensities"""
        return exposure.rescale_intensity(img, in_range=intensities)
