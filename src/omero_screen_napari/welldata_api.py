"""
This module contains the classes and functions to parse the plate, well, image and label data from the Omero Plate
and collect them in the omero_data data clasee to be used by the napari viewer.
"""

import logging
import random
import re
import traceback
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import omero
import polars as pl
from ezomero import get_image  # type: ignore
from omero.gateway import (
    BlitzGateway,
    FileAnnotationWrapper,
    ImageWrapper,
    MapAnnotationWrapper,
    PlateWrapper,
    WellWrapper,
    _OriginalFileWrapper,
)
from qtpy.QtWidgets import QMessageBox
from tqdm import tqdm
import skimage.transform as transform
import scipy.ndimage
from skimage.util import map_array

from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import (
    omero_data,
    reset_omero_data,
)
from omero_screen_napari.utils import omero_connect, correct_channel_order

logger = logging.getLogger("omero-screen-napari.welldata_api")


get_image: Callable = get_image  # allows type hints for get_image


@omero_connect
def parse_omero_data(
    omero_data: OmeroData,
    plate_id: str,
    well_pos: str,
    image_input: str,
    time: str = "All",
    conn: Optional[BlitzGateway] = None,
) -> None:
    """
    This functions controls the flow of the data
    via the seperate parser classes to the omero_data class that stores omero plate metadata
    to be supplied to the napari viewer.
    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        plate_id (int): Omero Screen Plate ID number provided by user via welldata_widget
        well_list (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        image_input (str): String Describing the combinations of images to be supplied for each well
        The images options are 'All'; 0-3; 1, 3, 4.
        conn (BlitzGateway): BlitzGateway connection.
        time (str): String describing the time frames. Options are 'All'; 0-3 (range); 0 (single).
    """
    plate_number = int(plate_id)
    if conn is not None:
        try:
            parse_plate_data(
                omero_data, plate_number, well_pos, image_input, conn, time=time
            )
            # clear well and image data to avoid appending to existing data
            omero_data.reset_well_and_image_data()
            for well_pos in omero_data.well_pos_list:
                logger.info(f"Processing well {well_pos}")  # noqa: G004
                well_image_parser(omero_data, well_pos, conn, time=time)
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Error parsing plate data: {e}\n{''.join(traceback.format_exception(None, e, e.__traceback__))}"
            )
            # Show a message box with the error message
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(str(e))
            msg_box.setWindowTitle("Error")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()


def parse_plate_data(
    omero_data: OmeroData,
    plate_id: int,
    well_pos: str,
    image_input: str,
    conn: BlitzGateway,
    time: str = "All",
) -> None:
    """
    Function that combines the different parser classes that are responsible to process userinput
    and plate data and collect data to populate the plate related fields in the omero_data data class.

    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        plate_id (int): Omero Screen Plate ID number provided by user via welldata_widget
        well_pos (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        image_input (str):  String Describing the combinations of images to be supplied for each well
                            The images options are 'All'; 0-3; 1, 3, 4.
        conn (BlitzGateway): BlitzGateway connection.
        time (str): String describing the time frames. Options are 'All'; 0-3 (range); 0 (single).
    """
    # check if plate-ID already supplied, if it is then skip PlateHandler
    if plate_id != omero_data.plate_id:
        reset_omero_data()
        omero_data.plate_id = plate_id
        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn, time=time
        )
        user_input.parse_data()
        logger.info(f"Loaded data for plate with ID {plate_id}")
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
        logger.info(
            f"Plate data for plate {plate_id} already exists. Skip relaod."
        )
        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn, time=time
        )
        user_input.parse_data()


def well_image_parser(
    omero_data: OmeroData, well_pos: str, conn: BlitzGateway, time: str = 'All'
):
    """
    Function that combines the different parser classes that are responsible to process well, image and label data.
    The data is collected to populate the well/image related fields in the omero_data data class.

    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        well_pos (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        conn (BlitzGateway): BlitzGateway connection.
        time (str): String describing the time frames. Options are 'All'; 0-3 (range); 0 (single).
    """
    well_data_parser = WellDataParser(omero_data, well_pos)
    well_data_parser.parse_well()
    if well := well_data_parser._well:
        # TODO: pass time here
        image_parser = ImageParser(omero_data, well, conn)
        image_parser.parse_images_and_labels()


# -----------------------------------------------USER INPUT -----------------------------------------------------
class UserInput:
    """
    Class to handle user input for plate data retrieval from Omero.
    The class checks if the plate exists in Omero, parses the image number, well positions and image index
    and stores them in the omero_data class.
    public methods:
    parse_data: Orchestrate the data parsing process.
    private methods:
    _check_plate_id: Verifies that plate_id input points to an existing plate in Omero.
    _parse_image_number: Infers the number of images per well in the first well of the plate.
    _well_data_parser: Parses the well data input string and checks if the well position inputs are valid.
    _image_index_parser: Parses the image index input string and checks if the image index inputs are valid.
    """

    def __init__(
        self,
        omero_data: OmeroData,
        plate_id: int,
        well_pos: str,
        image_input: str,
        conn,
        time: str = "All"
    ):
        self._omero_data: OmeroData = omero_data
        self._plate_id: int = plate_id
        self._well_pos: str = well_pos
        self._images: str = image_input
        self._conn: BlitzGateway = conn
        self._plate: Optional[PlateWrapper] = None  # added by _check_plate_id
        self._image_number: int = 0  # added by _parse_image_number
        self._well_pos_list: list[str] = []  # added by _well_data_parser
        self._time: str = time
        self._start: np.array = None
        self._length: np.array = None

    def parse_data(self):
        self._check_plate_id()
        self._load_dataset()
        self._parse_image_number()
        self._well_data_parser()
        self._omero_data.well_pos_list = self._well_pos_list
        self._image_index_parser()
        self._omero_data.image_index = self._image_index
        self._image_time_parser()

    def _check_plate_id(self) -> None:
        """
        Verifies that plate_id input points to an existing plate in Omero.
        Raises:
            ValueError: If the plate does not exist in Omero.
        """
        self._plate = self._conn.getObject("Plate", self._plate_id)
        if not self._plate:
            logger.error(f"Plate with ID {self._plate_id} does not exist.")
            raise ValueError(f"Plate with ID {self._plate_id} does not exist.")
        else:
            self._omero_data.plate = self._plate

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
        if not project:
            logger.error(
                f"Project for screen with ID {self._omero_data.project_id} not found."
            )
            raise ValueError(
                f"Project for screen with ID {self._omero_data.project_id} not found."
            )
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

    def _parse_image_number(self):
        """
        Infers the number of images per well in the first well of the plate.
        This presumes that each well has the same number of images.
        Raises:
            ValueError: If no plate is found.
        """
        if self._plate:
            first_well = list(self._plate.listChildren())[0]
            self._image_number = len(list(first_well.listChildren()))
        else:
            logger.error("No plate found, unable to parse image number.")
            raise ValueError("No plate found, unable to parse image number.")

    def _well_data_parser(self):
        """
        Parses the well data input string and checks if the well position inputs are valid.
        Raises:
            ValueError: If the well input format is invalid.
        """
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
        """
        Parses the image index input string and checks if the image index inputs are valid.
        Raises:
            ValueError: If the image index input format is invalid.
        """
        self._omero_data.image_input = self._images # store this for use in saving training data
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

    def _image_time_parser(self):
        """
        Parses the image time string and checks if the image crop inputs are valid.
        Raises:
            ValueError: If the image time input format is invalid.
        """
        time = self._time

        if not (
            time.lower() == "all"
            or re.match(r"^\d+(-\d+)?$", time)
        ):
            logger.error(
                f"Image time '{time}' doesn't match any of the expected patterns 'All, 1-3, 1'."
            )
            raise ValueError(
                f"Image time '{time}' doesn't match any of the expected patterns 'All, 1-3, 1'."
            )

        if time.lower() == "all":
            # Ignore and default to all time points
            return

        if "-" in time:
            # Handle range, e.g., '1-3'
            start, end = map(int, time.split("-"))
        else:
            # Handle single number, e.g., '1'
            start = int(time)
            end = start

        if end < start:
            logger.error(f"Invalid time range: {start}-{end}.")
            raise ValueError(f"Invalid time range: {start}-{end}.")

        # Get image dimensions
        if self._plate:
            first_well = list(self._plate.listChildren())[0]
            image = first_well.getImage(0)
            xyzct = [image.getSizeX(), image.getSizeY(), image.getSizeZ(), image.getSizeC(), image.getSizeT()]
        else:
            logger.error("No plate found, unable to parse image time.")
            raise ValueError("No plate found, unable to parse image time.")
        # Validate. Change start to zero-based indexing.
        if end > xyzct[0]:
            logger.error(f"Invalid end time: {end} > {xyzct[0]}.")
            raise ValueError(f"Invalid end time: {end} > {xyzct[0]}.")
        if start < 1:
            logger.error(f"Invalid start time: {start} < 1.")
            raise ValueError(f"Invalid start time: {start} < 1.")

        start -= 1
        length = end - start
        # XYZCT format
        self._omero_data.crop_start = (0, 0, 0, 0, start)
        self._omero_data.crop_length = (xyzct[0], xyzct[1], xyzct[2], xyzct[3], length)


# -----------------------------------------------PLATE DATA -----------------------------------------------------

# classes to parse the plate data from the Omero Plate: csv file, channel data, flatfield mask, pixel size,
# image intensity and pixels size

# -----------------------------------------------CSV FILE -----------------------------------------------------


class CsvFileParser:
    """
    Class to handle csv file retrieval and processing from Omero Plate
    public methods:
    parse_csv: Orchestrate the csv file handling. Check if csv file is available, if not download it.
    and make it available to the omero_data class as a polars LazyFrame object.
    private methods:
    _csv_available: checks if any csv file in the target directory contains the string self.plate_id in its name.
    _get_csv_file: gets the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
    _download_csv: downloads the csv file from the Omero Plate. If _cc.csv the file name will be
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_id: int = omero_data.plate_id
        self._plate = omero_data.plate
        self._data_path: Path = omero_data.data_path
        self._csv_file_path: Optional[Path] = None
        self._file_name: Optional[str] = None
        self._original_file: Optional[_OriginalFileWrapper] = None

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
        if self._csv_file_path:
            omero_data.plate_data = pl.scan_csv(self._csv_file_path)
            self._omero_data.csv_path = self._csv_file_path
        else:
            logger.error("No csv file assigned by CsvParser.")
            raise ValueError("No csv file by CsvParser")

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
                if name and name.endswith("_cc.csv"):
                    self._file_name = name
                    self._original_file = ann.getFile()
                    break  # Prioritize and stop searching if _cc.csv is found
                elif (
                    name
                    and name.endswith("final_data.csv")
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
        if self._file_name and self._original_file:
            saved_name = f"{self._plate_id}_{self._file_name.split('_')[-1]}"
            self._csv_file_path = self._data_path / saved_name
            with open(self._csv_file_path, "wb") as file_on_disk:
                for chunk in self._original_file.asFileObj():
                    file_on_disk.write(chunk)
        else:
            logger.error("Problem with parsing csv file.")
            raise ValueError("Problem with parsing csv file.")


# -----------------------------------------------CHANNEL DATA -----------------------------------------------------


class ChannelDataParser:
    """
    Class to handle channel data retrieval and processing from Omero Plate.
    public method:
    parse_channel_data: Coordinates the retrieval of channel data from Omero Plate
    private methods:
    _get_map_ann: Get channel information from Omero map annotations or raise exception if no appropriate annotation is found.
    _tidy_up_channel_data: Helper method that processes the channel data to a dictionary
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data: OmeroData = omero_data
        self._plate: PlateWrapper = omero_data.plate
        self._plate_id: int = omero_data.plate_id

    def parse_channel_data(self):
        """Process channel data to dictionary {channel_name: channel_index}
        and add it to viewer_data; raise exception if no channel data found."""
        self._get_map_ann()
        self._tidy_up_channel_data()
        self._omero_data.channel_data = self._channel_data
        logger.debug(
            f"Channel data: {self._channel_data} loaded to omero_data.channel_data"
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
                    self._map_annotations = ann_value
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
        Convert channel map annotation from list of tuples to dictionary.
        Eliminate spaces and swap Hoechst to DAPI if necessary.
        """
        channel_data = dict(self._map_annotations)
        sorted_channel_data = dict(
            sorted(channel_data.items(), key=lambda item: item[1])
        )
        self._channel_data = {
            key.strip(): value for key, value in sorted_channel_data.items()
        }
        if "Hoechst" in self._channel_data:
            self._channel_data["DAPI"] = self._channel_data.pop("Hoechst")
        # check if one of the channels is DAPI otherwise raise exception


# -----------------------------------------------FLATFIELD CORRECTION-----------------------------------------------------


class FlatfieldMaskParser:
    """
    Class to handle flatfield mask retrieval and processing from Omero Plate.
    public method:
    parse_flatfieldmask: Coordinates the retrieval of flatfield mask from Omero Plate
    private methods:
    _load_dataset: Looks for a data set that matches the plate name in the screen project
    _get_flatfieldmask: Gets flatfieldmasks from project linked to screen
    _get_map_ann: Get channel information from Omero map annotations or raise exception if no appropriate annotation is found.
    _check_flatfieldmask: Checks if the mappings with the plate channel date are consistent.
    """

    def __init__(self, omero_data: OmeroData, conn: BlitzGateway):
        self._omero_data: OmeroData = omero_data
        self._conn: BlitzGateway = conn

    def parse_flatfieldmask(self):
        """
        Coordinates the retrieval of flatfield mask from Omero Plate
        """
        self._get_flatfieldmask()
        self._get_map_ann()
        self._check_flatfieldmask()
        omero_data.flatfield_masks = self._flatfield_array

    def _get_flatfieldmask(self):
        """Gets flatfieldmasks from project linked to screen"""

        flatfield_mask_name = f"{self._omero_data.plate_id}_flatfield_masks"
        logger.debug(f"Flatfield mask name: {flatfield_mask_name}")  # noqa: G004
        flatfield_mask_found = False
        for image in self._omero_data.screen_dataset.listChildren():
            image_name = image.getName()
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

        logger.debug(
            f"Flatfield channels: {self._flatfield_channels.values()}, channel data {omero_data.channel_data.keys()}"  # noqa: G004
        )
        channel_check = True
        for key, value in self._flatfield_channels.items():
            # Extract the number from key, e.g., 'channel_1' -> '1'
            channel_number = key.split("_")[-1]

            # Find the channel name that corresponds to this number in channels
            expected_name = None
            for name, num in self._omero_data.channel_data.items():
                if num == channel_number:
                    expected_name = name
                    break
            # Check if the expected name matches the name in flatfield_channels
            if expected_name != value:
                channel_check = False
                break

        if not channel_check:
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
    public method:
    parse_intensities: Parse the intensities from the plate data and store them in the omero_data object.
    private methods:
    _set_keyword: Set the keyword based on the presence of "_cell" or "_nucleus" in the dataframe columns.
    _get_values: Filter the dataframe to only include columns with the channel keyword.
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_data: pl.LazyFrame = omero_data.plate_data
        self._keyword: Optional[str] = None
        self._intensities: Optional[dict[str, Tuple[int, int]]] = None

    def parse_intensities(self):
        """
        Parse the intensities from the plate data and store them in the omero_data object.
        Raises:
            ValueError: when intesities cannot be collected from the plate csv data.
        """
        self._set_keyword()
        self._get_values()
        if self._intensities:
            self._omero_data.intensities = self._intensities
        else:
            logger.error("Problem with loading intensities to scale channels.")
            raise ValueError(
                "Problem with loading intensities to scale channels."
            )

    def _set_keyword(self):
        """
        Set the keyword based on the presence of "_cell" or "_nucleus" in the dataframe columns.
        Raises:
            ValueError: when the dataframe does not contain the channel keywords.
            ValueError: when neither "_cell" nor "_nucleus" is present in the dataframe columns.
        """
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
        Filter the dataframe to only include columns with the channel keyword.
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
        self._random_wells: Optional[list[WellWrapper]] = None
        self._random_images: Optional[list[ImageWrapper]] = None
        self._pixel_size: Optional[Tuple[int]] = None

    def parse_pixel_size_values(self):
        self._check_wells_and_images()
        self._check_pixel_values()
        if self._pixel_size:
            self._omero_data.pixel_size = self._pixel_size
        else:
            logger.error("Problem with loading pixel size values.")
            raise ValueError("Problem with loading pixel size values.")

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
        if self._random_images:
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
        else:
            logger.error("No images found in the wells")
            raise ValueError("No images found in the wells")


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
        self._well: Optional[WellWrapper] = None
        self._well_id: Optional[int] = None
        self._metadata: Optional[dict] = None
        self._well_ifdata: Optional[pl.DataFrame] = None

    def parse_well(self):
        self._parse_well_object()
        if self._well and self._well_id:
            self._omero_data.well_list.append(self._well)
            self._omero_data.well_id_list.append(int(self._well_id))
        else:
            logger.error(f"Problem with loading well {self._well_pos} data.")
            raise ValueError(
                f"Problem with loading well {self._well_pos} data."
            )
        self._get_well_metadata()
        if self._metadata:
            self._omero_data.well_metadata_list.append(self._metadata)
        else:
            logger.error(f"No metadata found for well {self._well_pos}.")
            raise ValueError(f"No metadata found for well {self._well_pos}.")
        self._load_well_csvdata()
        if self._well_ifdata is not None:
            self._omero_data.well_ifdata = pl.concat(
                [self._omero_data.well_ifdata, self._well_ifdata]
            )
        else:
            logger.error(f"No well csv data found for well {self._well_pos}.")
            raise ValueError(
                f"No well csv data found for well {self._well_pos}."
            )

    def _parse_well_object(self):
        """
        Check if the well object exists in the plate and retrieve it.
        Raises:
            ValueError: _description_
        """
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
        """
        Get metadata for the well from the map annotations.
        The map annotations have to include a dictionary with a key "cell_line" to be valid.
        Raises:
            ValueError: If the well object is not found.
        """
        map_ann = None
        if not self._well:
            raise ValueError(f"Well at pos {self._well_pos} not found")
        for ann in self._well.listAnnotations():
            if ann and ann.getValue():
                map_ann = dict(ann.getValue())
                if "cell_line" in map_ann:
                    self._metadata = map_ann
                    break
                else:
                    logger.error(
                        f"No metadata with cell line name found for well {self._well_pos}"
                    )
                    raise ValueError(
                        f"No metadata with cell line name found for well {self._well_pos}"
                    )
        if not self._metadata:
            logger.error(f"No map annotations found for well {self._well_pos}")
            raise ValueError(
                f"No map annotations found for well {self._well_pos}"
            )

    def _load_well_csvdata(self):
        """
        Load the csv data for the well using the polars lazyframe to
        filter out the required data for the well.
        """
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
        well: WellWrapper,
        conn: BlitzGateway,
    ):
        self._omero_data: OmeroData = omero_data
        self._well: WellWrapper = well
        self._conn: BlitzGateway = conn
        self._image_index: list[int] = self._omero_data.image_index
        self._images: Optional[np.ndarray] = None
        self._image_ids: list[int] = []
        self._image_arrays: list[np.ndarray] = []
        self._label_arrays: list[np.ndarray] = []

    def parse_images_and_labels(self):
        self._parse_images()
        self._parse_labels()
        logger.info("Images and labels added to omero_data")

    def _parse_images(self):
        """
        Collects images for the well and adds them to the omero_data object.
        If the omero_data object already has images, the new images are concatenated to the existing images.
        Otherwise, the new images are added to the omero_data object.
        """
        self._collect_images()

        # Determine the axes to squeeze based on the number of timepoints
        squeeze_axes = (2,) if self._image_arrays[0].shape[0] > 1 else (1, 2)

        # Stack and squeeze the new images
        new_images = np.squeeze(
            np.stack(self._image_arrays, axis=0), axis=squeeze_axes
        )
        if self._omero_data.images.shape == (0,):
            self._omero_data.images = new_images
            self._omero_data.image_ids = self._image_ids
            logger.debug(
                f"Images loaded to empty omero_data.images, new shape: {self._omero_data.images.shape}"
            )
        else:
            self._omero_data.images = np.concatenate(
                (self._omero_data.images, new_images), axis=0
            )
            self._omero_data.image_ids.extend(self._image_ids)

        logger.debug(
            f"Images shape after processing: {self._omero_data.images.shape}"
        )

    def _parse_labels(self):
        """
        Collects labels for the well and adds them to the omero_data object.
        If the omero_data object already has labels, the new labels are concatenated to the existing labels.
        Otherwise, the new labels are added to the omero_data object.
        """
        self._collect_labels()
        if self._omero_data.labels.size == 0:
            logger.debug(f"Labels shape: {self._label_arrays[0].shape}")
            self._omero_data.labels = np.stack(self._label_arrays, axis=0)

            logger.debug(
                f"Labels loaded to empty omero_data.labels, new shape: {omero_data.labels.shape}"
            )
        else:
            self._omero_data.labels = np.concatenate(
                (self._omero_data.labels, self._label_arrays), axis=0
            )
            logger.debug(
                f"Labels added to omero_data.labels, new shape: {omero_data.labels.shape}"
            )

    def _collect_images(self):
        """
        Collects images and image arrays for the well and adds the to the image_ids and image_arrays list.
        """

        logger.info(f"Collecting images for well {self._well.getWellPos()}")
        start, length = _get_crop(self._omero_data)
        if start:
            logger.info(f"Using crop: {start} - {length}")

        for index in tqdm(self._image_index):
            if image := self._well.getImage(index):
                # Support time point...
                logger.info(f"Image {image.getId()}:tzyxc={image.getSizeT()},{image.getSizeZ()},{image.getSizeY()},{image.getSizeX()},{image.getSizeC()}")

                if mip_id := self.check_mip(image):
                    logger.info(
                        f"Image with index {index} has Z-stacks, looking for MIP with id {mip_id}."
                    )
                    _, image_array = get_image(self._conn, int(mip_id), start_coords=start, axis_lengths=length)
                else:
                    _, image_array = get_image(self._conn, image.getId(), start_coords=start, axis_lengths=length)
                flatfield_corrected_image = self._flatfield_correct_image(
                    image_array
                )
                logger.debug(f"Image shape: {flatfield_corrected_image.shape}")
                self._image_arrays.append(flatfield_corrected_image)
                self._image_ids.append(image.getId())
            else:
                logger.error(
                    f"Image with index {index} not found in well {self._well.getWellPos()}"
                )
                raise ValueError(
                    f"Image with index {index} not found in well {self._well.getWellPos()}"
                )

    def check_mip(self, image):
        if map_anns := image.listAnnotations(
            ns=omero.constants.metadata.NSCLIENTMAPANNOTATION
        ):
            for ann in map_anns:
                ann_values = dict(ann.getValue())
                for item in ann_values.values():
                    print(item)
                    if "mip" in item:
                        return item.split("_")[-1]
        if (
            image.getSizeZ() > 1
        ):  # check if image has z-stacks even though no mip is assigned
            raise ValueError(
                f"Image with index {self._image_index} has Z-stacks but no MIP assigned."
            )
        return None

    def _flatfield_correct_image(self, image_array):
        """
        Corrects the image array using the flatfield mask for the well.

        Args:
            image_array (np.ndarray): Image array to be corrected.

        Returns:
           np.ndarray: Corrected image array.
        """
        corrected_array = image_array / self._omero_data.flatfield_masks
        if len(corrected_array.shape) == 2:
            corrected_array = corrected_array[..., np.newaxis]
        logger.debug(f"Corrected image shape: {corrected_array.shape}")
        return corrected_array

    def _check_label_data(self):
        label_names = [
            int(image.getName().split("_")[0])
            for image in self._omero_data.screen_dataset.listChildren()
        ]
        logger.debug(f"Label names: {label_names}")
        all_images_in_labels = all(
            name in label_names for name in self._image_ids
        )
        if not all_images_in_labels:
            logger.error("Available label images do not match loaded images")
            raise ValueError(
                "Available label images do not match loaded images"
            )
        else:
            logger.debug("All label images found")

    def _collect_labels(self):
        label_names = [f"{name}_segmentation" for name in self._image_ids]
        start, length = _get_crop(self._omero_data)

        relevant_label_data = [
            label_data
            for label_data in self._omero_data.screen_dataset.listChildren()
            if any(
                label_name in label_data.getName()
                for label_name in label_names
            )
        ]

        for label_data in relevant_label_data:
            _, label_array = get_image(self._conn, label_data.getId(), start_coords=start, axis_lengths=length)
            if label_array.shape[-1] == 2:
                corrected_label_array = correct_channel_order(label_array)
                self._label_arrays.append(corrected_label_array.squeeze())
            else:
                # Create a tuple of axes to squeeze, excluding the last axis
                shape = label_array.shape
                axes_to_squeeze = tuple(i for i in range(len(shape) - 1) if shape[i] == 1)
                self._label_arrays.append(np.squeeze(label_array, axis=axes_to_squeeze))


# -----------------------------------------------Stitching-----------------------------------------------------

# class Stitcher:
#     def __init__(
#         self,
#         omero_data: OmeroData,
#         conn: BlitzGateway,
#     ):
#         self._omero_data: OmeroData = omero_data




def stitch_images(omero_data) -> np.ndarray:
    """Stitch the images in the array according to the specified pattern
    When a full well is imaged at 10 x21 single channel 10x images from an Operetta microscope
    are provided)
    returns: [np.ndarray] stitched image of shape (5*1080, 5*1080, 1)
    """
    logger.debug("Stitching images %s", omero_data.images.shape)
    assert omero_data.images.shape == (
        21,
        1080,
        1080,
        1,
    ), "The input array should be 21x1080x1080x1"


    # Creating an empty array to add as spacer
    empty_array = np.full((1080, 1080, 1), fill_value=omero_data.intensities[0][0])

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
                ] = omero_data.images[idx]
    logger.debug("Stitched image shape: %s", stitched_image.shape)
    return stitched_image

def stitch_images2(omero_data, rotation=0.0, overlap_x=0, overlap_y=0, edge=0, mode='reflect') -> np.ndarray:
    """Stitch the images in the array according to the specified pattern
    when a full well is imaged at 10x on an Operetta microscope. This is a 5x5 grid
    with corners excluded which creates 21 images.
    returns: [np.ndarray] stitched image
    """
    logger.debug("Stitching images %s", omero_data.images.shape)
    # NYXC
    assert len(omero_data.images.shape) == 4, "The input array should be 4D: N-images of YXC"
    assert omero_data.images.shape[0] == 21, "The input number of images should be size 21"

    # YX order
    indices_pattern = [
        [-1, 1, 2, 3, -1],  # Preserved -1 for empty
        [8, 7, 6, 5, 4],  # Adjusted for zero-based indexing
        [9, 10, 0, 11, 12],  # The first image is now 0 (zero-based)
        [17, 16, 15, 14, 13],  # Adjusted for zero-based indexing
        [-1, 18, 19, 20, -1],  # Preserved -1 for empty
    ]

    tiles = {}
    for y, row in enumerate(indices_pattern):
      for x, idx in enumerate(row):
        if idx != -1:
          d = tiles.get(x)
          if not d:
            tiles[x] = d = dict()
          d[y] = omero_data.images[idx]

    stitched_image = compose_tiles(tiles, rotation=rotation, ox=-overlap_x, oy=-overlap_y,
      edge=edge, mode=mode)

    logger.debug("Stitched image shape: %s", stitched_image.shape)
    return stitched_image

def compose_tiles(
  tiles: dict[int, dict[int, np.array]],
  rotation: float = 0,
  ox: int = 0,
  oy: int = 0,
  edge: int = 0,
  mode: str = 'reflect'):
  """
  Compose tiles into a single image. It is assumed all tiles are the same shape: YXC.
  Args:
      tiles (dict): Dictionary of dictionaries of np.array tiles, keyed by [x][y].
      rotation (float): Rotation angle (degrees in counter-clockwise direction).
      ox (int): Tile offset in x (use negative for overlap).
      oy (int): Tile offset in y (use negative for overlap).
      edge (int): Edge size for blending overlaps.
      mode (str): Mode used to fill the rotated image outside the bounds
      (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’).
  Returns
      composed (np.array): The composed image (YXC).
  """
  # Compute tile grid dimensions
  maxx = np.max(list(tiles.keys()))
  maxy = 0
  for x in tiles.values():
    maxy = np.max(list(x.keys()), initial=maxy)

  # Create rotated mask using the first image to set the dimensions.
  # The mask uses nearest-neighbour interpolation to effectivly mark the pixels
  # of interest.
  y = next(iter(tiles[maxx]))
  os = tiles[maxx][y].shape
  m = np.ones(os[0:2], dtype=int)
  m = transform.rotate(m, rotation, resize=True, preserve_range=True, order=0)
  ns = m.shape

  # Create weights for blending overlap.
  if edge:
    # Distance transform does not use out-of-bounds as background.
    # So pad with 1 pixel and crop.
    d = scipy.ndimage.distance_transform_edt(np.pad(m, 1))
    d = d[1:-1, 1:-1]
    d = np.clip(d, a_min=0, a_max=edge)
    m = d / edge

  # Create output.
  # Note that arrays are YXC format.
  channels = os[2]
  out = np.zeros(((maxy+1) * ns[0] + maxy * oy, (maxx+1) * ns[1] + maxx * ox, channels))
  sum = np.zeros(out.shape[0:2])

  # Rotate each image and insert
  for x, d in tiles.items():
    for y, im in d.items():
      l = []
      for c in range(channels):
        # Multiply the rotation by the mask (which optionally weights pixels).
        # The rotation uses bilinear interpolation with edge-pixel extension to generate
        # reasonable intensity edge pixels. The mode can be varied.
        l.append(m * transform.rotate(im[...,c], rotation, resize=True, preserve_range=True, order=1,
          mode=mode))
      # Original shape sets the translation
      xp = x * (os[1] + ox)
      yp = y * (os[0] + oy)
      # New shape defines the range of the rotation image.
      # Note that arrays are YX format.
      out[yp : yp + ns[0], xp : xp + ns[1],:] += np.dstack(l)
      sum[yp : yp + ns[0], xp : xp + ns[1]] += m

  indices = sum != 0
  for c in range(channels):
    out[..., c] = np.divide(out[..., c], sum, where=indices, out=np.zeros(sum.shape))
  return out

def stitch_labels(omero_data, rotation=0.0, overlap_x=0, overlap_y=0) -> np.ndarray:
    """Stitch the labels in the array according to the specified pattern
    when a full well is imaged at 10x on an Operetta microscope. This is a 5x5 grid
    with corners excluded which creates 21 images.
    Note: labels will be renumberd to unique objects.
    returns: [np.ndarray] stitched labels
    """
    logger.debug("Stitching labels %s", omero_data.labels.shape)
    # NYXC
    assert len(omero_data.labels.shape) == 4, "The input array should be 4D: N-images of YXC"
    assert omero_data.labels.shape[0] == 21, "The input number of images should be size 21"

    # YX order
    indices_pattern = [
        [-1, 1, 2, 3, -1],  # Preserved -1 for empty
        [8, 7, 6, 5, 4],  # Adjusted for zero-based indexing
        [9, 10, 0, 11, 12],  # The first image is now 0 (zero-based)
        [17, 16, 15, 14, 13],  # Adjusted for zero-based indexing
        [-1, 18, 19, 20, -1],  # Preserved -1 for empty
    ]

    tiles = {}
    for y, row in enumerate(indices_pattern):
      for x, idx in enumerate(row):
        if idx != -1:
          d = tiles.get(x)
          if not d:
            tiles[x] = d = dict()
          d[y] = omero_data.labels[idx]
          np.save(f'lx{x}y{y}.npy', d[y])

    stitched_image = compose_labels(tiles, rotation=rotation, ox=-overlap_x, oy=-overlap_y)

    logger.debug("Stitched labels shape: %s", stitched_image.shape)
    return stitched_image

def compose_labels(
    tiles: dict[int, dict[int, np.array]],
    rotation: float = 0,
    ox: int = 0,
    oy: int = 0):
    """
    Compose labels tiles into a single image. It is assumed all tiles are the same shape: YXC.
    The unique ID of labels will be remapped. Overlapping labels on adjacent tiles are mapped
    to the same ID.
    Args:
        tiles (dict): Dictionary of dictionaries of np.array tiles, keyed by [x][y].
        rotation (float): Rotation angle (degrees in counter-clockwise direction).
        ox (int): Tile offset in x (use negative for overlap).
        oy (int): Tile offset in y (use negative for overlap).
    Returns
        composed (np.array): The composed labels (YXC).
    """
    # Compute tile grid dimensions
    maxx = np.max(list(tiles.keys()))
    maxy = 0
    for x in tiles.values():
      maxy = np.max(list(x.keys()), initial=maxy)

    # Rotate the first image to set the dimensions.
    y = next(iter(tiles[maxx]))
    os = tiles[maxx][y].shape
    ns = transform.rotate(np.ones(os[0:2], dtype=int), rotation, resize=True,
      preserve_range=True, order=0).shape

    # Create output.
    # Note that arrays are YXC format.
    channels = os[2]
    out = list(np.zeros(((maxy+1) * ns[0] + maxy * oy, (maxx+1) * ns[1] + maxx * ox),
      dtype=tiles[maxx][y].dtype) for i in range(channels))

    border = 0
    if ox < 0:
      border = -ox
    if oy < 0:
      border = max(border, -oy)

    # Rotate each image and insert
    for x, d in tiles.items():
      for y, im in d.items():
        l = []
        # Original shape sets the translation
        xp = x * (os[1] + ox)
        yp = y * (os[0] + oy)
        for c in range(channels):
          # The rotation uses nearest neighbour interpolation to maintain IDs.
          i = transform.rotate(im[...,c], rotation, resize=True, preserve_range=True, order=0)
          out[c] = merge_labels(out[c], i, xp=xp, yp=yp, border=border)

    return np.dstack(out)

def merge_labels(
    im1: np.array,
    im2: np.array,
    xp: int = 0,
    yp: int = 0,
    border: int = 0):
    """
    Merges the labels in image 2 into image 1. Image 2 may be smaller than image 1.
    Scans pixels in the border against the current labels. Any overlapping labels
    in the new image adopt the ID of the overlapping label.
    Args:
        im1 (np.array): Current labels.
        im2 (np.array): New labels.
        xp (int): Offset in x.
        yp (int): Offset in y.
        border (int): Border width.
    Returns
        updated (np.array): The updated labels.
    """
    s = im2.shape
    # Avoid overlap analysis when no border or all-zero current image
    if not (border and im1.any()):
      return _merge_nonoverlapping_labels(im1, im2, xp=xp, yp=yp)

    # Extract current sub-image
    im1a = im1[yp : yp + s[0], xp : xp + s[1]]
    # Overlap mask
    overlap = (im1a != 0) & (im2 != 0)
    if not overlap.any():
      return _merge_nonoverlapping_labels(im1, im2, xp=xp, yp=yp)

    # Count size of label overlaps in border
    h1o = np.bincount(im1a.reshape(-1), weights=overlap.reshape(-1))
    h2o = np.bincount(im2.reshape(-1), weights=overlap.reshape(-1))
    # Require a new -> old ID overlap histogram.
    # Assume new IDs are sequential from 1.
    # Remap old IDs that are in the overlap from 1 to save memory.
    map = np.zeros(len(h1o), dtype=np.uint16)
    rmap = np.zeros(len(h1o), dtype=np.uint16)
    id = 0
    for i, c in enumerate(h1o):
      if c:
        map[i] = id
        rmap[id] = i
        id += 1
    h = np.zeros((np.nonzero(h2o)[0][-1] + 1, id), dtype=np.uint16)
    for a, b in zip(im2.reshape(-1), im1a.reshape(-1)):
      if a and b:
        h[a][map[b]] += 1

    # Greedy assignment of overlaps based on intersect over size.
    # Count size of labels.
    h1 = np.bincount(im1.reshape(-1))
    h2 = np.bincount(im2.reshape(-1))
    # Convert overlaps to a list
    overlaps = []
    # i=im1 mapped value; j=im2 value
    for j, a in enumerate(h):
      for i, c in enumerate(a):
        if c:
          # i=im1 value
          i = rmap[i]
          # Compute max intersect over size
          f = c / max(h1[i], h2[j])
          overlaps.append((i, j, c, f))
    overlaps.sort(reverse=True, key=lambda x: x[-1])

    # Renumber the labels.
    # Initialise as mapping to themselves.
    # Note: We use the maximum ID in the current image to offset the new image.
    omap1 = np.arange(len(h1))
    omap2 = np.arange(len(h2))
    map1 = np.zeros(len(h1), dtype=np.uint16)
    map2 = np.zeros(len(h2), dtype=np.uint16)
    m1 = len(h1)

    # List of overlap pixels to remove from each image
    remove1 = []
    remove2 = []

    # Remap the labels to use the ID from the object it overlaps.
    for i, j, c, f in overlaps:
      f1 = c / h1[i]
      f2 = c / h2[j]
      # Either image could be the parent so make the largest overlap the child.
      # Assign the child to the parent ID. If the child if already assigned
      # it can be assumed that this is a smaller overlap of the child with some
      # other object. Remove the overlap child pixels.
      # If the parent is already assigned then assume a better child has already
      # overlapped the parent. Remove the overlap child pixels.
      # This works for a greedy algorithm.
      if f1 > f2:
        # current image is the child
        if map1[i]:
          remove1.append(i)
          continue # Already assigned
        if map2[j]:
          remove1.append(i)
          continue # Already assigned
        map2[j] = j + m1
        map1[i] = map2[j]
      else:
        # new image is the child
        if map2[j]:
          remove2.append(j)
          continue # Already assigned
        if map1[i]:
          remove2.append(j)
          continue # Already assigned
        map1[i] = i
        map2[j] = map1[i]

    # Remove overlaps
    if remove2:
      for v in remove2:
        im2[(im2==v) & overlap] = 0
    if remove1:
      for v in remove1:
        im1a[(im1a==v) & overlap] = 0
      im1[yp : yp + s[0], xp : xp + s[1]] = im1a

    # Remap the new image to unique IDs (if not mapped)
    map1 = np.where(map1 == 0, omap1, map1)
    map2 = np.where(map2 == 0, omap2 + m1, map2)
    map2[0] = 0

    # Compress IDs to ascending from 1
    u = set(map1)
    u.update(map2)
    u.add(0) # Ensure zero is added so first mapped ID is 1
    m = np.zeros(max(u) + 1, dtype=np.uint16)
    for i, v in enumerate(sorted(u)):
      m[v] = i
    for i, v in enumerate(map1):
      map1[i] = m[v]
    for i, v in enumerate(map2):
      map2[i] = m[v]

    # Remap the images
    map_array(im1, omap1, map1, out=im1)
    map_array(im2, omap2, map2, out=im2)

    # Add the remapped labels using a binary OR. Overlapping pixels have been handled
    # to match one of the parent IDs (or removed).
    im1[yp : yp + s[0], xp : xp + s[1]] |= im2

    return im1

def _merge_nonoverlapping_labels(
    im1: np.array,
    im2: np.array,
    xp: int = 0,
    yp: int = 0,
    m1: int = 0):
    """
    Merges the labels in image 2 into image 1. Image 2 may be smaller than image 1.
    Args:
        im1 (np.array): Current labels.
        im2 (np.array): New labels.
        xp (int): Offset in x.
        yp (int): Offset in y.
        m1 (int): Maximum label in current.
    Returns
        updated (np.array): The updated labels.
    """
    s = im2.shape
    if not m1:
      m1 = np.max(im1)
    # Remap to unique IDs.
    # Simply add the previous max to the IDs and update the max.
    # This does not compress IDs as it is assumed both inputs
    # have ascending IDs from 1.
    np.add(im2, m1, where=im2!=0, out=im2)
    im1[yp : yp + s[0], xp : xp + s[1]] += im2

    return im1

def _get_crop(omero_data: OmeroData):
    if omero_data.crop_start:
        return omero_data.crop_start, omero_data.crop_length
    return None, None
