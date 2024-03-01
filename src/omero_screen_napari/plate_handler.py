import logging
from pathlib import Path
from typing import List, Tuple

import napari
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


@omero_connect
def retrieve_data(
    omero_data,
    plate_id: int,
    conn: BlitzGateway = None,
) -> None:
    """
    This functions controls the flow of the data
    via the seperate handler classes to the omero_data class that stores image data and metadata
    to be supplied to the viewer.
    This function is passed to welldata_widget and supplied by parameters from the
    plugin gui.
    Args:
        plate_id (int): Omero Screen Plate ID number provided by welldata_widget
        well_list (List[str]): List of wells provided by welldata_widget (e.g ['A1', 'A2'])
        images (str): String Describing the combinations of images to be supplied for each well
        The images options are 'All'; 0-3; 1, 3, 4. This will be processed by the ImageHandler to retrieve the relevant images
    """
    # check if plate-ID already supplied, if it is then skip PlateHandler
    if plate_id != omero_data.plate_id:
        reset_omero_data()
        logger.info(f"Retrieving new plate data for plate {plate_id}")  # noqa: G004
        omero_data.plate_id = plate_id
        plate = conn.getObject("Plate", plate_id)
        csv_manager = CsvFileManager(omero_data, plate)
        csv_manager.handle_csv()
        channel_manager = ChannelDataManager(omero_data, plate)
        channel_manager.get_channel_data()




#-----------------------------------------------CSV FILE HANDLING-----------------------------------------------------
class CsvFileManager:
    """
    Class to handle csv file retrieval and processing from Omero Plate
    Class methods:
    handle_csv: Orchestrate the csv file handling. Check if csv file is available, if not download it.
    and make it available to the omero_data class as a polars LazyFrame object.
    _csv_available: Helper; checks if any csv file in the target directory contains the string self.plate_id in its name.
    _get_csv_file: Helper; gets the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
    _download_csv: Helper; downloads the csv file from the Omero Plate. If _cc.csv the file name will be

    """

    def __init__(
        self, omero_data: OmeroData, plate: omero.gateway.PlateWrapper
    ):
        self.omero_data = omero_data
        self.plate_id = omero_data.plate_id
        logger.debug(f"Plate ID is: {self.plate_id}")  # noqa: G004
        self.plate = plate

    def handle_csv(self):
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
        omero_data.plate_data = pl.scan_csv(self.omero_data.csv_path)

    # helper functions for csv handling

    def _csv_available(self):
        """
        Check if any csv file in the directory contains the string self.plate_id in its name.
        """
        # TODO get filepath and generate attribute for platehandler class if file exists
        for file in self.csv_path.iterdir():
            if file.is_file() and str(self.plate_id) in file.name:
                self.omero_data.csv_path = file
                return True

    def _get_csv_file(self):
        """
        Get the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
        """
        self.original_file = None

        file_anns = self.plate.listAnnotations()
        for ann in file_anns:
            if isinstance(ann, FileAnnotationWrapper):
                self.file_name = ann.getFile().getName()
                if self.file_name.endswith("_cc.csv"):
                    self.original_file = ann.getFile()
                    break  # Prioritize and stop searching if _cc.csv is found
                elif (
                    self.file_name.endswith("final_data.csv")
                    and self.original_file is None
                ):
                    self.original_file = ann.getFile()
                    # Don't break to continue searching for a _cc.csv file

        if self.original_file is None:
            logger.error("No suitable csv file found for the plate.")
            raise ValueError("No suitable csv file found for the plate.")

    def _download_csv(self):
        """
        Download the csv file from the Omero Plate. If _cc.csv the file name will be
        {plate_id}_cc.csv, otherwise {plate_id}_data.csv
        """
        saved_name = f"{self.plate_id}_{self.file_name.split('_')[-1]}"
        logger.info(f"Downloading csv file {self.file_name} to {saved_name}")  # noqa: G004
        self.omero_data.csv_path = self.csv_path / saved_name
        file_content = self.original_file.getFileInChunks()
        with open(self.omero_data.csv_path, "wb") as file_on_disk:
            for chunk in file_content:
                file_on_disk.write(chunk)

#-----------------------------------------------CHANNEL DATA HANDLING-----------------------------------------------------

class ChannelDataManager:
    """
    Class to handle channel data retrieval and processing from Omero Plate.
    class methods:
    _get_channel_data: Coordinates the retrieval of channel data from Omero Plate
    _get_map_ann: Hhlper method that scans plate map annotations for channel data
    and raises an exceptions if no suitable annotation is found
    _tidy_up_channel_data: Helper method that processes the channel data to a dictionary
    """

    def __init__(self, omero_data: OmeroData, plate: PlateWrapper):
        self.omero_data = omero_data
        self.plate = plate
        self.plate_id: int = omero_data.plate_id

    def get_channel_data(self):
        """Process channel data to dictionary {channel_name: channel_index}
        and add it to viewer_data; raise exception if no channel data found."""
        self._get_map_ann()
        self._tidy_up_channel_data()
        self.omero_data.channel_data = self.channel_data
        logger.info(f"Channel data: {self.channel_data} loaded to omero_data.channel_data")  # noqa: G004

    def _get_map_ann(self):
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """

        annotations = self.plate.listAnnotations()

        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]

        if not map_annotations:
            logger.error(f"No MapAnnotations found for plate {self.plate_id}.")  # noqa: G004
            raise ValueError(
                f"No MapAnnotations found for plate {self.plate_id}."
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
            f"No DAPI or Hoechst channel information found for plate {self.plate_id}."
        )  # noqa: G004
        raise ValueError(
            f"No DAPI or Hoechst channel information found for platplate {self.omero_data.plate_id}."
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

#-----------------------------------------------Flatfield Correction Mask HANDLING-----------------------------------------------------

class FlatfieldMaskManager:

    def __init__(self, omero_data: OmeroData, conn: BlitzGateway):
        self.omero_data = omero_data
        self.conn = conn
    def get_flatefieldmask(self):
        #TODO set up and integration test
        pass
    
    def _load_dataset(self):
        project = self.conn.getObject("Project", self.omero_data.project_id)
        if dataset := self.conn.getObject(
        "Dataset",
        attributes={"name": str(self.omero_data.plate_id)},
        opts={"project": project.getId()},
        ):
            self.omero_data.screen_dataset = dataset
        else:
            logger.error(f"The plate {omero_data.plate_name} has not been assigned a dataset.")  # noqa: G004
            raise ValueError(f"The plate {omero_data.plate_name} has not been assigned a dataset.")

#TODO unit test for _get_map_ann
    def _get_flatfieldmask(self):
        """Gets flatfieldmasks from project linked to screen"""
        flatfield_mask_name = f"{self.omero_data.plate_id}_flatfield_masks"
        for image in self.omero_data.screen_dataset.listChildren():
            image_name = image.getName()
            if flatfield_mask_name == image_name:
                flatfield_mask_id = image.getId()
                self.flatfield_obj, flatfield_array = get_image(self.conn, flatfield_mask_id)
                flatfield_channels = dict(self._get_map_ann())
                for key, value in flatfield_channels.items():
                    flatfield_channels[key] = value.strip()

#TODO unit test for _get_map_ann
    def _get_map_ann(self):
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """
        annotations = self.flatfield_obj.listAnnotations()
        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]

        if not map_annotations:
            logger.error(f"No Flatfield Mask Channel info found in datset {self.omero_data.screen_dataset}.")  # noqa: G004
            raise ValueError(
                f"No Flatfield Mask Channel info found in datset {self.omero_data.screen_dataset}."
            )

        # Search for a map annotation with "dapi" or "hoechst" in its value (case-insensitive)
        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Assuming the value is a list of tuples or similar structure
            for key, _value in ann_value:
                if key.lower() in ["dapi", "hoechst"]:
                    return map_annotations


        # If no matching annotation is found, raise a specific exception
        logger.error(
            "No DAPI or Hoechst channel information found for flatfieldmasks."
        )  # noqa: G004
        raise ValueError(
           "No DAPI or Hoechst channel information found for flatfieldmasks."
        )

#TODO set up check for flatfieldmask and unit test
    # def _check_flatfieldmask(self, viewer_data, flatfield_channels):
    #     """Checks if flatfieldmask is correct"""
    #     # check if the channels in plate and flatfield_mask are the same
    #     reverse_flatfield_mask = {
    #         v: k.split("_")[-1] for k, v in flatfield_channels.items()
    #     }
    #     # Check if the mappings are consistent
    #     for channel, index in viewer_data.channel_data.items():
    #         try:
    #             assert reverse_flatfield_mask[channel] == index
    #         except AssertionError:
    #             print(
    #                 f"Inconsistency found: {channel} is mapped to {index} in plate_map but {reverse_flatfield_mask[channel]} in flatfield_mask"
    #             )
