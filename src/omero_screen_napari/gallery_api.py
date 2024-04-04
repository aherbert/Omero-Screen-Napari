from pydantic import BaseModel, validator
from omero_screen_napari.omero_data import OmeroData
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger("omero-screen-napari")


class UserData(BaseModel):
    well: str
    segmentation: str
    replacement: str
    crop_size: int
    cellcycle: str
    columns: int
    rows: int
    contour: bool
    channels: list[str]

    _omero_data_channel_keys = []

    @validator("channels", pre=True, always=True)
    def check_channels(cls, value):
        if not cls._omero_data_channel_keys:
            raise ValueError("omero_data.channel_data has not been set.")

        if not value:
            raise ValueError("No channels have been selected")

        for channel in value:
            if channel not in cls._omero_data_channel_keys:
                raise ValueError(
                    f"{channel} is not a valid key in omero_data.channel_data"
                )
        return value

    @classmethod
    def set_omero_data_channel_keys(cls, channel_keys):
        cls._omero_data_channel_keys = channel_keys


class CroppedImageParser:
    def __init__(self, omero_data: OmeroData, user_data: UserData):
        self._omero_data = omero_data
        self._user_data: UserData = user_data

    def filter_images(self):
        well_images = self.select_wells()
        return self.select_channels(well_images)

    def select_wells(self):
        wells = [well.getWellPos() for well in self._omero_data.well_list]

        try:
            index = wells.index(self._user_data.well)
        except ValueError as e:
            logger.error(
                f"The selected well {self._user_data.well} has not been loaded from the plate"
            )
            raise ValueError(
                f"The selected well {self._user_data.well} has not been loaded from the plate"
            ) from e
        num_images_per_well = self._omero_data.images.shape[0] // len(wells)
        start_idx = index * num_images_per_well
        end_idx = start_idx + num_images_per_well
        return self._omero_data.images[start_idx:end_idx, ...]

    def select_channels(self, image_array: np.ndarray):
        # transform channel number from str to int
        channel_data = {
            k: int(v) for k, v in self._omero_data.channel_data.items()
        }
        order_indices = [
            channel_data[channel] for channel in self._user_data.channels
        ]
        return image_array[..., order_indices]
