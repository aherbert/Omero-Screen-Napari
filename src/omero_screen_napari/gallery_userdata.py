import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("omero-screen-napari")

@dataclass
class UserData:
    well: str = field(default_factory=str)
    segmentation: str = field(default_factory=str)
    reload: bool = field(default_factory=str)
    crop_size: int = field(default_factory=int)
    cellcycle: str = field(default_factory=str)
    columns: int = field(default_factory=int)
    rows: int = field(default_factory=int)
    contour: bool = field(default_factory=bool)
    channels: list[str] = field(default_factory=list[str])

    def populate_from_dict(self, data: dict[str, Any]):
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    print(f"Updated {key} to {value}")
                else:
                    print(f"Error: {key} is not a valid attribute of UserData")



    def reset(self):
            """
            Reset to default values.
            """
            self.__init__()
















# from pydantic import BaseModel, validator


# class UserData(BaseModel):
#     well: str
#     segmentation: str
#     reload: bool
#     crop_size: int
#     cellcycle: str
#     columns: int
#     rows: int
#     contour: bool
#     channels: list[str]

#     _omero_data_channel_keys = []

#     @validator("channels", pre=True, always=True)
#     def check_channels(cls, value):
#         if not cls._omero_data_channel_keys:
#             raise ValueError("omero_data.channel_data has not been set.")

#         if not value:
#             raise ValueError("No channels have been selected")

#         for channel in value:
#             if channel not in cls._omero_data_channel_keys:
#                 raise ValueError(
#                     f"{channel} is not a valid key in omero_data.channel_data"
#                 )
#         return value

#     @classmethod
#     def set_omero_data_channel_keys(cls, channel_keys):
#         cls._omero_data_channel_keys = channel_keys

#     @classmethod
#     def set_defaults(cls, channel_keys, **kwargs):
#         """Set default values for an instance of UserData."""
#         default_values = {
#             "well": "default_well",
#             "segmentation": "default_segmentation",
#             "reload": True,
#             "crop_size": 100,
#             "cellcycle": "default_cellcycle",
#             "columns": 10,
#             "rows": 10,
#             "contour": True,
#             "channels": channel_keys,
#         } | kwargs
#         return cls(**default_values)

#     @classmethod
#     def reset_with_input(cls, instance, **kwargs):
#         """Reset the existing instance of UserData with new values from a dictionary."""
#         for field, value in kwargs.items():
#             if value is None and field == "well":
#                 value = "default_well"
#             if hasattr(instance, field):
#                 setattr(instance, field, value)
#         return instance

#     def to_dict(self):
#         return self.dict()
