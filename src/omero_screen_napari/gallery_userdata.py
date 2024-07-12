from pydantic import BaseModel, validator


class UserData(BaseModel):
    well: str
    segmentation: str
    reload: bool
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

    @classmethod
    def set_defaults(cls, channel_keys, **kwargs):
        """Set default values for an instance of UserData."""
        default_values = {
            "well": "default_well",
            "segmentation": "default_segmentation",
            "reload": True,
            "crop_size": 100,
            "cellcycle": "default_cellcycle",
            "columns": 10,
            "rows": 10,
            "contour": True,
            "channels": channel_keys,
        } | kwargs
        return cls(**default_values)

    @classmethod
    def reset_with_input(cls, instance, **kwargs):
        """Reset the existing instance of UserData with new values from a dictionary."""
        for field, value in kwargs.items():
            if hasattr(instance, field):
                setattr(instance, field, value)
        return instance

    def to_dict(self):
        return self.dict()
