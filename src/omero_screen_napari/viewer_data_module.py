from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omero.gateway import _DatasetWrapper, _PlateWrapper, _WellWrapper


@dataclass
class ViewerData:
    project_id: int = field(default_factory=int)
    plate_id: int = field(default_factory=int)
    plate_name: str = field(default_factory=str)
    plate: _PlateWrapper = field(default_factory=_PlateWrapper)
    plate_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    well: _WellWrapper = field(default_factory=_WellWrapper)
    well_name: str = field(default_factory=str)
    well_id: int = field(default_factory=int)
    screen_dataset: _DatasetWrapper = field(default_factory=_DatasetWrapper)
    pixel_size: tuple = field(default_factory=tuple)
    channel_data: dict = field(default_factory=dict)
    intensities: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    images: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    image_ids: list = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    stitched_images: np.ndarray = field(default_factory=lambda: np.empty((0,)))


@dataclass
class CroppedImages:
    cropped_regions: list = field(default_factory=list)
    cropped_labels: list = field(default_factory=list)
    classifier: dict = field(default_factory=dict)


viewer_data = ViewerData()
cropped_images = CroppedImages()
