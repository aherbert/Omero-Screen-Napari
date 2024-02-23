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
    csv_path: str = field(default_factory=str)
    flatfield_masks: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    well: list[_WellWrapper] = field(default_factory=list)
    well_name: list[str] = field(default_factory=list)
    well_id: list[int] = field(default_factory=list)
    screen_dataset: _DatasetWrapper = field(default_factory=_DatasetWrapper) # dataset with flatfield masks and segementations
    pixel_size: tuple = field(default_factory=tuple)
    channel_data: dict = field(default_factory=dict)
    intensities: dict = field(default_factory=dict)
    metadata: list[dict] = field(default_factory=list)
    images: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    image_ids: list = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    label_ids: list = field(default_factory=list)
    stitched_images: np.ndarray = field(default_factory=lambda: np.empty((0,)))


@dataclass
class CroppedImages:
    cropped_regions: list = field(default_factory=list)
    cropped_labels: list = field(default_factory=list)
    classifier: dict = field(default_factory=dict)


viewer_data = ViewerData()
cropped_images = CroppedImages()

