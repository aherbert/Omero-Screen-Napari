from dataclasses import dataclass, field
from omero.gateway import _PlateWrapper, _WellWrapper, _DatasetWrapper
import numpy as np
import pandas as pd


@dataclass
class ViewerData:
    plate_id: int = field(default_factory=int)
    plate: _PlateWrapper = field(default_factory=_PlateWrapper)
    plate_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    well: _WellWrapper = field(default_factory=_WellWrapper)
    screen_dataset: _DatasetWrapper = field(default_factory=_DatasetWrapper)
    channel_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    images: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    image_ids: list = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,)))


viewer_data = ViewerData()
