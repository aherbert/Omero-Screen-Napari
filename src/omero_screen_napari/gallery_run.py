#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

os.environ["USE_LOCAL_ENV"] = "0"


from omero_screen_napari._gallery_widget import show_gallery
from omero_screen_napari.omero_api import retrieve_data
from omero_screen_napari._omero_utils import omero_connect
from omero_screen_napari.viewer_data_module import viewer_data

plate_id = 1821
images = "0"

blue_channel = "DAPI"
green_channel = ""
red_channel = ""

channels = [red_channel, green_channel, blue_channel]
segmentation = "Cells"
replacement = "With"
crop_size = 100
cellcycle = "All"
rows = 5
columns = 5
contour = True

data_name = f"{plate_id}_galleries_{datetime.now().strftime('%Y%m%d%H%M')}"
data_path = Path.home() / "Desktop" / data_name
data_path.mkdir(exist_ok=False)

def save_fig(fig,
    path: Path, fig_id: str, tight_layout : bool = True, fig_extension: str = "pdf",
        resolution: int = 300) -> None:
    """
    coherent saving of matplotlib figures as pdfs (default)
    :rtype: object
    :param path: path for saving
    :param fig_id: name of saved figure
    :param tight_layout: option, default True
    :param fig_extension: option, default pdf
    :param resolution: option, default 300dpi
    :return: None, saves Figure in poth
    """

    dest = path / f"{fig_id}.{fig_extension}"
    print("Saving figure", fig_id)
    if tight_layout:
        fig.set_tight_layout(True)
    plt.savefig(dest, format=fig_extension, dpi=resolution)

@omero_connect
def retrieve_well_pos(plate_id, conn=None):
    plate = conn.getObject("Plate", plate_id)
    return [well.getWellPos() for well in plate.listChildren()]

if __name__ == "__main__":
    well_pos_list = retrieve_well_pos(plate_id)
    for well in well_pos_list:
        retrieve_data(plate_id, well, images)

    
        figure = show_gallery(
            channels,
            segmentation,
            replacement,
            crop_size,
            rows,
            columns,
            contour,
            cellcycle,
            block=False
        )
        save_fig(figure, data_path, f"{well}_gallery", tight_layout=False)
        plt.close(figure)
