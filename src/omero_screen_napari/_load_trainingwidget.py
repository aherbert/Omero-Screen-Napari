import napari
from pathlib import Path
import json
import logging


from omero_screen_napari._training_widget import (
    training_widget,
)
from omero_screen_napari._welldata_widget import (
    MockEvent,
    add_image_to_viewer,
    add_label_layers,
    clear_viewer_layers,
    handle_metadata_widget,
    set_color_maps,
)
from omero_screen_napari.gallery_api import (
    CroppedImageParser,
    RandomImageParser,
    UserData,
)
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import parse_omero_data

logger = logging.getLogger("omero-screen-napari")
logging.basicConfig(level=logging.DEBUG)

def parse_data(classifier_name: str):
    data_path = Path.home() / classifier_name
    try:
        data_path.exists()
    except FileNotFoundError as e:
        logger.error(f"File not found: {data_path}")
        return e
    
    with open(data_path / 'metadata.json') as f:
        metadata = json.load(f)
    return metadata



if __name__ == "__main__":

    metadata = parse_data("nuclei_classifier")
    print(metadata)


