import os
os.environ["USE_LOCAL_ENV"] = "1"

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.plate_handler import parse_omero_data
from omero_screen_napari.gallery_api import UserData, CroppedImageParser, RandomImageParser, ParseGallery

user_data_dict = {
    "well": "C2",
    "segmentation": "cell",
    "replacement": "True",
    "crop_size": 50,
    "cellcycle": "S",
    "columns": 4,
    "rows": 4,
    "contour": True,
    "channels": ["Tub", "EdU"],
}

def test_parse_omero_data_twice():
    omero_data.reset()
    plate_id = "53"
    well_pos = "C2"
    image_input = "0-2"
    parse_omero_data(omero_data, plate_id, well_pos, image_input)
    UserData.set_omero_data_channel_keys(omero_data.channel_data.keys())  # Inc
    user_data = UserData(**user_data_dict)
    print(user_data)
    manager = CroppedImageParser(omero_data, user_data)
    manager.parse_crops()
    print(len(omero_data.cropped_images))
    data_selector = RandomImageParser(omero_data, user_data)
    data_selector.parse_random_images()
    print(len(omero_data.cropped_images))
    print(len(omero_data.selected_images))
    print(omero_data.selected_images[0].shape)
    gallery_parser = ParseGallery(omero_data, user_data)
    gallery_parser.plot_gallery()
   