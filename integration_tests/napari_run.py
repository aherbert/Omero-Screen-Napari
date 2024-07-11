import napari
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

# Define the plate ID, well position, and image index
plate_id = "1237"
well_pos_list = "B7"
images = "0"

user_data_dict = {
    "well": well_pos_list,
    "segmentation": "cell",
    "reload": "Yes",
    "crop_size": 100,
    "cellcycle": "All",
    "columns": 10,
    "rows": 10,
    "contour": True,
    "channels": ["DAPI"],
}
class_options = ["unassigned", "normal", "micro", "collapsed"]
class_name = "nuclei_classifier"

# Create a Napari viewer instance
viewer = napari.Viewer()


# Function to load and visualize the data
def load_and_visualize_data(viewer, plate_id, well_pos_list, images):
    # Parse the OMERO data for the specified plate, well, and image
    parse_omero_data(omero_data, plate_id, well_pos_list, images)
    UserData.set_omero_data_channel_keys(omero_data.channel_data.keys())
    user_data = UserData(**user_data_dict)
    manager = CroppedImageParser(omero_data, user_data)
    manager.parse_crops()
    data_selector = RandomImageParser(omero_data, user_data)
    data_selector.parse_random_images()
    print(len(omero_data.cropped_images))
    print(len(omero_data.selected_images))
    # Clear existing viewer layers
    clear_viewer_layers(viewer)

    # Add the images to the viewer
    add_image_to_viewer(viewer)

    # Set the color maps for the viewer
    set_color_maps(viewer)

    # Add the label layers to the viewer
    add_label_layers(viewer)

    # Set up the slider event handling
    def slider_position_change(event):
        current_position = event.source.current_step[0]
        handle_metadata_widget(viewer, current_position)

    viewer.dims.events.current_step.connect(slider_position_change)
    _initial_position = viewer.dims.current_step[0]
    mock_event = MockEvent(viewer.dims)
    slider_position_change(mock_event)


# Load and visualize the data
load_and_visualize_data(viewer, plate_id, well_pos_list, images)

# Add your plugin widget to the viewer
viewer.window.add_dock_widget(training_widget(class_options, class_name), area='right')



# Run the Napari event loop
napari.run()


