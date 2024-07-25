import argparse
import logging

import napari
from omero_screen_napari._welldata_widget import (
    MockEvent,
    add_image_to_viewer,
    add_label_layers,
    clear_viewer_layers,
    handle_metadata_widget,
    set_color_maps,
    welldata_widget,
)
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import parse_omero_data

logger = logging.getLogger("omero-screen-napari")
logging.basicConfig(level=logging.DEBUG)


def load_and_visualize_data(viewer, plate_id, well_pos_list, images):
    parse_omero_data(omero_data, plate_id, well_pos_list, images)
    clear_viewer_layers(viewer)
    add_image_to_viewer(viewer)
    set_color_maps(viewer)
    add_label_layers(viewer)

    def slider_position_change(event):
        current_position = event.source.current_step[0]
        handle_metadata_widget(viewer, current_position)

    viewer.dims.events.current_step.connect(slider_position_change)
    mock_event = MockEvent(viewer.dims)
    slider_position_change(mock_event)




def main():
    parser = argparse.ArgumentParser(
        description="Load training data into Napari"
    )
    parser.add_argument("plate_id", type=str, help="Plate ID")
    parser.add_argument("well", type=str, help="Well Position")
    parser.add_argument("image", type=str, help="Image Index")

    args = parser.parse_args()

    plate_id = args.plate_id
    well_pos_list = args.well
    images = args.image

    viewer = napari.Viewer()
    load_and_visualize_data(viewer, plate_id, well_pos_list, images)

    # Create the widget instance using magicgui and add it to the viewer
    # Create the widget instance using magicgui and add it to the viewer
    widget_instance = welldata_widget()
    widget_instance.viewer.value = viewer  # Set the viewer instance
    widget_instance.plate_id.value = plate_id
    widget_instance.well_pos_list.value = well_pos_list
    widget_instance.images.value = images
    widget_instance.viewer.value = viewer  # Set the viewer instance

    viewer.window.add_dock_widget(widget_instance, area="right")
    napari.run()


if __name__ == "__main__":
    main()
