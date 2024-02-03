#!/usr/bin/env python3
import os

os.environ["USE_LOCAL_ENV"] = "1"
print(
    f'environment_var = {os.environ.get("USE_LOCAL_ENV")}'
)  # This should print "1"
import napari

from omero_screen_napari._welldata_widget import welldata_widget


def test_welldata_widget_interactively():
    # Start Napari

    viewer = napari.Viewer()

    # Initialize the welldata_widget and add it to the viewer
    widget = welldata_widget()
    viewer.window.add_dock_widget(widget)

    # Set default test parameters for convenience
    test_plate_id = 53
    test_well_position = "C2"
    test_images = 0
    test_image_id = 0

    # Pre-fill the widget with default test values
    widget.plate_id.value = test_plate_id
    widget.well_pos.value = test_well_position
    widget.images.value = test_images
    widget.image_id.value = test_image_id
    widget()
    # Keep the Napari viewer open for manual inspection
    napari.run()


if __name__ == "__main__":
    test_welldata_widget_interactively()
