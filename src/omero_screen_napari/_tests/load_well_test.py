#!/usr/bin/env python3
import os

os.environ["USE_LOCAL_ENV"] = "0"

import napari  # noqa: E402

from omero_screen_napari._welldata_widget import welldata_widget  # noqa: E402
from omero_screen_napari._gallery_widget import gallery_gui_widget


def test_welldata_widget_interactively():
    # Start Napari

    viewer = napari.Viewer()

    # Initialize the welldata_widget and add it to the viewer
    widget = welldata_widget()
    viewer.window.add_dock_widget(widget)

    # Set default test parameters for convenience
    test_plate_id = 1831
    test_well_position = "A6"
    test_images = 0
    test_image_id = 0

    # Pre-fill the widget with default test values
    widget.plate_id.value = test_plate_id
    widget.well_pos.value = test_well_position
    widget.images.value = test_images
    widget.image_id.value = test_image_id
    widget()
    gallery_widget_instance = gallery_gui_widget()
    viewer.window.add_dock_widget(gallery_widget_instance, area="right")

    # Keep the Napari viewer open for manual inspection
    napari.run()


if __name__ == "__main__":
    test_welldata_widget_interactively()
