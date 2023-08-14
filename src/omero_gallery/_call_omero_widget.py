"""
This module handles the widget to call Omero and load well images
"""
import napari
from omero_gallery.omero_utils import omero_connect
from magicgui import magic_factory
from omero.gateway import BlitzGateway
import numpy as np
from ezomero import get_image
from qtpy.QtWidgets import QMessageBox


@magic_factory(call_button="Enter")
def omero_widget(
    viewer: "napari.viewer.Viewer",
    plate_id: str = "Plate ID",
    well_pos: str = "Well Position",
):
    # Clear all layers from the viewer
    viewer.layers.select_all()
    viewer.layers.remove_selected()
    images = get_well_img(int(plate_id), well_pos)
    viewer.add_image(images, channel_axis=-1)


@omero_connect
def get_well_img(plate_id: int, well_pos: str, conn: BlitzGateway = None):
    """
    Get well image from plate and well position
    :param conn: Blitzgateway connection to Omero
    :return: well image
    """
    print(f"Plate_ID input received: {plate_id}")
    print(f"Well input received: {well_pos}")
    try:
        plate = conn.getObject("Plate", plate_id)
        if plate is None:
            raise ValueError(f"Plate with ID {plate_id} does not exist")

        images = None
        for well in plate.listChildren():
            if well.getWellPos() == well_pos:
                final_array = get_images_from_well(well, conn)
                images = final_array.squeeze()

        if images is not None and images.any():
            return images
        else:
            raise ValueError(
                f"Well position {well_pos} does not exist in plate"
            )
    except Exception as e:
        # Show a message box with the error message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(str(e))
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def get_images_from_well(well, conn):
    # List to store the individual image arrays
    image_arrays = []
    index = well.countWellSample()
    for index in range(index):
        image, image_array = get_image(conn, well.getImage(index).getId())
        image_arrays.append(image_array)
    return np.stack(image_arrays, axis=0)


if __name__ == "__main__":

    @omero_connect
    def image_test(well_id, conn=None):
        well = conn.getObject("Well", well_id)
        return get_images_from_well(well, conn)

    array = image_test(15401)
    print(array.shape)
