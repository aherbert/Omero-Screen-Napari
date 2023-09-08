from PyQt5.QtWidgets import QMessageBox
from qtpy import QtWidgets, QtCore
from omero_gallery import _call_omero_widget
from omero_gallery._call_omero_widget import (
    get_well_img,
    _get_flatfieldmask,
    _get_images_from_well,
)
from conftest import omero_conn, PLATE_ID
import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def app():
    app = QApplication([])
    yield app


# Fixture for monkeypatching
@pytest.fixture
def patch_get_well_img(monkeypatch):
    def no_op_decorator(func):
        return func

    # Replace the @omero_connect decorator with the no-op decorator
    monkeypatch.setattr(
        _call_omero_widget, "get_well_img", no_op_decorator(get_well_img)
    )


def test_get_well_img(omero_conn):
    # Call the get_well_img function with test data
    well_pos = "C2"
    images = get_well_img(PLATE_ID, well_pos)

    # Verify the result
    assert images.shape == (
        3,
        1080,
        1080,
        4,
    ), "Did not load the correct images"


def test_get_flatfieldarray(omero_conn):
    flatfield_array = _get_flatfieldmask(3, omero_conn)
    print(flatfield_array.shape)


def test_images_from_well(omero_conn):
    well = omero_conn.getObject("Well", 51)
    image_ids, final_array = _get_images_from_well(well, omero_conn)
    print(final_array.shape)


from PyQt5.QtWidgets import QMessageBox


# def test_get_well_img_exception_shows_message_box(
#     monkeypatch, patch_get_well_img, omero_conn
# ):
#     # Create a container to store the message box instance
#     message_box_instance = []
#
#     # Custom QMessageBox class
#     class TestMessageBox(QMessageBox):
#         def exec_(self):
#             # Save the instance and details
#             message_box_instance.append(self)
#             self.saved_text = self.text()
#             self.saved_icon = self.icon()
#             self.saved_title = self.windowTitle()
#             self.saved_buttons = self.standardButtons()
#
#     # Replace the constructor of QMessageBox with TestMessageBox
#     monkeypatch.setattr(QMessageBox, "__init__", TestMessageBox.__init__)
#
#     # Assuming a plate_id and well_pos that will cause an exception
#     plate_id = 999999
#     well_pos = "C2"
#
#     # Call the get_well_img function
#     get_well_img(plate_id, well_pos)
#
#     # Retrieve the message box instance and check its properties
#     message_box = message_box_instance[0]
#     expected_text = f"Plate with ID {plate_id} does not exist"
#     assert message_box.saved_text == expected_text
#     assert message_box.saved_icon == QMessageBox.Warning
#     assert message_box.saved_title == "Error"
#     assert message_box.saved_buttons == QMessageBox.Ok


# def test_omero_widget(qtbot, make_napari_viewer_proxy):
#     # Create the widget (replace with actual code to create the widget)
#
#     viewer = make_napari_viewer_proxy()
#     print(type(viewer))
# widget = omero_widget(viewer=viewer)
# # Add the widget to qtbot
# qtbot.addWidget(widget)
#
# # Simulate user input for plate_id and well_pos (replace with actual code to find these fields)
# plate_id_field = widget.findChild(QtWidgets.QLineEdit, "PlateIDField")
# well_pos_field = widget.findChild(QtWidgets.QLineEdit, "WellPosField")
# qtbot.keyClicks(plate_id_field, "123")
# qtbot.keyClicks(well_pos_field, "A1")
#
# # Simulate a click on the call_button (replace with actual code to find the button)
# call_button = widget.findChild(QtWidgets.QPushButton, "CallButton")
# qtbot.mouseClick(call_button, QtCore.Qt.LeftButton)
#
# # Verify that the plate_id and well_pos fields were updated (replace with actual verification code)
# assert plate_id_field.text() == "123"
# assert well_pos_field.text() == "A1"
