from qtpy import QtWidgets, QtCore

from omero_gallery._call_omero_widget import omero_widget


def test_omero_widget(qtbot, make_napari_viewer_proxy):
    # Create the widget (replace with actual code to create the widget)

    viewer = make_napari_viewer_proxy()
    print(type(viewer))
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
