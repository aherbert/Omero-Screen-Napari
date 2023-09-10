from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage
from omero_gallery.viewer_data_module import cropped_images
import numpy as np
import sys
from magicgui import magic_factory

global_widget = None


@magic_factory(
    call_button="Enter",
)
def gui_widget():
    print("Hello world!")
    start_gui()


def on_enter_button_clicked():
    print("Hello, World!")


def normalize_to_uint8(img):
    flipped_img = img[:, :, ::-1]
    # Find the min and max values of the image
    min_val = np.min(flipped_img)
    max_val = np.max(flipped_img)

    # Normalize the image to [0, 1]
    img_normalized = (flipped_img - min_val) / (max_val - min_val)

    # Scale the image to [0, 255]
    img_scaled = img_normalized * 255

    return img_scaled.astype("uint8")


def array_to_qimage(array):
    # Normalize the array to uint8
    array = normalize_to_uint8(array)

    # Existing QImage conversion code
    height, width, _ = array.shape
    bytesPerLine = width * 3
    array = np.require(array, np.uint8, "C")
    return QImage(
        array.data, width, height, bytesPerLine, QImage.Format_RGB888
    )


def start_gui():
    global global_widget
    if global_widget is not None:
        global_widget.close()

    global_widget = QWidget()
    global_widget.resize(500, 500)
    global_widget.move(300, 300)
    global_widget.setWindowTitle("Simple")

    # Create layout
    layout = QVBoxLayout()

    # Create an "Enter" button
    enter_button = QPushButton("Enter")
    enter_button.resize(100, 40)

    # Connect the "Enter" button to the function
    enter_button.clicked.connect(on_enter_button_clicked)

    # Create image label
    image_label = QLabel()

    # Set the size policy of the image label
    sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    image_label.setSizePolicy(sizePolicy)

    # Get the first cropped image
    cropped_image = cropped_images.cropped_regions[0]

    # Convert NumPy array to QImage and then QPixmap
    qimage = array_to_qimage(cropped_image)
    pixmap = QPixmap.fromImage(qimage)

    # Set QPixmap to QLabel
    image_label.setPixmap(pixmap)
    image_label.setScaledContents(True)

    # Add button to layout
    layout.addWidget(enter_button)
    layout.addWidget(image_label)

    # Set the layout for the widget
    global_widget.setLayout(layout)

    global_widget.show()


if __name__ == "__main__":
    sample_image = np.load("/Users/hh65/Desktop/sample_data/sample_img.npy")

    class MockCroppedImages:
        def __init__(self):
            self.cropped_regions = [
                sample_image,
            ]

    cropped_images = MockCroppedImages()
    app = QApplication(sys.argv)
    start_gui()
    sys.exit(app.exec_())
