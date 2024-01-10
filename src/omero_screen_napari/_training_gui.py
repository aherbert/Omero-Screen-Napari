from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QLineEdit,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtCore import Qt


from omero_screen_napari.viewer_data_module import viewer_data, cropped_images
from omero_screen_napari._gallery_widget import (
    select_channels,
    generate_crops,
    draw_contours,
    fill_missing_channels,
)
from omero_screen_napari.omero_utils import omero_connect
from omero_screen_napari._handle_traningdata import (
    get_saved_data,
    save_trainingdata,
)
import numpy as np
from functools import partial
from magicgui import magic_factory
from magicgui.widgets import Container
from scipy.ndimage import zoom

channel_list = list(viewer_data.channel_data.keys())

# iniatiate combined widget with two magic factories in a container
def gui_widget():
    # Call the magic factories to get the widget instances
    global training_widget_instance
    training_widget_instance = training_widget()

    saving_widget_instance = saving_widget()
    print(training_widget_instance.exp_name.value)

    return Container(widgets=[training_widget_instance, saving_widget_instance])



@magic_factory(
    call_button="Start",
    segmentation={"choices": ["Nuclei", "Cells"]},
    crop_size={"choices": [20, 30, 50, 100]},
    cellcycle={"choices": ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]},
)
def training_widget(
    # viewer: "napari.viewer.Viewer",

    segmentation: str,
    crop_size: int,
    cellcycle: str,
    exp_name: str = None,
    contour: bool = True,
    red_channel: str = channel_list[2] if len(channel_list) > 0 else None,
    green_channel: str = channel_list[1] if len(channel_list) > 1 else None,
    blue_channel: str = channel_list[0] if len(channel_list) > 2 else None,
):
    if exp_name != "None":
        get_saved_data(viewer_data.well_id, exp_name)
        training_widget_instance.segmentation.value = cropped_images.classifier['segmentation']
        training_widget_instance.crop_size.value = cropped_images.classifier['crop_size']
        training_widget_instance.cellcycle.value = cropped_images.classifier['cellcycle']
        training_widget_instance.contour.value = cropped_images.classifier['contour']
        training_widget_instance.blue_channel.value = cropped_images.classifier['blue_channel']
        training_widget_instance.green_channel.value = cropped_images.classifier['green_channel']
        training_widget_instance.red_channel.value = cropped_images.classifier['red_channel']

    else:
        classifier = {
            "segmentation": segmentation,
            "crop_size": crop_size,
            "cellcycle": cellcycle,
            "exp_name": exp_name,
            "contour": contour,
            "blue_channel": blue_channel,
            "green_channel": green_channel,
            "red_channel": red_channel,
            "labels": []
        }
        cropped_images.classifier = classifier

    channels = [training_widget_instance.blue_channel.value, training_widget_instance.green_channel.value, training_widget_instance.red_channel.value]
    get_cropped_images(
        channels, segmentation, crop_size, contour, cellcycle
    )
    if exp_name == "None":
        cropped_images.classifier["labels"] = ["unassigned"] * len(
            cropped_images.cropped_regions
        )
    app = QApplication.instance()
    if app is None:
        app = QApplication(
            []
        )  # Create new QApplication instance if it doesn't exist
    widget = TrainingDataWidget()


@magic_factory(call_button="Save")
def saving_widget(exp_name: str):
    well_id = viewer_data.well_id
    save_trainingdata(well_id, exp_name)


## functionality of first magic factory: training_widget


def get_cropped_images(channels, segmentation, crop_size, contour, cellcycle):
    filtered_channels = list(filter(None, channels))

    try:
        images = select_channels(filtered_channels)
        generate_crops(images, segmentation, cellcycle, crop_size)
        adjust_crops(contour, channels, crop_size)

    except Exception as e:
        # Show a message box with the error message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(str(e))
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def adjust_crops(contour, channels, crop_size):
    cropped_images.cropped_regions = [
        fill_missing_channels(img, channels, crop_size)
        for img in cropped_images.cropped_regions
    ]

    if contour == True:
        cropped_images.cropped_regions = [
            draw_contours(cell, label)
            for cell, label in zip(
                cropped_images.cropped_regions, cropped_images.cropped_labels
            )
        ]


# Initialize global variables for handling the images
global_widget = None
current_image_index = 0
info_label = None

# Initialize global variables for storing images and labels
image_data = None
label_data = None


class TrainingDataWidget(QWidget):
    def __init__(self):
        super(TrainingDataWidget, self).__init__()
        self.current_image_index = 0
        self.image_data = None
        self.label_data = None
        self.class_name_edits = []
        self.initialize_ui()
        self.image_label = self.create_image_label()
        self.info_label = self.create_info_label()
        self.setLayout(self.create_main_layout())
        self.update_image_and_info()
        self.show()

    def create_main_layout(self):
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.create_navigation_buttons())
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(self.create_text_fields_and_buttons())
        return main_layout

    def initialize_ui(self):
        self.resize(500, 650)
        self.move(300, 300)
        self.setWindowTitle("Training Data Widget")

    def create_navigation_buttons(self):
        navigation_layout = QHBoxLayout()
        previous_button = QPushButton("Previous")
        previous_button.clicked.connect(self.on_previous_button_clicked)
        enter_button = self.create_enter_button()
        navigation_layout.addWidget(previous_button)
        navigation_layout.addWidget(enter_button)
        return navigation_layout

    def create_enter_button(self):
        enter_button = QPushButton("Next")
        enter_button.resize(100, 40)
        enter_button.clicked.connect(self.on_next_button_clicked)
        return enter_button

    def create_image_label(self):
        self.image_label = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.update_image_and_info()
        return self.image_label

    def create_info_label(self):
        self.info_label = QLabel()
        return self.info_label

    def create_text_fields_and_buttons(self):
        # get list of labels
        labels = set(cropped_images.classifier['labels'])
        other_values = [value for value in labels if value != 'unassigned']
        # Calculate how many 'unassigned' values we need
        num_unassigned = 4 - len(other_values)
        final_labels = other_values + ['unassigned'] * num_unassigned
        text_button_layout = QVBoxLayout()
        for i in range(4):
            hbox = QHBoxLayout()
            line_edit = QLineEdit(final_labels[i])
            self.class_name_edits.append(line_edit)
            button = QPushButton("Enter")
            button.clicked.connect(
                partial(self.on_class_enter_button_clicked, i)
            )
            hbox.addWidget(line_edit)
            hbox.addWidget(button)
            text_button_layout.addLayout(hbox)
        return text_button_layout

    def update_image_and_info(self):
        if self.current_image_index < len(cropped_images.cropped_regions):
            cropped_image = cropped_images.cropped_regions[
                self.current_image_index
            ]
            classifier_label = cropped_images.classifier['labels'][
                self.current_image_index
            ]  # New line
            qimage = array_to_qimage(cropped_image)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(
                pixmap.scaled(800, 400, Qt.KeepAspectRatio)
            )
            self.update_info_label(classifier_label)  # Updated
        else:
            print("Reached the end of the image list")

    def update_info_label(self, classifier_label):  # Updated
        if hasattr(self, "info_label"):
            label_text = f"Cell {self.current_image_index + 1} of {len(cropped_images.cropped_regions)}"

            # Using classifier_label now
            label_text += f" - Classifier: {classifier_label}"

            self.info_label.setText(label_text)

    def add_image_data(self, class_name):
        current_image = cropped_images.cropped_regions[
            self.current_image_index
        ]
        if self.image_data is None:
            self.image_data = np.expand_dims(current_image, axis=0)
            self.label_data = np.array([class_name])
        else:
            self.image_data = np.concatenate(
                (self.image_data, np.expand_dims(current_image, axis=0)),
                axis=0,
            )
            self.label_data = np.append(self.label_data, class_name)

    def on_class_enter_button_clicked(self, class_index):
        class_name = self.class_name_edits[class_index].text()
        cropped_images.classifier['labels'][
            self.current_image_index
        ] = class_name  # Update the classifier label
        self.update_image_and_info()  # Refresh the displayed information

    def on_next_button_clicked(self):
        if (
            self.current_image_index < len(cropped_images.cropped_regions) - 1
        ):  # Check bounds
            self.current_image_index += 1
            self.update_image_and_info()

    def on_previous_button_clicked(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_and_info()


## Helper functions to add the image to the viewer


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
    channel_list = [
        cropped_images.classifier['red_channel'],
        cropped_images.classifier['green_channel'],
        cropped_images.classifier['blue_channel']]
    channel_list = [item for item in channel_list if item != '']

    # If the input array is a single-channel image, just show a single grayscale image
    if len(channel_list) == 1:
        squashed_image = np.max(array, axis=-1)
        gray_image = 255 - squashed_image  # invert grayscale
        height, width = gray_image.shape
        bytesPerLine = width
        gray_image = np.require(gray_image, np.uint8, "C")
        return QImage(gray_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        # Convert RGB channel to reversed gray

    def rgb_to_reversed_gray(channel):
        gray = 255 - channel  # Inverting the grayscale image
        return gray

        # Convert single channel grayscale to 3-channel grayscale

    def to_three_channel(image):
        return np.stack([image, image, image], axis=-1)

        # For the selected channels

    gray_images = [rgb_to_reversed_gray(array[:, :, i]) for i in range(len(channel_list))]

    # Resize the grayscale images and convert them to 3-channel grayscale
    resized_gray_images = [to_three_channel(zoom(image, (6, 6), order=1)) for image in gray_images]

    # Rescale the original image too, to match the other images in size
    resized_rgb = zoom(array, (6, 6, 1), order=1)

    # Concatenate the grayscale images and the RGB image side by side
    concatenated_image = np.hstack(resized_gray_images + [resized_rgb])

    # Adjust the title height and create QImage
    title_height = 30  # Height of the area reserved for titles
    height, width, _ = concatenated_image.shape
    total_height = height + title_height
    bytesPerLine = width * 3
    q_image = QImage(width, total_height, QImage.Format_RGB888)
    q_image.fill(QColor(255, 255, 255))  # Fill with white or any background color

    # Draw titles on the image
    painter = QPainter(q_image)
    painter.setPen(QColor(0, 0, 0))  # Black color for text
    painter.setFont(QFont('Arial', 12))  # Font settings

    # Adjust single_image_width based on the number of channels
    num_images = len(channel_list) + 1  # +1 for the overlay
    single_image_width = width // num_images

    # Add 'Overlay' to the channel list for the title
    channel_list.append('Overlay')

    # Draw titles
    for i, channel in enumerate(channel_list):
        title = channel
        painter.drawText(single_image_width * i + 10, title_height - 10, title)

    # Draw the concatenated image below the titles
    painter.drawImage(0, title_height,
                      QImage(concatenated_image.data, width, height, bytesPerLine, QImage.Format_RGB888))

    painter.end()

    return q_image

if __name__ == "__main__":

    @omero_connect
    def get_well(conn=None):
        well = conn.getObject("Well", 51)
        viewer_data.well = well

    sample_data = np.load("../../sample_data/sample_imgs.npy")
    cropped_images.cropped_regions = [
        sample_data[i, :, :, :] for i in range(sample_data.shape[0])
    ]

    cropped_images.classifier['labels'] = ["unassigned"] * len(
        cropped_images.cropped_regions
    )

    get_well()
    app = QApplication([])  # QApplication instance
    widget = TrainingDataWidget()
    app.exec_()
