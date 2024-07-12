import json
import logging
import os
from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, RadioButtons
from qtpy.QtWidgets import QMessageBox

from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.gallery_userdata_singleton import (
    userdata as global_user_data,
)
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import omero_data

logger = logging.getLogger("omero-screen-napari")
logging.basicConfig(level=logging.DEBUG)


class ImageNavigator:
    def __init__(self, class_options: list | None):
        self.current_index = 0
        self.first_load = True  # Flag to check if it is the first image load
        self.saved_contrast_limits = None  # Variable to save user settings
        if class_options is None:
            self.class_options = [
                "unassigned",
                "class1",
                "class2",
                "class3",
                "class4",
            ]
        else:
            self.class_options = class_options
        self.class_choice = RadioButtons(
            label="Select class:",
            choices=self.class_options,
            value="unassigned",
        )
        self.class_choice.changed.connect(self.assign_class)

    def next_image(self):
        if omero_data.selected_images:
            self.current_index = (self.current_index + 1) % len(
                omero_data.selected_images
            )
            self.update_image()

    def previous_image(self):
        if omero_data.selected_images:
            self.current_index = (self.current_index - 1) % len(
                omero_data.selected_images
            )
            self.update_image()

    def update_image(self):
        viewer = napari.current_viewer()
        current_choices = self.class_choice.choices
        self.class_choice.changed.disconnect(self.assign_class)

        # Save current settings if there are any layers and it's not the first load
        if not self.first_load and viewer.layers:
            self.saved_contrast_limits = viewer.layers[0].contrast_limits
            logger.debug(
                f"Saving contrast limits: {self.saved_contrast_limits}"
            )

        # Clear existing layers
        viewer.layers.clear()
        logger.debug("Viewer layers cleared.")

        if omero_data.selected_images:
            image = omero_data.selected_images[self.current_index]
            logger.debug(f"Selected image shape: {image.shape}")

            try:
                # Check the number of channels and adjust accordingly
                if omero_data.cropped_images[0].shape[-1] == 1:
                    # Single channel, load as greyscale
                    grayscale_image = np.mean(image, axis=-1)
                    logger.debug(
                        f"Grayscale image shape: {grayscale_image.shape}"
                    )
                    inverted_image = 1.0 - grayscale_image  # Invert the image
                    viewer.add_image(
                        inverted_image,
                        name=f"Cropped Image {self.current_index}",
                        colormap="gray",
                    )
                    logger.debug("Inverted grayscale image added to viewer.")
                else:
                    # Three channels, load as RGB
                    viewer.add_image(
                        image, name=f"Cropped Image {self.current_index}"
                    )
                    logger.debug("RGB image added to viewer.")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error adding image to viewer: {e}")
                return

            # Verify the layer was added
            if not viewer.layers:
                logger.error(
                    "No layers present in the viewer after adding the image."
                )
                return

            # Apply contrast limits if there are any and not the first load
            if not self.first_load and self.saved_contrast_limits:
                min_intensity, max_intensity = (
                    viewer.layers[0].data.min(),
                    viewer.layers[0].data.max(),
                )
                if (
                    self.saved_contrast_limits[0] >= min_intensity
                    and self.saved_contrast_limits[1] <= max_intensity
                ):
                    logger.debug(
                        f"Applying contrast limits: {self.saved_contrast_limits}"
                    )
                    viewer.layers[
                        0
                    ].contrast_limits = self.saved_contrast_limits
                else:
                    logger.warning(
                        f"Contrast limits {self.saved_contrast_limits} are out of range for the new image intensity values."
                    )
            else:
                logger.debug(
                    "No contrast limits to apply or first image load."
                )

            self.first_load = False  # Update the flag after the first load
        else:
            logger.warning("No selected images to load.")

        # Restore class choices and reconnect signal
        self.class_choice.choices = current_choices
        self.class_choice.changed.connect(self.assign_class)
        self.update_class_choice()
        logger.debug("Class choices restored and signal reconnected.")

        # Force a refresh of the viewer
        viewer.update_console({"layers": viewer.layers})
        logger.debug("Viewer updated successfully.")

    def assign_class(self, class_name: str):
        if omero_data.selected_classes:
            omero_data.selected_classes[self.current_index] = class_name

    def update_class_choice(self):
        if omero_data.selected_classes:
            current_class = omero_data.selected_classes[self.current_index]
        else:
            current_class = "unassigned"
        self.class_choice.value = (
            current_class
            if current_class in self.class_choice.choices
            else "unassigned"
        )

    def add_class(self, class_name: str):
        if class_name and class_name not in self.class_choice.choices:
            self.class_choice.choices = list(self.class_choice.choices) + [
                class_name
            ]
            logger.debug(f"Class {class_name} added to choices.")

    def reset_class_options(self):
        self.class_choice.choices = ["unassigned"]
        logger.debug("Class choices reset to default.")


class TrainingWidget:
    def __init__(
        self,
        class_options: list[str] | None,
        class_name: str | None,
        user_data: UserData | None,
        omero_data: OmeroData,
    ):
        self.image_navigator = ImageNavigator(class_options)
        self.user_data = user_data
        self.omero_data = omero_data
        self.setup_key_bindings(napari.current_viewer())

        self.load_image = magicgui(call_button="Load Images")(self.load_image)
        self.next_image = magicgui(call_button="Next Image")(self.next_image)
        self.previous_image = magicgui(call_button="Previous Image")(
            self.previous_image
        )
        self.add_class = magicgui(
            call_button="Enter", text_input={"label": "Class name"}
        )(self.add_class)
        self.reset_class_options = magicgui(call_button="Reset class options")(
            self.reset_class_options
        )
        name_input = class_name or "Classifier Name"

        self.save_training_data = magicgui(
            call_button="Save training data",
            text_input={"label": "filename", "value": name_input},
        )(self.save_training_data)

        self.container = self.create_container()

    def load_image(self):
        if not omero_data.selected_images:
            print("No images to load.")
            return
        selected_images_length = len(omero_data.selected_images)
        omero_data.selected_classes = [
            "unassigned" for _ in range(selected_images_length)
        ]
        self.image_navigator.current_index = 0
        self.image_navigator.update_image()

    def next_image(self):
        if not omero_data.selected_classes:
            print("No images loaded.")
            return
        self.image_navigator.next_image()

    def previous_image(self):
        if not omero_data.selected_classes:
            print("No images loaded.")
            return
        self.image_navigator.previous_image()

    def add_class(self, text_input: str):
        if text_input:
            self.image_navigator.add_class(text_input)
            self.add_class.text_input.value = ""

    def reset_class_options(self):
        self.image_navigator.reset_class_options()

    def save_training_data(self, text_input: str):
        try:
            # Ensure user enters a classifier name
            self._validate_classifier_name(text_input)

            home_dir = Path.home()
            classifier_dir = home_dir / text_input
            self._create_directory(classifier_dir)

            file_path = classifier_dir / f"{self.omero_data.image_ids[0]}.npy"
            meta_data_path = classifier_dir / "metadata.json"

            if self._file_exists_and_confirm(file_path):
                return

            # Create dictionaries of training and metadata
            training_dict = self._create_training_dict()
            metadata = self._create_metadata_dict()

            if self._metadata_exists_and_confirm(meta_data_path, metadata):
                return

            # Save the training data and metadata
            np.save(file_path, training_dict)
            self._save_metadata(meta_data_path, metadata)
            logger.info(f"File and metadata saved to: {classifier_dir}")

            # Show success message
            self._show_success_message(
                f"Data for image {self.omero_data.image_ids[0]} successfully saved."
            )
        except Exception as e:  # noqa: BLE001
            logger.error(e)
            self._show_error_message(str(e))

    def _validate_classifier_name(self, text_input: str):
        if not text_input.strip():
            logger.error("No classifier name provided.")
            raise ValueError(
                "Failed to create directory: no classifier name provided."
            )

    def _create_directory(self, directory: Path):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise ValueError(
                f"Failed to create directory {directory}: {e}"
            ) from e

    def _file_exists_and_confirm(self, file_path: Path) -> bool:
        if file_path.exists():
            return not self._show_confirmation_dialog(
                f"The file {file_path} already exists. Do you want to overwrite it?",
                "Warning",
            )
        return False

    def _create_training_dict(self) -> dict:
        return {
            "target": self.omero_data.selected_classes,
            "data": self.omero_data.selected_images,
        }

    def _create_metadata_dict(self) -> dict:
        user_data_dict = self.user_data.to_dict()
        user_data_dict.pop("well", None)
        return {
            "user_data": user_data_dict,
            "class_options": self.image_navigator.class_options,
        }

    def _metadata_exists_and_confirm(
        self, meta_data_path: Path, metadata: dict
    ) -> bool:
        if meta_data_path.exists():
            with meta_data_path.open("r") as json_file:
                existing_metadata = json.load(json_file)

            if existing_metadata != metadata:
                return not self._show_confirmation_dialog(
                    "Metadata have changed. Do you want to overwrite the file?",
                    "Warning",
                )
            logger.info("Metadata is already in place and identical.")
            return True
        return False

    def _show_confirmation_dialog(self, message: str, title: str) -> bool:
        # sourcery skip: class-extract-method
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle(title)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        reply = msg_box.exec_()
        return reply == QMessageBox.Yes

    def _show_error_message(self, message: str):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _show_success_message(self, message: str):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Success")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _save_metadata(self, meta_data_path: Path, metadata: dict):
        with meta_data_path.open("w") as json_file:
            json.dump(metadata, json_file)

    def setup_key_bindings(self, viewer):
        @viewer.bind_key("w")
        def trigger_next_image(event=None):
            self.next_image()

        @viewer.bind_key("q")
        def trigger_previous_image(event=None):
            self.previous_image()

    def create_container(self):
        return Container(
            widgets=[
                self.load_image,
                self.previous_image,
                self.next_image,
                self.add_class,
                self.image_navigator.class_choice,
                self.reset_class_options,
                self.save_training_data,
            ]
        )


def training_widget(
    class_options: list[str] | None = None,
    class_name: str | None = None,
    user_data: UserData | None = global_user_data,
) -> Container:
    widget = TrainingWidget(class_options, class_name, user_data, omero_data)
    return widget.container
