import json
import logging
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


def training_widget(
    class_options: list[str] | None = None,
    class_name: str | None = None,
    user_data: UserData | None = global_user_data,
) -> Container:
    widget = TrainingWidget(class_options, class_name, user_data, omero_data)
    return widget.container


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
        self.class_name = class_name or "Classifier Name"
        self.training_data_saver = TrainingDataSaver(
            self.class_name,
            self.omero_data,
            self.user_data,
            self.image_navigator,
        )
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

        self.save_training_data = magicgui(
            call_button="Save training data",
            text_input={"label": "filename", "value": self.class_name},
        )(self.save_training_data)

        self.container = self.create_container()

    def load_image(self):
        if not self.omero_data.selected_images:
            print("No images to load.")
            return
        selected_images_length = len(self.omero_data.selected_images)
        self.omero_data.selected_classes = [
            "unassigned" for _ in range(selected_images_length)
        ]
        self.image_navigator.current_index = 0
        self.image_navigator.update_image()

    def next_image(self):
        if not self.omero_data.selected_classes:
            print("No images loaded.")
            return
        self.image_navigator.next_image()

    def previous_image(self):
        if not self.omero_data.selected_classes:
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
        self.training_data_saver._save_data()

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
class TrainingDataSaver:
    def __init__(
        self,
        classifier_name: str,
        omero_data: OmeroData,
        user_data: UserData,
        image_navigator: ImageNavigator,
    ):
        self.classifier_name = classifier_name
        self.omero_data = omero_data
        self.user_data = user_data
        self.image_navigator = image_navigator
        self.home_dir = Path.home() / "omeroscreen_trainingdata"
        self.classifier_dir = self.home_dir / self.classifier_name
        self.file_name, self.file_path, self.meta_data_path = self._set_paths()
        self.training_dict = self._create_training_dict()
        self.metadata = self._create_metadata_dict()

    def _save_data(self):
        try:
            self._validate_classifier_name(self.classifier_name)
            if not self.classifier_dir.exists():
                self._create_and_save()
                return

            file_check = self._check_directory_contents()
            self._handle_saving_logic(file_check)
        except Exception as e:  # noqa: BLE001
            logger.error(e)
            self._show_error_message(str(e))

    def _create_and_save(self):
        self._create_directory(self.classifier_dir)
        self.save_both()

    def _check_directory_contents(self):
        logger.info(f"Directory {self.classifier_dir} already exists.")
        contents = list(self.classifier_dir.iterdir())
        return self.check_files(contents)

    def _handle_saving_logic(self, file_check):
        if file_check in ["empty", "neither"]:
            self.save_both()
        elif file_check == "metadata":
            self._handle_metadata_present()
        elif file_check == "training_data":
            self._handle_training_data_present()
        elif file_check == "both":
            self._handle_both_present()

    def _handle_metadata_present(self):
        if self.compare_metadata():
            self._save_training_data()
        elif self._show_confirmation_dialog(
            "Metadata has changed. Do you want to overwrite them and save {self.file_name}?"
        ):
            self.save_both()

    def _handle_training_data_present(self):
        if self._show_confirmation_dialog(
            "The file {self.file_name} already exists without metadata. Do you want to overwrite the file and save the metadata?"
        ):
            self.save_both()

    def _handle_both_present(self):
        if self.compare_metadata():
            if self._show_confirmation_dialog(f"Do you want to overwrite {self.file_name}?"):
                self._save_training_data()
        elif self._show_confirmation_dialog(
            "{self.file_name} and metadata have changed. Do you want to overwrite both?"
        ):
            self.save_both()

    def save_both(self):
        np.save(self.file_path, self.training_dict)
        self._save_metadata(self.meta_data_path, self.metadata)
        logger.info(f"File and metadata saved to: {self.classifier_dir}")
        self._show_success_message(f"Data for {self.file_name} and metadata successfully saved.")

    def _save_training_data(self):
        np.save(self.file_path, self.training_dict)
        logger.info(f"File saved to: {self.file_path}, metadata already present")
        self._show_success_message(f"Data for image {self.file_name} successfully saved, with metadata present.")

    def _set_paths(self):
        plate = self.omero_data.plate_id
        well = self.omero_data.well_pos_list[0]
        image = self.omero_data.image_index[0]
        file_name = f"{plate}_{well}_{image}.npy"
        file_path = self.classifier_dir / file_name
        meta_data_path = self.classifier_dir / "metadata.json"
        return file_name, file_path, meta_data_path

    def check_files(self, contents):
        has_metadata = any(file.name == "metadata.json" for file in contents)
        has_self_file = any(file.name == self.file_name for file in contents)

        if not contents:
            return "empty"
        if has_metadata:
            return "both" if has_self_file else "metadata"
        return "training_data" if has_self_file else "neither"

    def compare_metadata(self) -> bool:
        with self.meta_data_path.open("r") as json_file:
            existing_metadata = json.load(json_file)
        return existing_metadata == self.metadata

    def _validate_classifier_name(self, text_input: str):
        if not text_input.strip():
            logger.error("No classifier name provided.")
            raise ValueError("Failed to create directory: no classifier name provided.")

    def _create_directory(self, directory: Path):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise ValueError(f"Failed to create directory {directory}: {e}") from e

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

    def _save_metadata(self, meta_data_path: Path, metadata: dict):
        with meta_data_path.open("w") as json_file:
            json.dump(metadata, json_file)

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

    def _show_confirmation_dialog(self, message: str) -> bool:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle('Warning')
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        reply = msg_box.exec_()
        return reply == QMessageBox.Yes
