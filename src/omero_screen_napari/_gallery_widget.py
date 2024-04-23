import logging
from pathlib import Path
import datetime

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container


from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.gallery_api import show_gallery, run_gallery_parser
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.gallery_userdata_singleton import userdata


logger = logging.getLogger("omero-screen-napari")


def gallery_gui_widget():
    # Call the magic factories to get the widget instances
    gallery_widget_instance = gallery_widget()
    reset_widget_instance = reset_widget()
    analysis_widget_instance = run_analysis_widget()
    return Container(widgets=[gallery_widget_instance, reset_widget_instance, analysis_widget_instance])


@magic_factory(
    call_button="Enter",
)
def reset_widget():
    omero_data.cropped_images = []
    omero_data.cropped_labels = []
@magic_factory(
    call_button="Enter",
)
def run_analysis_widget(wells: str, galleries: int):
    well_list = wells.split(", ")
    run_gallery_parser(omero_data, userdata, well_list, galleries)


@magic_factory(
    call_button="Enter",
    segmentation={"choices": ["nucleus", "cell"]},
    crop_size={"choices": [20, 30, 50, 100]},
    cellcycle={"choices": ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]},
)
def gallery_widget(
    #viewer: "napari.viewer.Viewer",
    well: str,
    segmentation: str,
    crop_size: int,
    cellcycle: str,
    columns: int = 4,
    rows: int = 4,
    reload: bool = True,
    contour: bool = True,
    blue_channel: str = "DAPI",
    green_channel: str = "Tub",
    red_channel: str = "EdU",
):
    channels = [blue_channel, green_channel, red_channel]  # to match rgb order
    channels = [channel for channel in channels if channel != ""]
    user_data_dict = {
        "well": well,
        "segmentation": segmentation,
        "reload": reload,
        "crop_size": crop_size,
        "cellcycle": cellcycle,
        "columns": columns,
        "rows": rows,
        "contour": contour,
        "channels": channels,
    }
    UserData.set_omero_data_channel_keys(omero_data.channel_data.keys())
    UserData.reset_with_input(userdata, **user_data_dict)
    show_gallery(omero_data, userdata)


# def show_gallery(
#     channels,
#     segmentation,
#     replacement,
#     crop_size,
#     rows,
#     columns,
#     contour,
#     cellcycle,
#     block=False,
# ):
#     filtered_channels = list(filter(None, channels))
#     try:
#         images = select_channels(filtered_channels)
#         if replacement == "With" or cropped_images.cropped_regions == []:
#             generate_crops(images, segmentation, cellcycle, crop_size)
#         return plot_random_gallery(
#             channels,
#             crop_size,
#             cellcycle,
#             block,
#             contour=contour,
#             n_row=rows,
#             n_col=columns,
#         )
#     except Exception as e:  # noqa: BLE001
#         logger.error(e)
#         # Show a message box with the error message
#         msg_box = QMessageBox()
#         msg_box.setIcon(QMessageBox.Warning)
#         msg_box.setText(str(e))
#         msg_box.setWindowTitle("Error")
#         msg_box.setStandardButtons(QMessageBox.Ok)
#         msg_box.exec_()


# def select_channels(channels: list[str]) -> np.ndarray:
#     """
#     Selects the channels to be used for the gallery.
#     """
#     channel_data = {k: int(v) for k, v in viewer_data.channel_data.items()}
#     assert all(
#         item in channel_data for item in channels
#     ), "Selected channels not in data"
#     order_indices = [channel_data[channel] for channel in channels]
#     return viewer_data.images[..., order_indices]


# # prepare the images for plotting
# def generate_crops(image_data, segmentation, cellcycle, crop_size):
#     """
#     Crop regions around each segmented object in the image.
#     """
#     cropped_regions = []
#     cropped_labels = []
#     # Iterate through individual images
#     for i in range(image_data.shape[0]):
#         # Extract data and labels for the current image
#         current_data = image_data[i]
#         if segmentation == "Nuclei":
#             current_labels = viewer_data.labels[..., 0][i]
#         else:
#             current_labels = viewer_data.labels[..., 1][i]
#         current_image_id = viewer_data.image_ids[i]
#         if cellcycle == "All":
#             df = viewer_data.plate_data[
#                 (viewer_data.plate_data["image_id"] == current_image_id)
#             ]
#         else:
#             assert (
#                 cellcycle in viewer_data.plate_data.cell_cycle.unique()
#             ), "Cellcycle phase not in data"
#             df = viewer_data.plate_data[
#                 (viewer_data.plate_data["image_id"] == current_image_id)
#                 & (viewer_data.plate_data["cell_cycle"] == cellcycle)
#             ]

#         # Identify unique objects in the segmentation
#         for _, row in df.iterrows():
#             if segmentation == "Nuclei":
#                 centroid_row = row["centroid-0"]
#                 centroid_col = row["centroid-1"]
#             else:
#                 centroid_row = row["centroid-0_x"]
#                 centroid_col = row["centroid-1_x"]

#             cropped_region, cropped_label = crop_region(
#                 current_data,
#                 current_labels,
#                 centroid_row,
#                 centroid_col,
#                 crop_size,
#             )
#             # Pad the cropped region and label if necessary
#             if (
#                 cropped_region.shape[0] != crop_size
#                 or cropped_region.shape[1] != crop_size
#             ):
#                 cropped_region, cropped_label = pad_region(
#                     cropped_region, cropped_label, crop_size
#                 )
#             corrected_cropped_label = erase_masks(cropped_label.copy())
#             # Add cropped region and label to lists
#             cropped_regions.append(cropped_region)
#             cropped_labels.append(corrected_cropped_label)

#     (
#         filtered_cropped_regions,
#         filtered_cropped_labels,
#     ) = filter_and_update_cropped_images(cropped_regions, cropped_labels)
#     cropped_images.cropped_regions = filtered_cropped_regions
#     cropped_images.cropped_labels = filtered_cropped_labels


# def filter_and_update_cropped_images(cropped_regions, cropped_labels):
#     """
#     Filters out cropped regions and labels where the label is effectively empty
#     and updates the global storage with the filtered lists.
#     """
#     filtered_cropped_regions = []
#     filtered_cropped_labels = []

#     # Iterate over cropped labels and corresponding regions
#     for region, label in zip(cropped_regions, cropped_labels):
#         if np.any(label):  # Check if there's any mask data in the label
#             filtered_cropped_regions.append(region)
#             filtered_cropped_labels.append(label)
#     return filtered_cropped_regions, filtered_cropped_labels


# def crop_region(
#     current_data, current_labels, centroid_row, centroid_col, crop_size
# ):
#     """
#     Crop a region around the centroid of a segmented object.
#     """
#     # Calculate crop coordinates
#     crop_row_start = int(max(0, centroid_row - crop_size // 2))
#     crop_row_end = int(
#         min(current_data.shape[0], centroid_row + crop_size // 2)
#     )
#     crop_col_start = int(max(0, centroid_col - crop_size // 2))
#     crop_col_end = int(
#         min(current_data.shape[1], centroid_col + crop_size // 2)
#     )
#     # Crop the region from the image
#     cropped_region = current_data[
#         crop_row_start:crop_row_end, crop_col_start:crop_col_end, :
#     ]
#     # Crop the corresponding region from the segmentation labels
#     cropped_label = current_labels[
#         crop_row_start:crop_row_end, crop_col_start:crop_col_end
#     ]
#     return cropped_region, cropped_label


# def pad_region(cropped_region, cropped_label, crop_size):
#     """
#     Pad the cropped region and label to the desired crop size."""

#     pad_row = crop_size - cropped_region.shape[0]
#     pad_col = crop_size - cropped_region.shape[1]
#     cropped_region = np.pad(
#         cropped_region,
#         ((0, pad_row), (0, pad_col), (0, 0)),
#         mode="constant",
#     )
#     cropped_label = np.pad(
#         cropped_label,
#         ((0, pad_row), (0, pad_col)),
#         mode="constant",
#     )
#     return cropped_region, cropped_label


# def erase_masks(cropped_label):
#     """
#     Erases all masks in the cropped_label that do not correspond to the center.

#     Parameters:
#         cropped_label (numpy.ndarray): The cropped label image containing possibly multiple masks.

#     Returns:
#         numpy.ndarray: The modified cropped_label with only the mask corresponding to the center.
#     """

#     center_row, center_col = np.array(cropped_label.shape) // 2

#     unique_labels = np.unique(cropped_label)
#     for unique_label in unique_labels:
#         if unique_label == 0:  # Skip background
#             continue

#         binary_mask = cropped_label == unique_label

#         if np.sum(binary_mask) == 0:  # Check for empty masks
#             continue

#         label_props = regionprops(label(binary_mask))

#         if len(label_props) == 1:
#             cropped_centroid_row, cropped_centroid_col = label_props[
#                 0
#             ].centroid

#             # Using a small tolerance value for comparing centroids
#             tol = 10

#             if (
#                 abs(cropped_centroid_row - center_row) > tol
#                 or abs(cropped_centroid_col - center_col) > tol
#             ):
#                 cropped_label[binary_mask] = 0

#     return cropped_label


# def choose_random_images(
#     cropped_images: List[np.array], n_row: int, n_col: int
# ) -> Tuple[List[np.array]]:
#     sample_size = min(n_row * n_col, len(cropped_images.cropped_regions))
#     chosen_indices = random.sample(
#         range(len(cropped_images.cropped_regions)), sample_size
#     )

#     chosen_cells = [cropped_images.cropped_regions[i] for i in chosen_indices]
#     chosen_labels = [cropped_images.cropped_labels[i] for i in chosen_indices]

#     # check for identical arrays
#     if check_identical_arrays(chosen_cells):
#         raise ValueError(
#             "There are identical arrays in the chosen cells. Please try again."
#         )
#     else:
#         logger.debug("No identical arrays found in the chosen cells.")
#     # Remove displayed cells and labels from cropped images
#     cropped_images.cropped_regions = [
#         item
#         for index, item in enumerate(cropped_images.cropped_regions)
#         if index not in chosen_indices
#     ]
#     cropped_images.cropped_labels = [
#         item
#         for index, item in enumerate(cropped_images.cropped_labels)
#         if index not in chosen_indices
#     ]
#     masked_chosen_cells = apply_mask_to_images(chosen_cells, chosen_labels)

#     return masked_chosen_cells, chosen_labels


# def check_identical_arrays(arr_list):
#     # Check each array with every other array in the list
#     n = len(arr_list)
#     for i in range(n):
#         for j in range(
#             i + 1, n
#         ):  # Start from i+1 to avoid comparing the same array
#             if np.array_equal(arr_list[i], arr_list[j]):
#                 return True  # Found identical arrays
#     return False  # No identical arrays found


# def apply_mask_to_images(images, masks):
#     """
#     Nullify pixels in color images that don't overlap with the corresponding masks.
#     Images are expected to be in the shape of (H, W, 3) and masks in the shape of (H, W).

#     :param images: List of numpy arrays, each representing a color image (H, W, 3).
#     :param masks: List of numpy arrays, each representing a mask (H, W).
#     :return: List of numpy arrays, images after applying masks.
#     """
#     masked_images = []

#     for image, mask in zip(images, masks):
#         # Ensure the mask is expanded to match the image's 3 channels
#         expanded_mask = (
#             np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2) > 0
#         )
#         # Apply the expanded mask to the image
#         masked_image = np.where(expanded_mask, image, 0)
#         masked_images.append(masked_image)

#     return masked_images


# # Plot the gallery


# def prepare_images(
#     chosen_cells: List[np.array],
#     chosen_labels: List[np.array],
#     channels: dict[str, int],
#     crop_size: int,
#     contour: bool,
# ) -> List[np.array]:
#     prepared_cells = [
#         fill_missing_channels(cell, channels, crop_size)
#         for cell in chosen_cells
#     ]
#     if contour:
#         prepared_cells = [
#             draw_contours(cell, label)
#             for cell, label in zip(prepared_cells, chosen_labels)
#         ]
#     return prepared_cells


# def calculate_dynamic_padding(
#     img_height: int, img_width: int
# ) -> Tuple[int, int]:
#     padding_height = int(img_height * 0.02)
#     padding_width = int(img_width * 0.02)
#     return padding_height, padding_width


# def create_gallery_image(
#     prepared_cells: List[np.array],
#     n_row: int,
#     n_col: int,
#     padding_height: int,
#     padding_width: int,
# ) -> np.array:
#     img_height, img_width, img_channels = prepared_cells[0].shape

#     # Adjust gallery dimensions to include border padding
#     gallery_height = n_row * img_height + (n_row + 1) * padding_height
#     gallery_width = n_col * img_width + (n_col + 1) * padding_width

#     # Create an array filled with 1.0 (white background)
#     gallery_image = np.full(
#         (gallery_height, gallery_width, img_channels),
#         fill_value=1.0,
#         dtype=np.float64,
#     )

#     for row in range(n_row):
#         for col in range(n_col):
#             idx = row * n_col + col
#             if idx >= len(prepared_cells):
#                 break
#             # Adjust start positions to account for the border padding
#             start_row = row * (img_height + padding_height) + padding_height
#             end_row = start_row + img_height
#             start_col = col * (img_width + padding_width) + padding_width
#             end_col = start_col + img_width
#             gallery_image[
#                 start_row:end_row, start_col:end_col, :
#             ] = prepared_cells[idx]

#     return gallery_image


# def add_scale_bar(
#     ax, gallery_width: int, gallery_height: int, channels, crop_size
# ):
#     physical_scale_bar_length = 10 if crop_size <= 30 else 25  # in microns
#     scale_bar_length_in_pixels = int(
#         physical_scale_bar_length / viewer_data.pixel_size[0]
#     )
#     scale_bar_length_in_pixels = int(
#         physical_scale_bar_length / viewer_data.pixel_size[0]
#     )
#     bar_height = 1
#     start_x = gallery_width - scale_bar_length_in_pixels - gallery_width * 0.04
#     start_y = gallery_height - bar_height - gallery_width * 0.01
#     color = "black" if channels.count("") == 2 else "white"
#     scale_bar = patches.Rectangle(
#         (start_x, start_y),
#         scale_bar_length_in_pixels,
#         bar_height,
#         linewidth=1,
#         edgecolor=color,
#         facecolor=color,
#     )
#     ax.add_patch(scale_bar)
#     label_x = start_x + scale_bar_length_in_pixels / 2
#     label_y = start_y - 0.5
#     ax.text(
#         label_x,
#         label_y,
#         f"{physical_scale_bar_length} µm",
#         color=color,
#         ha="center",
#         va="bottom",
#         fontsize=12,
#     )


# def plot_gallery(
#     gallery_image, channels, viewer_data, cellcycle, crop_size, block
# ):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     if channels.count("") == 2:
#         ax.imshow(gallery_image[..., 0], cmap="gray_r")
#     else:
#         ax.imshow(gallery_image)
#     metadata_str = ", ".join(
#         [f"{key}: {value}" for key, value in viewer_data.metadata.items()]
#     )
#     channel_list = [channel for channel in channels if channel != ""]
#     ax.set_title(
#         f"{viewer_data.plate_name}, well: {viewer_data.well_name}\n{metadata_str}\nchannels: {', '.join(channel_list)}, cellcycle phase: {cellcycle}",
#         fontsize=12,
#         fontweight="bold",
#     )
#     plt.axis("off")
#     # Add scale bar
#     gallery_height, gallery_width, _ = gallery_image.shape
#     add_scale_bar(ax, gallery_width, gallery_height, channels, crop_size)
#     logger.info("plotting gallery")
#     plt.show(block=block)
#     return fig


# def plot_random_gallery(
#     channels,
#     crop_size,
#     cellcycle,
#     block,
#     contour=True,
#     n_row=4,
#     n_col=4,
# ):
#     chosen_cells, chosen_labels = choose_random_images(
#         cropped_images, n_row, n_col
#     )
#     prepared_cells = prepare_images(
#         chosen_cells, chosen_labels, channels, crop_size, contour
#     )
#     img_height, img_width = prepared_cells[0].shape[:2]
#     padding_height, padding_width = calculate_dynamic_padding(
#         img_height, img_width
#     )
#     gallery_image = create_gallery_image(
#         prepared_cells, n_row, n_col, padding_height, padding_width
#     )
#     return plot_gallery(
#         gallery_image, channels, viewer_data, cellcycle, crop_size, block
#     )


# def draw_contours(img, label):
#     channel_num = img.shape[-1]
#     contours = find_contours(label, 0.5)
#     for contour in contours:
#         for coords in contour:
#             x, y = coords.astype(int)
#             img[x, y] = [1] * channel_num  # White color
#     return img


# def fill_missing_channels(img, channels, crop_size):
#     logger.debug(f"filling missing channels: {channels}")
#     logger.debug(f"image shape: {img.shape}")

#     # Initialize an empty list to hold the channel arrays
#     result_img = []
#     empty_image = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

#     # Count the number of non-empty channels
#     num_non_empty = len([ch for ch in channels if ch != ""])

#     if num_non_empty == 1:
#         # If there's only one non-empty channel, ensure it's placed at the front
#         result_img = [img[..., 0], empty_image, empty_image]
#     else:
#         # For all other cases, adhere to the order, inserting np.zeros where needed
#         img_channel_index = (
#             0  # Keep track of the index for non-empty channels in img
#         )
#         for channel in channels:
#             if channel:
#                 # Append the channel data from img according to its current non-empty index
#                 result_img.append(img[..., img_channel_index])
#                 img_channel_index += 1  # Move to the next channel in img for the next non-empty channel
#             else:
#                 result_img.append(empty_image)

#     return np.stack(result_img, axis=-1)
