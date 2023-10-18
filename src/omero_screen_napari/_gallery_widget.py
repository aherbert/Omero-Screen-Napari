import random

import matplotlib.pyplot as plt
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container
from qtpy.QtWidgets import QMessageBox
from skimage.measure import regionprops, label, find_contours

from omero_screen_napari.viewer_data_module import viewer_data, cropped_images

def gallery_gui_widget():
    # Call the magic factories to get the widget instances
    gallery_widget_instance = gallery_widget()
    reset_widget_instance = reset_widget()
    return Container(widgets=[gallery_widget_instance, reset_widget_instance])
@magic_factory(
    call_button="Enter",
)
def reset_widget():
    cropped_images.cropped_regions = []
    cropped_images.cropped_labels = []


@magic_factory(
    call_button="Enter",
    segmentation={"choices": ["Nuclei", "Cells"]},
    replacement={"choices": ["With", "Without"]},
    crop_size={"choices": [20, 30, 50, 100]},
    cellcycle={"choices": ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]},
)
def gallery_widget(
    # viewer: "napari.viewer.Viewer",
    segmentation: str,
    replacement: str,
    crop_size: int,
    cellcycle: str,
    columns: int = 4,
    rows: int = 4,
    contour: bool = True,
    blue_channel: str = "DAPI",
    green_channel: str = "Tub",
    red_channel: str = "EdU",
):
    channels = [red_channel, green_channel, blue_channel]

    show_gallery(
        channels, segmentation, replacement, crop_size, rows, columns, contour, cellcycle
    )


def show_gallery(
    channels, segmentation, replacement, crop_size, rows, columns, contour, cellcycle
):
    filtered_channels = list(filter(None, channels))

    try:
        images = select_channels(filtered_channels)
        if replacement == "With" or cropped_images.cropped_regions == []:
            generate_crops(images, segmentation, cellcycle, crop_size)
        plot_random_gallery(
            channels,
            crop_size,
            cellcycle,
            contour=contour,
            n_row=rows,
            n_col=columns,
            padding=5,
        )
    except Exception as e:

        # Show a message box with the error message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(str(e))
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def select_channels(channels: list[str]) -> np.ndarray:
    """
    Selects the channels to be used for the gallery.
    """

    channel_data = {k: int(v) for k, v in viewer_data.channel_data.items()}
    assert all(
        item in channel_data for item in channels
    ), "Selected channels not in data"
    order_indices = [channel_data[channel] for channel in channels]
    return viewer_data.images[..., order_indices]


def generate_crops(image_data, segmentation, cellcycle, crop_size):
    """
    Crop regions around each segmented object in the image.
    """
    cropped_regions = []
    cropped_labels = []
    # Iterate through individual images
    for i in range(image_data.shape[0]):
        # Extract data and labels for the current image
        current_data = image_data[i]
        if segmentation == "Nuclei":
            current_labels = viewer_data.labels[..., 0][i]
        else:
            current_labels = viewer_data.labels[..., 1][i]
        current_image_id = viewer_data.image_ids[i]
        if cellcycle == "All":
            df = viewer_data.plate_data[
                (viewer_data.plate_data["image_id"] == current_image_id)
            ]
        else:
            assert (
                cellcycle in viewer_data.plate_data.cell_cycle.unique()
            ), "Cellcycle phase not in data"
            df = viewer_data.plate_data[
                (viewer_data.plate_data["image_id"] == current_image_id)
                & (viewer_data.plate_data["cell_cycle"] == cellcycle)
            ]

        # Identify unique objects in the segmentation
        for index, row in df.iterrows():
            if segmentation == "Nuclei":
                centroid_row = row["centroid-0"]
                centroid_col = row["centroid-1"]
            else:
                centroid_row = row["centroid-0_x"]
                centroid_col = row["centroid-1_x"]

            cropped_region, cropped_label = crop_region(
                current_data,
                current_labels,
                centroid_row,
                centroid_col,
                crop_size,
            )
            # Pad the cropped region and label if necessary
            if (
                cropped_region.shape[0] != crop_size
                or cropped_region.shape[1] != crop_size
            ):
                cropped_region, cropped_label = pad_region(
                    cropped_region, cropped_label, crop_size
                )
            corrected_cropped_label = erase_masks(cropped_label.copy())
            # Add cropped region and label to lists
            cropped_regions.append(cropped_region)
            cropped_labels.append(corrected_cropped_label)
    cropped_images.cropped_regions = cropped_regions
    cropped_images.cropped_labels = cropped_labels


def crop_region(
    current_data, current_labels, centroid_row, centroid_col, crop_size
):
    """
    Crop a region around the centroid of a segmented object.
    """
    # Calculate crop coordinates
    crop_row_start = int(max(0, centroid_row - crop_size // 2))
    crop_row_end = int(
        min(current_data.shape[0], centroid_row + crop_size // 2)
    )
    crop_col_start = int(max(0, centroid_col - crop_size // 2))
    crop_col_end = int(
        min(current_data.shape[1], centroid_col + crop_size // 2)
    )
    # Crop the region from the image
    cropped_region = current_data[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end, :
    ]
    # Crop the corresponding region from the segmentation labels
    cropped_label = current_labels[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end
    ]
    return cropped_region, cropped_label


def pad_region(cropped_region, cropped_label, crop_size):
    """
    Pad the cropped region and label to the desired crop size."""

    pad_row = crop_size - cropped_region.shape[0]
    pad_col = crop_size - cropped_region.shape[1]
    cropped_region = np.pad(
        cropped_region,
        ((0, pad_row), (0, pad_col), (0, 0)),
        mode="constant",
    )
    cropped_label = np.pad(
        cropped_label,
        ((0, pad_row), (0, pad_col)),
        mode="constant",
    )
    return cropped_region, cropped_label


def erase_masks(cropped_label):
    """
    Erases all masks in the cropped_label that do not correspond to the center.

    Parameters:
        cropped_label (numpy.ndarray): The cropped label image containing possibly multiple masks.

    Returns:
        numpy.ndarray: The modified cropped_label with only the mask corresponding to the center.
    """

    center_row, center_col = np.array(cropped_label.shape) // 2

    unique_labels = np.unique(cropped_label)
    for unique_label in unique_labels:
        if unique_label == 0:  # Skip background
            continue

        binary_mask = cropped_label == unique_label

        if np.sum(binary_mask) == 0:  # Check for empty masks
            continue

        label_props = regionprops(label(binary_mask))

        if len(label_props) == 1:
            cropped_centroid_row, cropped_centroid_col = label_props[
                0
            ].centroid

            # Using a small tolerance value for comparing centroids
            tol = 10

            if (
                abs(cropped_centroid_row - center_row) > tol
                or abs(cropped_centroid_col - center_col) > tol
            ):
                cropped_label[binary_mask] = 0

    return cropped_label


def plot_random_gallery(
    channels, crop_size, cellcycle, contour=True, n_row=4, n_col=4, padding=5
):
    """
    Plot a gallery of randomly chosen images in n_row x n_col grid with padding between images.
    """
    # Randomly choose images
    sample_size = min(n_row * n_col, len(cropped_images.cropped_regions))
    chosen_indices = random.sample(
        range(len(cropped_images.cropped_regions)), sample_size
    )

    # Use the chosen indices to get the corresponding images from both lists
    chosen_cells = [cropped_images.cropped_regions[i] for i in chosen_indices]
    chosen_cells = [
        fill_missing_channels(chosen_cell, channels, crop_size)
        for chosen_cell in chosen_cells
    ]
    chosen_labels = [cropped_images.cropped_labels[i] for i in chosen_indices]
    # delete cells, labels that are displayed from cropped images so that they wont appear again
    cropped_images.cropped_regions = [item for index, item in enumerate(cropped_images.cropped_regions) if
                                      index not in chosen_indices]
    print(f"cropped image number: {len(cropped_images.cropped_regions)}")
    cropped_images.cropped_labels = [item for index, item in enumerate(cropped_images.cropped_labels) if
                                      index not in chosen_indices]
    print(f"cropped label number: {len(cropped_images.cropped_labels)}")
    channel_num = chosen_cells[0].shape[-1]
    if contour == True:
        chosen_cells = [
            draw_contours(cell, label)
            for cell, label in zip(chosen_cells, chosen_labels)
        ]
    len(chosen_cells)
    # Get image dimensions from the first image
    img_height, img_width, img_channels = chosen_cells[0].shape

    # Create an empty array to hold the gallery image
    gallery_height = n_row * img_height + (n_row - 1) * padding
    gallery_width = n_col * img_width + (n_col - 1) * padding
    gallery_image = np.zeros(
        (gallery_height, gallery_width, channel_num), dtype=np.float64
    )  # 3 for RGB channels

    # Populate the gallery image with individual images
    image_idx = 0
    for row in range(n_row):
        for col in range(n_col):
            if image_idx >= len(cropped_images.cropped_regions):
                break
            start_row = row * (img_height + padding)
            end_row = start_row + img_height
            start_col = col * (img_width + padding)
            end_col = start_col + img_width
            gallery_image[
                start_row:end_row, start_col:end_col, :
            ] = chosen_cells[image_idx]
            image_idx += 1

    # Plot the gallery image
    fig, ax = plt.subplots(figsize=(10, 10))
    if channels.count("") == 2:
        index = next((i for i, s in enumerate(channels) if s != ""), None)
        ax.imshow(gallery_image[..., index], cmap="gray_r")
    else:
        ax.imshow(gallery_image)
    ax.set_title(
        f"{viewer_data.plate_name}, {viewer_data.well_name}, {channels}, {cellcycle}"
    )
    plt.axis("off")
    fig.resolution = 300
    plt.show(block=False)


def draw_contours(img, label):
    channel_num = img.shape[-1]
    contours = find_contours(label, 0.5)
    for contour in contours:
        for coords in contour:
            x, y = coords.astype(int)
            img[x, y] = [1] * channel_num  # White color
    return img


def fill_missing_channels(img, channels, crop_size):
    empty_image = np.zeros((crop_size, crop_size), dtype=np.int8)
    i = 0
    img_list = []
    for channel in channels:
        if channel != "":
            img_list.append(img[..., i])
            i += 1
        else:
            img_list.append(empty_image)
    return np.stack(img_list, axis=-1)
