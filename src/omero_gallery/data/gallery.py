from omero_gallery.viewer_data_module import viewer_data
from omero_gallery._welldata_widget import _get_data
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label, find_contours
from skimage import exposure


def show_gallery():
    images = scale_img(viewer_data.images[..., [3, 1, 0]])
    cropped_regions, cropped_labels = generate_crops(images)
    plot_random_gallery(cropped_regions, cropped_labels)


def generate_crops(image_data, crop_size=50):
    """
    Crop regions around each segmented object in the image.

    Parameters:
        image_data (np.ndarray): The image data array of shape (I, H, W, C) where I is images, H is height, W is width, and C is channels.
        segmentation_labels (np.ndarray): The segmentation labels of shape (I, H, W).
        crop_size (int): The size of the cropped region.

    Returns:
        list, list: Lists of cropped image regions and cropped segmentation labels.
                   Each cropped region is an np.ndarray of shape (crop_size, crop_size, C).
    """
    cropped_regions = []
    cropped_labels = []

    # Iterate through individual images
    for i in range(image_data.shape[0]):
        # Extract data and labels for the current image
        current_data = image_data[i]
        current_labels = viewer_data.labels[..., 1][i]
        current_image_id = viewer_data.image_ids[i]
        df = viewer_data.plate_data[
            viewer_data.plate_data["image_id"] == current_image_id
        ]
        # Identify unique objects in the segmentation
        for index, row in df.iterrows():
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

    return cropped_regions, cropped_labels


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
    images, labels, contour=True, n_row=4, n_col=4, padding=5
):
    """
    Plot a gallery of randomly chosen images in a n_row x n_col grid with padding between images.

    Parameters:
        images (list): List of images to be plotted. Each image is an np.ndarray of shape (H, W, C).
        n_row (int): Number of rows in the gallery.
        n_col (int): Number of columns in the gallery.
        padding (int): Padding size between images.
    """
    # Randomly choose images
    sample_size = min(n_row * n_col, len(images))
    chosen_indices = random.sample(range(len(images)), sample_size)

    # Use the chosen indices to get the corresponding images from both lists
    chosen_cells = [images[i] for i in chosen_indices]
    chosen_labels = [labels[i] for i in chosen_indices]

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
        (gallery_height, gallery_width, 3), dtype=np.float64
    )  # 3 for RGB channels

    # Populate the gallery image with individual images
    image_idx = 0
    for row in range(n_row):
        for col in range(n_col):
            if image_idx >= len(images):
                break
            print(chosen_cells[image_idx].shape)
            start_row = row * (img_height + padding)
            end_row = start_row + img_height
            start_col = col * (img_width + padding)
            end_col = start_col + img_width
            gallery_image[
                start_row:end_row, start_col:end_col, :
            ] = chosen_cells[image_idx]
            image_idx += 1

    # Plot the gallery image
    plt.imshow(gallery_image)
    plt.axis("off")
    plt.show()


def draw_contours(img, label):
    contours = find_contours(label, 0.5)
    for contour in contours:
        for coords in contour:
            x, y = coords.astype(int)
            img[x, y] = [1, 1, 1]  # White color
    return img


def scale_img(img: np.array, percentile: tuple = (0.1, 99.9)) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))


# Get cropped regions


# Each element of `cropped_regions` is a cropped region around a segmented object
# For example, to access the first cropped region:
# first_cropped_region = cropped_regions[0]


if __name__ == "__main__":
    plate_id = "3"
    well_pos = "C2"

    def images():
        _get_data(plate_id, well_pos)

        images = scale_img(viewer_data.images[..., [3, 1, 0]])
        cropped_regions, cropped_labels = generate_crops(images)
        plot_random_gallery(cropped_regions, cropped_labels)
        # img = cropped_regions[0]
        # label = cropped_labels[0]
        # contours = find_contours(label, 0.5)
        # # Loop over contours and set pixels on the original image
        # for contour in contours:
        #     for coords in contour:
        #         x, y = coords.astype(int)
        #         img[x, y] = [1, 1, 1]  # White color
        #
        # plt.imshow(img)
        # plt.show()

        #
        # img1 = cropped_regions[0][..., 0]
        # img2 = cropped_regions[0][..., 1]
        # img3 = cropped_regions[0][..., 3]
        #
        # img = np.stack([img3, img2, img1], axis=-1)
        #
        # # Plot the overlay image
        # # plt.imshow(img1, cmap="Blues_r")
        # plt.imshow(img)
        #
        # # plt.imshow(img3, cmap="Reds_r", alpha=0.5)
        #
        # plt.axis("off")
        # plt.title("Overlay of Channel 0 (Grey) and Channel 2 (Green)")
        # plt.show()

    images()
