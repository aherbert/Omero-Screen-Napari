import numpy as np
import matplotlib.pyplot as plt
from omero_screen_napari.omero_data import OmeroData



# Constants
image_size = 1080
num_channels = 3
num_blobs = 20
random_seed = 42

# Set the random seed for reproducibility
np.random.seed(random_seed)

# Create the main image array with noise in the third channel
image = np.random.randint(0, 256, (image_size, image_size, num_channels), dtype=np.uint8)
image[:, :, 2] = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)

# Create masks
nucleus_mask = np.zeros((image_size, image_size), dtype=np.uint8)
cell_shape_mask = np.zeros((image_size, image_size), dtype=np.uint8)

# Function to add circular blobs to the image
def add_circular_blob(image, y, x, radius, channel):
    rr, cc = np.ogrid[:image_size, :image_size]
    mask = (rr - y) ** 2 + (cc - x) ** 2 <= radius ** 2
    image[mask, channel] = 255

# Generate centroids
centroids_nucleus = []
centroids_cell = []
for _ in range(num_blobs):
    y, x = np.random.randint(50, image_size - 50, size=2)
    centroids_nucleus.append((y, x))
    # Shift cell centroid slightly to ensure overlap
    shift_y, shift_x = np.random.randint(-5, 5, size=2)
    centroids_cell.append((y + shift_y, x + shift_x))

# Add blobs to channels and masks
for (y_nucleus, x_nucleus), (y_cell, x_cell) in zip(centroids_nucleus, centroids_cell):
    # Nucleus
    add_circular_blob(image, y_nucleus, x_nucleus, 20, 0)
    nucleus_mask[y_nucleus-20:y_nucleus+20, x_nucleus-20:x_nucleus+20] = 1
    
    # Cell Shape
    add_circular_blob(image, y_cell, x_cell, 40, 1)
    cell_shape_mask[y_cell-40:y_cell+40, x_cell-40:x_cell+40] = 1

# Combine masks into a single mask array
masks = np.stack((nucleus_mask, cell_shape_mask), axis=-1)

# # Display the generated image and masks
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 3, 1)
# plt.title("Generated Image")
# plt.imshow(image)

# plt.subplot(1, 3, 2)
# plt.title("Nucleus Mask")
# plt.imshow(nucleus_mask, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title("Cell Shape Mask")
# plt.imshow(cell_shape_mask, cmap='gray')

# plt.show()

# Output the centroids for further use
print("Nucleus Centroids:")
print(centroids_nucleus)
print("\nCell Shape Centroids:")
print(centroids_cell)

omero_data = OmeroData()
