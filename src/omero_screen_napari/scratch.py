import matplotlib.pyplot as plt
import numpy as np


def load_imagedata(file_path):
    return np.load(file_path, allow_pickle=True).item()

def show_image(path, number, mask=False):

    returned_data = load_imagedata(path)
    index = 1 if mask is True else 0
    title= "mask" if mask is True else "image"
    image_array = returned_data['data'][index][number]
    plt.imshow(image_array, cmap='gray')  # 'gray_r' for inverted grayscale
    plt.axis('off')  # Hide the axis
    plt.title(f'{title} {number}')
    plt.show()

if __name__ == "__main__":
    path = "/Users/hh65/omeroscreen_trainingdata/nuclei_classifier/1907_B5_0.npy"
    returned_data = load_imagedata(path)
    print(returned_data.keys())
    print(returned_data['data'][1][2].shape)

    show_image(path, 50, mask=False)

