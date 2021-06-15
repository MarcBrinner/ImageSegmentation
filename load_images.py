import numpy as np
import os
import re
from PIL import Image

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def load_image(index):
    files = os.listdir("Data/disparity")
    files = sorted_alphanumeric(files)
    file = files[index]
    depth_image = Image.open(f"Data/disparity/{file}")
    RGB_image = Image.open(f"Data/RGB/{file}")
    return np.asarray(depth_image), np.asarray(RGB_image)

def plot_array(array, normalize=False):
    plot_array = np.copy(array)
    if normalize:
        plot_array = plot_array + np.min(plot_array)
        plot_array = np.asarray(plot_array / np.max(plot_array) * 255, dtype="unint8")

    image = Image.fromarray(plot_array)
    image.show()

if __name__ == '__main__':
    load_image(0)