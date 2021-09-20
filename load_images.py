import numpy as np
import os
import re
from PIL import Image

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def load_image(index):
    files = os.listdir("data/disparity")
    files = sorted_alphanumeric(files)
    file = files[index]
    depth_image = Image.open(f"data/disparity/{file}")
    RGB_image = Image.open(f"data/RGB/{file}")
    annotation = Image.open(f"data/annotation/{file}")
    return np.asarray(depth_image), np.asarray(RGB_image), np.asarray(annotation)

if __name__ == '__main__':
    load_image(0)