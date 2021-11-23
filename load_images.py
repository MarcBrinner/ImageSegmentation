import numpy as np
import os
import re

import plot_image
import utils
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


def load_image_and_surface_information(index):
    try:
        data = {}
        data["depth"], data["rgb"], data["annotation"] = load_image(index)
        data["lab"] = utils.rgb_to_Lab(data["rgb"])
        data["surfaces"] = np.load(f"out/{index}/Q.npy")
        data["depth"] = np.load(f"out/{index}/depth.npy")
        data["angles"] = np.load(f"out/{index}/angles.npy")
        data["patches"] = np.load(f"out/{index}/patches.npy")
        data["points_3d"] = np.load(f"out/{index}/points.npy")
        data["depth_edges"] = np.load(f"out/{index}/edges.npy")
        data["vectors"] = np.load(f"out/{index}/vectors.npy")
        data["num_surfaces"] = int(np.max(data["surfaces"]) + 1)
        return data
    except:
        print("The index is not available or the surfaces for this image have not detected yet."
              "Try to use the methods in \"find_surfaces.py\" first to detect and save surface data, so that this data can subsequently be loaded.")
        quit()

def load_mrcnn_predictions(image_data, index):
    data = {}
    data["final_surfaces"] = np.load(f"data/mask_rcnn predictions/{index}.npy").astype("int64")
    return data