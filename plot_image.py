import numpy as np
import matplotlib.pyplot as PLT
from PIL import Image

def plot_array(array, normalize=False):
    if normalize:
        image = Image.fromarray((array - np.min(array)) / (np.max(array) - np.min(array)) * 255)
    else:
        image = Image.fromarray(array)

    image.show()

def plot_array_PLT(array):
    PLT.imshow(array)
    PLT.show()

def plot_normals(normals_as_vectors):
    normals_rgb = np.asarray(normals_as_vectors*127.5 + 127.5, dtype="uint8")
    plot_array(normals_rgb)