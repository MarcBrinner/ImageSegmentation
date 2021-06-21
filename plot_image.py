import numpy as np
import matplotlib.pyplot as PLT
from PIL import Image

def plot_array(array, normalize=False):
    if normalize:
        image = Image.fromarray((plot_array - np.min(plot_array)) / (np.max(plot_array) - np.min(plot_array)) * 255)
    else:
        image = Image.fromarray(array)

    image = Image.fromarray(image)
    image.show()

def plot_array_PLT(array):
    PLT.imshow(array)
    PLT.show()