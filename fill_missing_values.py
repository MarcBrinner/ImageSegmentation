import numpy as np
import load_images

def calculate_possible_values(image):
    missing_values = np.ma.masked_equal(0, image)
    print(missing_values)
    quit()

if __name__ == '__main__':
    image = load_images.load_image(0)
    calculate_possible_values(image)
    quit()