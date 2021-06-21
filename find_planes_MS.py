import numpy as np
import load_images
import image_operations
import seaborn as sns
from collections import Counter
from numba import njit
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

@njit()
def extract_data_points(depth_image, normal_image):
    points = []
    height, width = np.shape(depth_image)

    coordinate_normalizer = min(width, height)
    for y in range(0, height):
        for x in range(0, width):
            points.append(
                [x / coordinate_normalizer, y / coordinate_normalizer, normal_image[y][x][0], normal_image[y][x][1], normal_image[y][x][2], depth_image[y][x]])
    return points

def find_planes_MS(depth_image, rgb_image):
    normal_image = image_operations.determine_normal_vectors_final(depth_image, rgb_image)
    converted_depth_image = image_operations.convert_depth_image(depth_image)
    data_points = np.asarray(extract_data_points(converted_depth_image, normal_image))
    ms = MeanShift(bandwidth=0.3).fit_predict(data_points)
    #ms = KMeans(n_init=1, n_clusters=15).fit_predict(data_points)
    counts = dict(Counter(ms))
    palette = np.asarray(sns.color_palette(None, len(counts)))
    height, width = np.shape(depth_image)
    print_image = np.zeros((height, width, 3))
    counter = 0
    for y in range(0, height):
        for x in range(0, width):
            print_image[y][x] = palette[ms[counter]]
            counter += 1
    load_images.plot_array(np.asarray(print_image*255, dtype="uint8"))

def main():
    images = load_images.load_image(110)
    find_planes_MS(*images)

if __name__ == '__main__':
    main()
