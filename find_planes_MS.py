import seaborn as sns
from calculate_normals import *
from plot_image import *
from collections import Counter
from numba import njit
from load_images import *
from image_operations import convert_depth_image
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
    normal_image = calculate_normal_vectors_like_paper(depth_image)
    converted_depth_image = convert_depth_image(depth_image)
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
    plot_array(np.asarray(print_image*255, dtype="uint8"))

@njit()
def do_iteration_2(image, number, size):
    height, width = np.shape(image)
    new_image = np.zeros(np.shape(image))
    for i in range(size, height-size):
        for j in range(size, width-size):
            counter = 0
            for k in range(max(0, i - size), min(height, i + size + 1)):
                for l in range(max(0, j - size), min(width, j + size + 1)):
                    counter += image[k][l]
            if counter > number:
                new_image[i][j] = 1.0
    return new_image


@njit()
def find_planes_with_curvature_scores(depth_image):
    height, width = np.shape(depth_image)
    scores = np.zeros(np.shape(depth_image))
    factor_x = math.tan(viewing_angle_x / 2) * 2 / width
    factor_y = math.tan(viewing_angle_y / 2) * 2 / height
    values = []
    for y in range(height):
        print(y)
        for x in range(width):
            d = depth_image[y][x]
            if d < 0.0001:
                continue
            depth_factor_x = factor_x * d
            depth_factor_y = factor_y * d
            curvature_scores = calculate_curvature_scores(depth_image, 3, y, x, np.asarray([depth_factor_x, depth_factor_y]), d, width, height)
            values.append(np.max(curvature_scores))
            scores[y][x] = np.max(curvature_scores)
    values.sort()
    threshold = values[int(len(values)*0.65)]
    edge_image = np.zeros(np.shape(depth_image), dtype="uint8")
    for y in range(height):
        for x in range(width):
            if scores[y][x] < threshold:
                edge_image[y][x] = 1
    return edge_image

def main():
    images = load_image(110)
    depth_image = uniform_filter_without_zero(images[0], 3)
    edges = find_planes_with_curvature_scores(depth_image)
    edges = do_iteration_2(edges, 12, 2)
    plot_array(edges, normalize=True)

if __name__ == '__main__':
    main()
