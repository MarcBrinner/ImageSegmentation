import seaborn as sns
import matplotlib.pyplot as plt
import time
import multiprocessing
from mean_field_update_tf import mean_field_update_model
from calculate_normals import *
from plot_image import *
from collections import Counter
from numba import njit, prange
from load_images import *
from image_operations import convert_depth_image, rgb_to_Lab
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from skimage import measure

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
def smooth_surface_calculations(depth_image):
    height, width = np.shape(depth_image)
    plane_image = np.zeros((height, width))
    factor_x = math.tan(viewing_angle_x / 2) * 2 / width
    factor_y = math.tan(viewing_angle_y / 2) * 2 / height
    threshold = 0.007003343675404672
    size = 4
    positions = [[0, size], [0, 2*size], [size, 2*size], [2*size, 2*size], [2*size, size], [2*size, 0], [size, 0], [0, 0]]
    for y in prange(height):
        #print(y)
        for x in range(width):
            d = depth_image[y][x]
            if d < 0.0001:
                continue
            depth_factor_x = factor_x * d
            depth_factor_y = factor_y * d
            curvature_scores = calculate_curvature_scores(depth_image, size, y, x, np.asarray([depth_factor_x, depth_factor_y]), d, width, height)

            for i in range(len(positions)):
                if np.max(np.abs(np.asarray([curvature_scores[p[0]][p[1]] for p in [positions[i], positions[i-1], positions[i-2], positions[i-3]]]))) < 0 or \
                    np.sum(curvature_scores) < threshold*32:
                    plane_image[y][x] = 1
    return plane_image

def find_consecutive_patches(image):
    free_indices = []
    next_index = 1
    assignments = np.zeros(np.shape(image), dtype="uint8")
    height, width = np.shape(image)
    for y in range(height):
        print(y)
        for x in range(width):
            if image[y][x] == 1:
                if y > 0 and image[y-1][x] == image[y][x]:
                    assignments[y][x] = assignments[y-1][x]
                if x > 0 and image[y][x-1] == image[y][x]:
                    if assignments[y][x] != 0:
                        other_index = assignments[y][x-1]
                        assignments[assignments == other_index] = assignments[y][x]
                    else:
                        assignments[y][x] = assignments[y][x - 1]
                if assignments[y][x] == 0:
                    if len(free_indices) > 0:
                        index = free_indices.pop()
                    else:
                        index = next_index
                        next_index += 1
                    assignments[y][x] = index
    while len(free_indices) > 0:
        next_index -= 1
        index = free_indices.pop()
        assignments[assignments == next_index] = index
    return assignments

def color_patches(image):
    palette = np.asarray(sns.color_palette(None, np.max(image)+1))
    height, width = np.shape(image)
    colored_image = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            colored_image[y][x] = palette[image[y][x]]
    return colored_image

@njit()
def remove_small_patches(index_image, surface_image, segment_count):
    height, width = np.shape(index_image)
    counter = np.zeros(segment_count+1, dtype="uint32")
    neighbors = []
    for i in range(segment_count+1):
        neighbors.append([0])
    zero_indices = set()
    for y in range(height):
        for x in range(width):
            index = index_image[y][x]
            counter[index] += 1
            if y < height - 1:
                other_index = index_image[y + 1][x]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)
            if x < width - 1:
                other_index = index_image[y][x+1]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)
            if surface_image[y][x] == 0:
                zero_indices.add(index)

    for i in range(len(neighbors)):
        new_list = list(set(list(neighbors[i])))
        new_list.remove(0)
        if i in new_list:
            new_list.remove(i)
        neighbors[i] = new_list

    relabeling = {0: 0}
    too_small = set()
    for i in range(1, segment_count+1):
        if counter[i] <= 10:
            too_small.add(i)
            neighbors[neighbors[i][0]].remove(i)

    for val in zero_indices:
        if len(neighbors[val]) == 1:
            relabeling[val] = neighbors[val][0]
        else:
            relabeling[val] = 0

    for val in too_small:
        if val not in relabeling:
            relabeling[val] = relabeling[neighbors[val][0]]

    free_indices = list(too_small) + list(zero_indices)
    new_segment_count = segment_count - len(free_indices) + 1
    free_indices = [x for x in free_indices if x < new_segment_count]
    for i in range(segment_count+1):
        if i not in relabeling:
            if i < new_segment_count:
                relabeling[i] = i
            else:
                relabeling[i] = free_indices.pop()
                for j in range(segment_count+1):
                    if j in relabeling and relabeling[j] == i:
                        relabeling[j] = relabeling[i]

    for y in range(height):
        for x in range(width):
            index_image[y][x] = relabeling[index_image[y][x]]

@njit()
def get_unary_potentials(surface_image, number_of_labels):
    height, width = np.shape(surface_image)
    unary_potentials = np.zeros((height * width, number_of_labels))
    index = 0
    for y in range(height):
        for x in range(width):
            if surface_image[y][x] == 0:
                unary_potentials[index] = np.ones(number_of_labels)/(number_of_labels-1)
                unary_potentials[index][0] = 0
            else:
                unary_potentials[index] = np.ones(number_of_labels)/(number_of_labels-1) * 0.3
                unary_potentials[index][surface_image[y][x]] += 0.7
                unary_potentials[index][0] = 0
            index += 1
    return unary_potentials

@njit()
def extract_features(depth_image, lab_image):
    angle_image = calculate_normals_as_angles_final(depth_image)
    height, width = np.shape(depth_image)
    features_1 = np.zeros((height * width, 2))
    features_2 = np.zeros((height * width, 3))
    features_3 = np.zeros((height * width, 5))
    features_4 = np.zeros((height * width, 5))
    for y in range(height):
        for x in range(width):
            features_1[y*width + x] = np.asarray([y, x])
            features_2[y*width + x] = np.asarray([y, x, depth_image[y][x]])
            features_3[y*width + x] = np.asarray([y, x, lab_image[y][x][0], lab_image[y][x][1], lab_image[y][x][2]])
            features_4[y*width + x] = np.asarray([y, x, angle_image[y][x][0], angle_image[y][x][1]])
    return features_1, features_2, features_3, features_4

@njit(parallel=True)
def calc_similarities(features, feature, theta):
    return np.exp(-np.sum(np.square(np.subtract(features, feature)) * theta, axis=-1))

@njit(parallel=True)
def mean_field_update(Q, unary_potentials, features, theta_1, theta_2, theta_3, w_1, w_2, w_3, number_of_labels):
    features_1, features_2, features_3 = features
    n, _ = np.shape(features_1)
    Q_new = np.zeros((n, number_of_labels))

    for index in prange(n):
        print(index/n*100)
        similarities_1 = calc_similarities(features_1, features_1[index], theta_1)
        similarities_2 = calc_similarities(features_2, features_2[index], theta_2)
        similarities_3 = calc_similarities(features_3, features_3[index], theta_3)
        similarities_1[index] = 0
        similarities_2[index] = 0
        similarities_3[index] = 0

        label_messages = np.zeros(number_of_labels)
        combination = similarities_1 * w_1 + similarities_2 * w_2 + similarities_3 * w_3
        for l in range(number_of_labels):
            label_messages[l] = np.sum(combination * Q[l])
        label_compatibilities = np.zeros(number_of_labels)
        for label in range(number_of_labels):
            for other_label in range(number_of_labels):
                if other_label != label:
                    label_compatibilities[label] += label_messages[other_label]
        for label in range(number_of_labels):
            Q_new[index][label] = np.exp(-unary_potentials[index][label] - label_compatibilities[label])
        Q_new[index] = Q_new[index] / np.sum(Q_new[index])
    return Q_new

def mean_field_update_NN(Q, unary_potentials, features, theta_1, theta_2, theta_3, theta_4, w_1, w_2, w_3, w_4, weight, number_of_labels, batch_size = 64):
    features_1, features_2, features_3, features_4 = features
    n, _ = np.shape(features_1)

    matrix = np.ones((number_of_labels, number_of_labels)) - np.identity(number_of_labels)
    MFI_NN = mean_field_update_model(n, number_of_labels, Q, features_1, features_2, features_3, features_4, matrix, w_1, w_2, w_3, w_4, theta_1, theta_2, theta_3, theta_4, weight, batch_size)
    #test_index = 163610
    #out = MFI_NN.predict([features_1[test_index:test_index+batch_size], features_2[test_index:test_index+batch_size],
    #                      features_3[test_index:test_index+batch_size], features_4[test_index:test_index+batch_size], unary_potentials[test_index:test_index+batch_size]], batch_size=batch_size)
    #print()
    Q_new = MFI_NN.predict([features_1, features_2, features_3, features_4, unary_potentials], batch_size=batch_size)

    return Q_new

def plot_surfaces(Q):
    image = np.argmax(Q, axis=-1)
    plt.imshow(np.reshape(image, (480, 640)), cmap='nipy_spectral')
    plt.show()

def plot_surface_image(surfaces):
    plt.imshow(surfaces, cmap='nipy_spectral')
    plt.show()
    return

def find_smooth_surfaces_with_curvature_scores(depth_image):
    depth_image = gaussian_filter_with_depth_factor(depth_image, 4)
    print("Filter applied.")
    surfaces = smooth_surface_calculations(depth_image)
    print("Surface patches found.")
    surfaces = do_iteration_2(surfaces, 11, 2)
    surfaces = do_iteration_2(surfaces, 5, 1)
    print("Smoothing iterations done.")
    #indexed_surfaces = find_consecutive_patches(surfaces)
    indexed_surfaces, segment_count = measure.label(surfaces, background=-1, return_num=True)
    remove_small_patches(indexed_surfaces, surfaces, segment_count)
    print("Image cleaning done.")
    #colored_image = color_patches(indexed_surfaces)
    plot_surface_image(indexed_surfaces)
    return indexed_surfaces

def assign_all_pixels_to_surfaces(surface_image, depth_image, rgb_image, load_iteration=0):
    log_depth = convert_depth_image(depth_image)
    features = extract_features(log_depth, rgb_image)
    print("Features extracted.")
    number_of_labels = int(np.max(surface_image)+1)
    unary_potentials = get_unary_potentials(surface_image, number_of_labels)
    print("Potentials calculated.")
    #Q = np.transpose(unary_potentials)
    #print(np.shape(Q))
    if load_iteration >= 1:
        Q = np.load(f"it_{load_iteration}.npy")
    else:
        Q = unary_potentials
    new_Q = mean_field_update_NN(Q, unary_potentials, features, np.asarray([[1/25, 1/25]]), np.asarray([[1/80, 1/80, 100]]),
                                 np.asarray([[1/100, 1/100, 1/100, 1/100, 1/100]]), np.asarray([[1/200, 1/200, 11, 11]]), 0.05, 0.2, 1, 1.5, 0.01, number_of_labels)#0.0001
    np.save(f"it_{load_iteration+1}.npy", new_Q)
    print("Mean field update 1 done.")
    plot_surfaces(new_Q)

def find_surfaces(depth_image, rgb_image):
    surfaces = find_smooth_surfaces_with_curvature_scores(depth_image)
    assign_all_pixels_to_surfaces(surfaces, depth_image, rgb_image)
    return surfaces

def main():
    images = load_image(110)

    #plt.imshow(images[1])
    #plt.show()
    #quit()
    surfaces = find_surfaces(*images)
    #plot_array(surfaces, normalize=True)

if __name__ == '__main__':
    #Q = np.load("it_1.npy")
    #print(np.sum(Q < 700) / np.sum(Q > 0))
    #print(np.min(Q))
    #quit()
    main()
