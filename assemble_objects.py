import numpy as np
import tensorflow as tf
import calculate_normals
import find_planes
from image_operations import rgb_to_Lab
from tensorflow.keras import layers, Model
from scipy.spatial.transform import Rotation
from numba import njit
from load_images import *
from find_planes import plot_surfaces

@njit()
def determine_neighbors(index_image, segment_count):
    height, width = np.shape(index_image)
    neighbors = []
    for i in range(segment_count+1):
        neighbors.append([0])
    for y in range(height):
        for x in range(width):
            index = index_image[y][x]
            if y < height - 1:
                other_index = index_image[y + 1][x]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)
            if x < width - 1:
                other_index = index_image[y][x + 1]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)

    for i in range(len(neighbors)):
        new_list = list(set(neighbors[i]))
        if i in new_list:
            new_list.remove(i)
        neighbors[i] = new_list

    return neighbors

def angles_to_normals(angles):
    n, _ = np.shape(angles)
    normals = np.zeros((n, 3))
    axis_1 = np.asarray([1.0, 0.0, 0.0])
    axis_2 = np.asarray([0.0, 1.0, 0.0])
    middle_vector = np.asarray([0, 0, -1])
    for i in range(n):
        vec = middle_vector.copy()
        rot = Rotation.from_rotvec(angles[i][0] * axis_1)
        vec = rot.apply(vec)
        rot = Rotation.from_rotvec(angles[i][1] * axis_2)
        vec = rot.apply(vec)
        normals[i] = vec
    return normals

@njit()
def calc_average_normal_and_position_and_counts(angle_image, index_image, number_of_surfaces):
    height, width = np.shape(index_image)
    counter = np.zeros((number_of_surfaces, 1))
    angles = np.zeros((number_of_surfaces, 2))
    positions = np.zeros((number_of_surfaces, 2))
    for y in range(height):
        for x in range(width):
            surface = index_image[y][x]
            counter[surface][0] += 1
            angles[surface] += angle_image[y][x]
            positions[surface] += np.asarray([x, y])
    div_counter = counter.copy()
    for i in range(number_of_surfaces):
        if div_counter[i] == 0:
            div_counter[i] = 1
    angles = angles/div_counter
    positions = positions/div_counter
    return angles, positions, counter

def determine_centroid(depth_image, positions):
    centroids = np.zeros((len(positions), 3), dtype=np.int32)
    for i, pos in enumerate(positions):
        discrete_y = int(pos[1])
        discrete_x = int(pos[0])
        centroids[i] = np.asarray([discrete_x, discrete_y, depth_image[discrete_y][discrete_x]], dtype="uint32")
    return centroids

def determine_convexity(normals, centroids, neighbors, number_of_surfaces):
    convexity = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(number_of_surfaces):
        for j in range(i+1, number_of_surfaces):
            if j not in neighbors[i]:
                convexity[i][j] = 0
                continue
            diff = centroids[i] - centroids[j]
            v1 = np.dot(diff, normals[i])
            v2 = np.dot(diff, normals[j])
            if v1 - v2 > 0:
                convexity[i][j] = 1
                convexity[j][i] = 1
    return convexity

def histogram_calculations(lab_image, angle_image, surfaces, number_of_surfaces, pixels_per_surface):
    histograms_color = np.zeros((256, 256, number_of_surfaces))
    histograms_angles = np.zeros((40, 40, number_of_surfaces))
    normalized_angles = np.asarray(angle_image / np.pi + 1 * 40, dtype="uint8")
    inverse_pixels = 1 / pixels_per_surface
    height, width = np.shape(surfaces)
    for y in range(height):
        for x in range(width):

            s = surfaces[y][x]
            if s < 0.0001:
                continue

            a_1, a_2 = normalized_angles[y][x]
            l, a, b = lab_image[y][x]

            histograms_angles[a_1][a_2] += inverse_pixels[s]
            histograms_color[a][b][s] += inverse_pixels[s]

    return histograms_color

def chi_squared_distances_model(pool_size=(5, 5), strides=(2, 2)):
    input = layers.Input(shape=(256, 256, None))
    number_of_surfaces = tf.shape(input)[-1]
    pool = layers.AveragePooling2D(pool_size=pool_size, strides=strides)(input)
    pool = tf.transpose(pool, (0, 3, 1, 2))
    expansion_1 = tf.repeat(tf.expand_dims(pool, axis=1), repeats=[number_of_surfaces], axis=1)
    expansion_2 = tf.repeat(tf.expand_dims(pool, axis=2), repeats=[number_of_surfaces], axis=2)
    squared_difference = tf.square(tf.subtract(expansion_1, expansion_2))
    addition = tf.add(expansion_1, expansion_2)
    distance = tf.reduce_sum(tf.reduce_sum(tf.math.divide_no_nan(squared_difference, addition), axis=-1), axis=-1)
    model = Model(inputs=[input], outputs=distance)
    return lambda histogram: model.predict(np.asarray([histogram]))

def determine_occlusion(number_of_surfaces, centroids, depth_image, surfaces):
    occlusions = np.zeros((number_of_surfaces, number_of_surfaces))
    for p_1 in range(number_of_surfaces):
        for p_2 in range(p_1+1, number_of_surfaces):
            c_1 = centroids[p_1]
            c_2 = centroids[p_2]

            dif = c_2[:2] - c_1[:2]

            length = np.max(dif)

            direction = dif/length

            positions = np.flip(np.expand_dims(c_1[:2], axis=-1) + np.ndarray.astype(np.round(np.expand_dims(direction, axis=-1) *
                                                                    (lambda x: np.stack([x, x]))(np.arange(1, int(np.floor(length))))), dtype="int32"), axis=1)

            surfaces[positions] = 1
    return surfaces

def main():
    Q, depth_image, angles = find_planes.test_model_on_image([110])
    Q = np.argmax(Q, axis=-1)
    number_of_surfaces = int(np.max(Q) + 1)
    avg_angles, avg_position, counts = calc_average_normal_and_position_and_counts(angles, Q, number_of_surfaces)
    centroids = determine_centroid(depth_image, avg_position)
    occlusions = determine_occlusion(number_of_surfaces, centroids, depth_image, Q)
    plot_surfaces(occlusions, False)

if __name__ == '__main__':
    main()