import numpy as np
import tensorflow as tf
import find_edges
import matplotlib.pyplot as plt

import plot_image
from standard_values import *
from image_operations import rgb_to_Lab
from tensorflow.keras import layers, Model
from scipy.spatial.transform import Rotation
from numba import njit
from load_images import *
from find_planes import plot_surfaces
from image_processing_models_GPU import calculate_nearest_point_function, extract_texture_function
from sklearn.cluster import KMeans


train_indices = [109, 108, 107, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94]
number_of_prototypes = 50

def train_k_means_for_texture_vectors():
    texture_model = extract_texture_function()
    vectors = []
    for index in train_indices:
        print(index)
        _, rgb, _ = load_image(index)
        vectors.append(np.reshape(texture_model(rgb), (19200, 256)))
    vectors = np.vstack(vectors)
    k_means = KMeans(n_clusters=number_of_prototypes, verbose=1, n_init=5)
    k_means.fit(vectors)
    np.save("cluster_centers.npy", k_means.cluster_centers_)
    prediction = k_means.predict(vectors[:19200])
    prediction = np.reshape(prediction, (120, 160))
    plt.imshow(prediction, cmap='nipy_spectral')
    plt.show()

def test_clusters():
    centers = np.load("cluster_centers.npy")
    k_means = KMeans(n_clusters=number_of_prototypes, n_init=1, max_iter=1)
    _, rgb, _ = load_image(110)
    texture_model = extract_texture_function()
    vec = np.reshape(texture_model(rgb), (19200, 256))
    k_means.fit(vec)
    k_means.cluster_centers_ = centers
    prediction = k_means.predict(vec)
    print(np.shape(prediction))
    prediction = np.reshape(prediction, (120, 160))
    plt.imshow(prediction, cmap='nipy_spectral')
    plt.show()

def texture_image(rgb, texture_model):
    centers = np.load("cluster_centers.npy")
    k_means = KMeans(n_clusters=number_of_prototypes, n_init=1, max_iter=1)
    vec = np.reshape(texture_model(rgb), (19200, 256))
    k_means.fit(vec)
    k_means.cluster_centers_ = centers
    prediction = k_means.predict(vec)
    prediction = np.reshape(prediction, (120, 160))
    return prediction

@njit()
def determine_neighbors_with_border_centers_calc(surfaces, number_of_surfaces, depth_edges):
    height, width = np.shape(surfaces)
    neighbors = np.zeros((number_of_surfaces, number_of_surfaces))
    border_centers = np.zeros((number_of_surfaces, number_of_surfaces, 2))
    new_surfaces = surfaces.copy() * (np.ones_like(depth_edges) - depth_edges)
    for y in range(height):
        for x in range(width):
            if depth_edges[y][x] == 1 or surfaces[y][x] == 0:
                continue
            index = new_surfaces[y][x]
            if y < height - 1:
                other_index = new_surfaces[y + 1][x]
                neighbors[index][other_index] += 1
                neighbors[other_index][index] += 1
                positions = np.asarray([2*x+1, 2*y], dtype="float64")
                border_centers[index][other_index] += positions
                border_centers[other_index][index] += positions
            if x < width - 1:
                other_index = new_surfaces[y][x + 1]
                neighbors[index][other_index] += 1
                neighbors[other_index][index] += 1
                positions = np.asarray([2 * x, 2 * y + 1], dtype="float64")
                border_centers[index][other_index] += positions
                border_centers[other_index][index] += positions
    neighbors[0] = 0
    neighbors[:, 0] = 0
    border_centers[0] = 0
    border_centers[:, 0] = 0
    for i in range(number_of_surfaces):
        neighbors[i][i] = 0
        border_centers[i][i] = 0
        for j in range(i+1, number_of_surfaces):
            if neighbors[i][j] < 5:
                neighbors[i][j] = 0
                neighbors[j][i] = 0
    border_centers = np.divide(border_centers, 2*np.expand_dims(neighbors, axis=-1))

    neighbors_list = []
    for i in range(number_of_surfaces):
        current_list = []
        for j in range(1, number_of_surfaces):
            if neighbors[i][j] > 0:
                current_list.append(j)

        neighbors_list.append(current_list)

    return neighbors_list, border_centers

def determine_centroids(depth_image, positions, average_positions, points_in_space):
    for i in range(len(positions)):
        del positions[i][0]
    positions = [np.asarray(p) for p in positions]
    centroids = []
    for i in range(len(positions)):
        if len(positions[i]) == 0:
            centroids.append([-1, -1])
            continue
        point = positions[i][np.argmin(np.sum(np.square(positions[i] - average_positions[i]), axis=-1))]
        #centroids.append([point[0], point[1], depth_image[point[1]][point[0]]])
        centroids.append(points_in_space[point[0]][point[1]])
    return np.asarray(centroids)


def determine_neighbors_with_border_centers(*args):
    neighbor_list, border_centers = determine_neighbors_with_border_centers_calc(*args)
    border_centers[np.isnan(border_centers)] = 0
    return neighbor_list, border_centers


def extract_points(surface_patches, number_of_surfaces):
    points = []
    for i in range(number_of_surfaces):
        points.append([[0, 0]])

    for y in range(height):
        for x in range(width):
            points[surface_patches[y][x]].append([x, y])

    return points

def prepare_border_centers(neighbors, border_centers):
    result = []
    for i in range(len(neighbors)):
        current_list = []
        for neighbor in neighbors[i]:
            current_list.append(border_centers[i][neighbor])
        if len(current_list) == 0:
            current_list.append([0, 0])
        result.append(current_list)
    return result

@njit()
def determine_neighbors(surfaces, number_of_surfaces, depth_edges):
    height, width = np.shape(surfaces)
    neighbors = np.zeros((number_of_surfaces, number_of_surfaces))
    new_surfaces = surfaces.copy() * (np.ones_like(depth_edges) - depth_edges)
    for y in range(height):
        for x in range(width):
            if depth_edges[y][x] == 1:
                continue
            index = new_surfaces[y][x]
            if y < height - 1:
                other_index = new_surfaces[y + 1][x]
                neighbors[index][other_index] += 1
                neighbors[other_index][index] += 1
            if x < width - 1:
                other_index = new_surfaces[y][x + 1]
                neighbors[index][other_index] += 1
                neighbors[other_index][index] += 1
    for i in range(number_of_surfaces):
        for j in range(i+1, number_of_surfaces):
            if neighbors[i][j] < 5:
                neighbors[i][j] = 0
                neighbors[j][i] = 0
    neighbors_list = []
    for i in range(number_of_surfaces):
        current_list = []
        for j in range(1, number_of_surfaces):
            if neighbors[i][j] > 0:
                current_list.append(j)
        if i in current_list:
            current_list.remove(i)
        neighbors_list.append(current_list)

    return neighbors_list

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
def calc_average_position_and_counts(index_image, number_of_surfaces):
    height, width = np.shape(index_image)
    counter = np.zeros((number_of_surfaces, 1))
    positions = []
    for _ in range(number_of_surfaces):
        positions.append([[0, 0]])
    for y in range(1, height-1):
        for x in range(1, width-1):
            surface = index_image[y][x]
            counter[surface] += 1
            positions[surface].append([x, y])
    final_positions = np.zeros((number_of_surfaces, 2))
    for i in range(number_of_surfaces):
        if counter[i] > 0:
            final_positions[i] = np.sum(np.asarray(positions[i]), axis=0) / counter[i]
    for i in range(number_of_surfaces):
        del positions[i][0]
    return final_positions, counter, positions

def swap_values(vec):
    return np.asarray([vec[1], vec[0], vec[2]])

def determine_convexity_with_closest_points(angles, closest_points, neighbors, number_of_surfaces, depth_image, join_matrix, surfaces):
    convexity = np.zeros((number_of_surfaces, number_of_surfaces))
    concave = np.zeros((number_of_surfaces, number_of_surfaces))

    for i in range(number_of_surfaces):
        for j in range(len(neighbors[i])):
            surface_2 = neighbors[i][j]
            index_1 = neighbors[surface_2].index(i)
            if surface_2 < i:
                continue

            c_1 = closest_points[i][j]
            c_2 = closest_points[surface_2][index_1]

            connected = True
            direction = c_2 - c_1
            length = np.max(np.abs(direction))
            direction = direction / max(length, 1)
            height = depth_image[c_1[1]][c_1[0]]
            s = surfaces[c_1[1]][c_1[0]]
            different_surface = False
            factor = abs(np.dot(np.abs(direction), [factor_x, factor_y]) / np.sum(direction) * 6)
            for m in range(length + 1):
                point = [int(p) for p in np.round(c_1 + m * direction)]
                new_s = surfaces[point[1]][point[0]]
                if new_s != s:
                    different_surface = not different_surface
                new_height = depth_image[point[1]][point[0]]
                if np.abs(new_height - height) > factor * height:
                    if s != new_s:
                        connected = False
                        break
                height = new_height
                s = new_s

            if not connected:
                continue

            depth_1 = depth_image[c_1[1]][c_1[0]]
            depth_2 = depth_image[c_2[1]][c_2[0]]
            middle_depth = (depth_1 + depth_2) / 2
            diff = c_1 - c_2
            diff = [diff[0]*factor_x*middle_depth, -diff[1]*factor_y*middle_depth, depth_1-depth_2]
            diff = diff/np.linalg.norm(diff)
            normal_1, normal_2 = angles_to_normals(np.asarray([angles[c_1[1]][c_1[0]], angles[c_2[1]][c_2[0]]]))
            v1 = np.dot(diff, normal_1)
            v2 = np.dot(diff, normal_2)

            if v1 - v2 > 0.01:
                convexity[i][surface_2] = 1
                convexity[surface_2][i] = 1
                # direction = c_2 - c_1
                # length = np.max(np.abs(direction))
                # direction = direction / max(length, 1)
                # for m in range(length + 1):
                #     point = c_1 + m * direction
                #     surfaces[int(point[1])][int(point[0])] = 70
            else:
                concave[i][surface_2] = 1
                concave[surface_2][i] = 1
                # direction = c_2 - c_1
                # length = np.max(np.abs(direction))
                # direction = direction / max(length, 1)
                # for m in range(length + 1):
                #     point = c_1 + m * direction
                #     surfaces[int(point[1])][int(point[0])] = 0

    join_matrix = join_matrix + convexity
    join_matrix[join_matrix > 1] = 1
    return join_matrix, concave

def determine_similar_patches(texture_similarities, color_similarities, normal_similarities, thresholds, neighbors):
    similar_patches_1 = np.zeros_like(texture_similarities)
    similar_patches_1[texture_similarities < thresholds[0][0]] = 1
    similar_patches_1[color_similarities > thresholds[0][1]] = 0

    similar_patches_2 = np.zeros_like(texture_similarities)
    similar_patches_2[texture_similarities < thresholds[1][0]] = 1
    similar_patches_2[color_similarities > thresholds[1][1]] = 0

    similar_patches_3 = np.zeros_like(texture_similarities)
    similar_patches_3[color_similarities < thresholds[2][1]] = 1
    similar_patches_3[normal_similarities > thresholds[2][2]] = 0
    similar_patches_3[texture_similarities > thresholds[2][0]] = 0

    similar_patches_4 = np.zeros_like(texture_similarities)
    similar_patches_4[color_similarities < thresholds[3][1]] = 1
    similar_patches_4[normal_similarities > thresholds[3][2]] = 0
    similar_patches_4[texture_similarities > thresholds[3][0]] = 0

    for i in range(np.shape(texture_similarities)[0]):
        similar_patches_3[i][i] = 0
        similar_patches_4[i][i] = 0
        for n in neighbors[i]:
            similar_patches_3[i][n] = 0
            similar_patches_4[i][n] = 0

    return similar_patches_1, similar_patches_2, similar_patches_3, similar_patches_4

def determine_similar_neighbors(similar_patches, more_similar_patches, neighbors, join_matrix, number_of_surfaces, concave):
    similar_neighbors = np.zeros((number_of_surfaces, number_of_surfaces))
    for s_1 in range(number_of_surfaces):
        for s_2 in range(s_1 + 1, number_of_surfaces):
            if s_2 in neighbors[s_1]:
                if similar_patches[s_1][s_2] == 1 and concave[s_1][s_2] == 0:
                    similar_neighbors[s_1][s_2] = 1
                    similar_neighbors[s_2][s_1] = 1
                elif more_similar_patches[s_1][s_2]:
                    similar_neighbors[s_1][s_2] = 1
                    similar_neighbors[s_2][s_1] = 1

    join_matrix = join_matrix + similar_neighbors
    join_matrix[join_matrix > 1] = 1
    return join_matrix

def join_surfaces_according_to_join_matrix(join_matrix, surfaces, number_of_surfaces):
    labels = np.arange(number_of_surfaces)
    for i in range(number_of_surfaces):
        for j in range(i + 1, number_of_surfaces):
            if join_matrix[i][j] == 1:
                surfaces[surfaces == labels[j]] = labels[i]
                labels[j] = labels[i]
    return surfaces, labels

def color_and_angle_histograms(lab_image, angle_image, surfaces, number_of_surfaces, pixels_per_surface, texture_image, patches):
    histograms_color = np.zeros((256, 256, number_of_surfaces))
    histograms_angles = np.zeros((40, 40, number_of_surfaces))
    histograms_texture = np.zeros((number_of_surfaces, 256))
    normalized_angles = np.asarray((angle_image / (2*np.pi) + 0.5) * 39, dtype="uint8")
    inverse_pixels = 1 / pixels_per_surface * 10
    height, width = np.shape(surfaces)
    counts = np.zeros(number_of_surfaces)
    for y in range(height):
        for x in range(width):

            s = surfaces[y][x]
            if s < 0.0001:
                continue

            a_1, a_2 = normalized_angles[y][x]
            l, a, b = np.round(lab_image[y][x] + 127)

            histograms_angles[a_1][a_2][s] += inverse_pixels[s]
            histograms_color[int(a)][int(b)][s] += inverse_pixels[s]

            if y % 4 == 0 and x % 4 == 0:
                s = patches[y][x]
                if s < 0.0001:
                    continue
                new_x = int(x/4)
                new_y = int(y/4)
                texture = texture_image[new_y][new_x]
                histograms_texture[s] += texture
                counts[s] += 1
    histograms_texture = histograms_texture / np.expand_dims(counts, axis=-1)
    histograms_texture[np.isnan(histograms_texture)] = 0

    histograms_texture = histograms_texture/np.sum(histograms_texture, axis=0, keepdims=True) * 10
    histograms_texture[np.isnan(histograms_texture)] = 0
    return histograms_color, histograms_angles, histograms_texture

def chi_squared_distances_model(pool_size=(5, 5), strides=(2, 2)):
    input = layers.Input(shape=(None, None, None))
    number_of_surfaces = tf.shape(input)[-1]
    pool = layers.AveragePooling2D(pool_size=pool_size, strides=strides)(input)
    pool = tf.transpose(pool, (0, 3, 1, 2))
    expansion_1 = tf.repeat(tf.expand_dims(pool, axis=1), repeats=[number_of_surfaces], axis=1)
    expansion_2 = tf.repeat(tf.expand_dims(pool, axis=2), repeats=[number_of_surfaces], axis=2)
    squared_difference = tf.square(tf.subtract(expansion_1, expansion_2))
    addition = tf.add(expansion_1, expansion_2)
    distance = tf.reduce_sum(tf.reduce_sum(tf.math.divide_no_nan(squared_difference, addition), axis=-1), axis=-1)
    model = Model(inputs=[input], outputs=distance[0])
    return lambda histogram: model.predict(np.asarray([histogram]))

def chi_squared_distances_model_1D():
    input = layers.Input(shape=(None, None))
    number_of_surfaces = tf.shape(input)[-1]
    transpose = tf.transpose(input, (0, 2, 1))
    expansion_1 = tf.repeat(tf.expand_dims(transpose, axis=1), repeats=[number_of_surfaces], axis=1)
    expansion_2 = tf.repeat(tf.expand_dims(transpose, axis=2), repeats=[number_of_surfaces], axis=2)
    squared_difference = tf.square(tf.subtract(expansion_1, expansion_2))
    addition = tf.add(expansion_1, expansion_2)
    distance = tf.reduce_sum(tf.math.divide_no_nan(squared_difference, addition), axis=-1)
    model = Model(inputs=[input], outputs=distance[0])
    return lambda histogram: model.predict(np.asarray([histogram]))


def determine_occlusion_line_points(similarities, number_of_surfaces, centroids, surface_points):
    centroid_inputs = []
    points_inputs = []

    for i in range(number_of_surfaces):
        current_list = []
        for j in range(number_of_surfaces):
            if similarities[i][j] > 0:
                current_list.append(centroids[j])
        if len(current_list) > 0:
            points_inputs.append(surface_points[i])
            centroid_inputs.append(current_list)

    return points_inputs, centroid_inputs

def determine_occlusion(candidates, closest_points, surfaces, number_of_surfaces, depth_image, join_matrix, relabeling):
    closest_points_array = np.zeros((number_of_surfaces, number_of_surfaces, 2), dtype="int32")
    new_surfaces = surfaces.copy()
    number_of_candidates = np.sum(candidates, axis=1)

    index_1 = 0
    for i in range(number_of_surfaces):
        if number_of_candidates[i] == 0:
            continue
        index_2 = 0
        for j in range(number_of_surfaces):
            if candidates[i][j] == 0:
                continue
            closest_points_array[i][j] = closest_points[index_1][index_2]
            index_2 += 1
        index_1 += 1

    for i in range(number_of_surfaces):
        for j in range(i+1, number_of_surfaces):
            if candidates[i][j] == 0:
                continue
            l_1 = relabeling[i]
            l_2 = relabeling[j]
            if l_1 == l_2:
                continue
            p_1 = closest_points_array[i][j]
            p_2 = closest_points_array[j][i]

            # direction = p_2 - p_1
            # length = np.max(np.abs(direction))
            # direction = direction / max(length, 1)
            # for m in range(int(length + 1)):
            #     point = p_1 + m * direction
            #     surfaces[int(point[1])][int(point[0])] = 70
            # continue

            dif = p_2 - p_1
            length = np.max(np.abs(dif))
            direction = dif / length

            positions = np.swapaxes(np.expand_dims(p_1, axis=-1) + np.ndarray.astype(np.round(np.expand_dims(direction, axis=-1) *
                                    (lambda x: np.stack([x, x]))(np.arange(1, int(np.floor(length))))), dtype="int32"), axis1=0, axis2=1)
            pos_1 = p_1.copy()
            pos_2 = p_2.copy()
            index_1 = 0
            index_2 = len(positions)
            index = -1
            other_index = False
            for p in positions:
                index += 1
                s = surfaces[p[1]][p[0]]
                l = relabeling[s]
                if s == i:
                    index_1 = index
                    pos_1 = p.copy()
                    continue
                elif s == j:
                    index_2 = index
                    pos_2 = p.copy()
                    break
                elif l == l_1 or l == l_2:
                    other_index = False
                    break
                elif s != 0:
                    other_index = True
            if not other_index:
                continue

            d_1 = depth_image[pos_1[1]][pos_1[0]]
            d_2 = depth_image[pos_2[1]][pos_2[0]]
            d_dif = d_2 - d_1

            occluded = True
            index_dif = 1 / (index_2 - index_1) * d_dif

            for k in range(index_2 - index_1 - 1):
                p = positions[index_1 + k + 1]
                if depth_image[p[1]][p[0]] > k * index_dif + d_1:
                    occluded = False
                    break
            if occluded:
                join_matrix[i][j] = 1
                join_matrix[j][i] = 1

            # for pos1, pos2 in positions:
            #     if occluded:
            #         new_surfaces[pos2][pos1] = 0
            #     else:
            #         new_surfaces[pos2][pos1] = 30
    return join_matrix, new_surfaces

#@njit()
def find_even_planes_and_most_common_normal(angle_histogram, number_of_surfaces):
    l = np.shape(angle_histogram)[1]
    planes = np.zeros(number_of_surfaces)
    normals = np.zeros((number_of_surfaces, 2), dtype="float32")
    for i in range(number_of_surfaces):
        indices = [[0, 0]]
        for j in range(l):
            for k in range(l):
                if angle_histogram[i][j][k] > 0.7:
                    indices.append([j, k])
        del indices[0]
        if len(indices) == 0:
            continue
        indices_array = np.asarray(indices)
        center = np.expand_dims(np.sum(indices_array, axis=0), axis=0) / len(indices)
        if np.max(np.sqrt(np.sum(np.square(np.subtract(indices_array, center)), axis=-1))) < 3:
            planes[i] = 1
            n_1, n_2 = 0, 0
            count = 0
            for index in indices:
                weight = angle_histogram[i][index[0]][index[1]]
                n_1 += ((index[0] / 39) - 0.5) * (2 * np.pi) * weight
                n_2 += ((index[1] / 39) - 0.5) * (2 * np.pi) * weight
                count += weight
            n_1 = n_1/count
            n_2 = n_2/count
            normals[i] = np.asarray([n_1, n_2])
    return planes, normals

def determine_coplanarity(similarities, angle_similarities, number_of_surfaces, centroids, angle_histogram):
    planes, normals = find_even_planes_and_most_common_normal(np.moveaxis(angle_histogram, 2, 0), number_of_surfaces)
    normal_vectors = angles_to_normals(normals)
    coplanarity = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(number_of_surfaces):
        if planes[i] > 0:
            for j in range(i+1, number_of_surfaces):
                if planes[j] > 0 and similarities[i][j] > 0:
                    diff = (lambda x: x/np.linalg.norm(x))(centroids[i] - centroids[j])
                    if angle_similarities[i][j] < 1 and np.abs(np.dot(diff, normal_vectors[i])) < 0.1 and np.abs(np.dot(diff, normal_vectors[j])) < 0.1:
                        coplanarity[i][j] = 1
                        coplanarity[j][i] = 1
    return coplanarity

def extract_information(rgb_image, texture_model, surfaces, patches, number_of_surfaces, normal_angles, lab_image, depth_image, points_in_space):
    average_positions, counts, positions = calc_average_position_and_counts(surfaces, number_of_surfaces)
    centroids = determine_centroids(depth_image, positions, average_positions, points_in_space)
    texture = texture_image(rgb_image, texture_model)
    histogram_color, histogram_angles, histogram_texture = color_and_angle_histograms(lab_image, normal_angles,
                                                                                      surfaces, number_of_surfaces,
                                                                                      counts, texture, patches)
    depth_edges = find_edges.find_edges_from_depth_image(depth_image)
    return average_positions, histogram_color, histogram_angles, histogram_texture, depth_edges, centroids

def determine_convexly_connected_surfaces(nearest_points_func, surface_patch_points, neighbors, border_centers, normal_angles, number_of_surfaces, depth_image, join_matrix, surfaces):
    nearest_points = nearest_points_func(surface_patch_points, prepare_border_centers(neighbors, border_centers))
    nearest_points = [np.asarray(p.numpy(), dtype="int32") for p in nearest_points]
    join_matrix = determine_convexity_with_closest_points(normal_angles, nearest_points, neighbors, number_of_surfaces,
                                                          depth_image, join_matrix, surfaces)
    return join_matrix

def determine_occluded_patches(nearest_points_func, similar_patches, coplanarity, number_of_surfaces, positions, surface_patch_points, surfaces, depth_image, join_matrix, relabeling):
    candidates = similar_patches + coplanarity
    candidates[candidates > 1] = 1
    input = determine_occlusion_line_points(candidates, number_of_surfaces, positions, surface_patch_points)
    if len(input[0]) == 0: return join_matrix, surfaces
    nearest_points_for_occlusion = nearest_points_func(*input)
    nearest_points_for_occlusion = [np.asarray(p.numpy(), dtype="int32") for p in nearest_points_for_occlusion]

    join_matrix, new_surfaces = determine_occlusion(candidates, nearest_points_for_occlusion, surfaces, number_of_surfaces,
                                      depth_image, join_matrix, relabeling)
    return join_matrix, new_surfaces

def get_GPU_models():
    return chi_squared_distances_model((8, 8), (4, 4)), \
           chi_squared_distances_model((4, 4)), \
           chi_squared_distances_model_1D(), \
           extract_texture_function(), \
           calculate_nearest_point_function()

def texture_similarity_calc(texture_vecs, number_of_surfaces):
    vecs_1 = np.tile(np.expand_dims(texture_vecs, axis=0), [number_of_surfaces, 1, 1])
    vecs_2 = np.tile(np.expand_dims(texture_vecs, axis=1), [1, number_of_surfaces, 1])
    diff = vecs_1 - vecs_2
    return np.sqrt(np.sum(np.square(diff), axis=-1))

def assemble_surfaces(surfaces, normal_angles, rgb_image, lab_image, depth_image, number_of_surfaces, patches, models, vectors, points_in_space):
    #plot_image.plot_normals(np.reshape(angles_to_normals(np.reshape(normal_angles, (480*640, 2))),(480, 640, 3)))
    #quit()
    color_similarity_model, angle_similarity_model, texture_similarity_model, texture_model, nearest_points_func = models
    plot_surfaces(surfaces, False)
    average_positions, histogram_color, histogram_angles, histogram_texture, depth_edges, centroids \
        = extract_information(rgb_image, texture_model, surfaces, patches, number_of_surfaces, normal_angles, lab_image, depth_image, points_in_space)

    color_similarities = color_similarity_model(histogram_color)
    angle_similarities = angle_similarity_model(histogram_angles)
    texture_similarities = texture_similarity_calc(histogram_texture, number_of_surfaces)

    neighbors, border_centers = determine_neighbors_with_border_centers(surfaces, number_of_surfaces, depth_edges)
    surface_patch_points = extract_points(patches, number_of_surfaces)

    similarity_neighbors, similarity_neighbors_concave, similarity_occlusion, similarity_occlusion_coplanar\
        = determine_similar_patches(texture_similarities, color_similarities, angle_similarities, [(9, 0.6), (5, 0.4), (7, 0.5, 1.1), (12, 1.1, 1.1)], neighbors)
    coplanarity = determine_coplanarity(similarity_occlusion_coplanar, angle_similarities, number_of_surfaces, centroids, histogram_angles)

    join_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
    join_matrix, concave = determine_convexly_connected_surfaces(nearest_points_func, surface_patch_points, neighbors,
                                                                 border_centers, normal_angles, number_of_surfaces, depth_image, join_matrix, surfaces)
    join_matrix = determine_similar_neighbors(similarity_neighbors, similarity_neighbors_concave, neighbors, join_matrix, number_of_surfaces, concave)
    _, relabeling = join_surfaces_according_to_join_matrix(join_matrix, surfaces.copy(), number_of_surfaces)
    join_matrix, surfaces = determine_occluded_patches(nearest_points_func, similarity_occlusion, coplanarity, number_of_surfaces,
                                                       average_positions, surface_patch_points, surfaces, depth_image, join_matrix, relabeling)

    surfaces, _ = join_surfaces_according_to_join_matrix(join_matrix, surfaces, number_of_surfaces)
    plot_surfaces(surfaces, False)

def main():
    models = get_GPU_models()
    for index in list(range(109, 110)):
        print(index)
        index = 0
        depth, rgb, annotation = load_image(index)
        lab = rgb_to_Lab(rgb)
        Q = np.load(f"out/{index}/Q.npy")
        depth_image = np.load(f"out/{index}/depth.npy")
        angles = np.load(f"out/{index}/angles.npy")
        patches = np.load(f"out/{index}/patches.npy")
        vectors = np.load(f"out/{index}/vectors.npy")
        points_in_space = np.load(f"out/{index}/points.npy")
        Q = np.argmax(Q, axis=-1)
        number_of_surfaces = int(np.max(Q) + 1)
        assemble_surfaces(Q, angles, rgb, lab, depth_image, number_of_surfaces, patches, models, vectors, points_in_space)
    quit()

if __name__ == '__main__':
    #train_k_means_for_texture_vectors()
    #test_clusters()
    #quit()
    #f = extract_features_function()
    #depth, rgb, annotation = load_image(110)
    #out = f(rgb)
    #print(np.shape(out))
    #quit()
    main()