import time

import numba
import numpy as np
import tensorflow as tf
from standard_values import *
from image_operations import rgb_to_Lab
from tensorflow.keras import layers, Model
from scipy.spatial.transform import Rotation
from numba import njit
from load_images import *
from find_planes import plot_surfaces
from image_processing_models_GPU import extract_texture_function


train_indices = [109, 108, 107, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94]
number_of_surfaces = 0

@njit()
def remove_disconnected_components(surfaces, centroid_indices):
    connected_indices = np.zeros_like(surfaces)
    for i in range(len(centroid_indices)):
        if centroid_indices[i][0] == -1:
            continue
        queue = [list(centroid_indices[i])]
        while len(queue) > 0:
            y, x = queue.pop()
            connected_indices[y][x] = 1.0
            for y_2, x_2 in [[y-1, x], [y+1, x], [y, x-1], [y, x+1]]:
                if y_2 < 0 or x_2 < 0 or y_2 >= height or x_2 >= width:
                    continue
                if connected_indices[y_2][x_2] > 0 or surfaces[y_2][x_2] != i:
                    continue
                queue.append([y_2, x_2])
    for i in range(height):
        for j in range(width):
            if connected_indices[i][j] == 0:
                surfaces[i][j] = 0
    return surfaces

@njit()
def determine_neighbors_with_border_centers(surfaces, depth_edges, number_of_surfaces):
    height, width = np.shape(surfaces)
    neighbors = np.zeros((number_of_surfaces, number_of_surfaces))
    border_centers = np.zeros((number_of_surfaces, number_of_surfaces, 2))
    new_surfaces = surfaces.copy() * depth_edges
    for y in range(height):
        for x in range(width):
            if depth_edges[y][x] == 0 or surfaces[y][x] == 0:
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
        for j in range(number_of_surfaces):
            if neighbors[i][j] < 4:
                neighbors[i][j] = 1
                neighbors[j][i] = 1

    border_centers = np.divide(border_centers, 2*np.expand_dims(neighbors, axis=-1))

    neighbors_list = []
    for i in range(number_of_surfaces):
        current_list = []
        for j in range(1, number_of_surfaces):
            if neighbors[i][j] > 3:
                current_list.append(j)

        neighbors_list.append(current_list)
    return neighbors_list, border_centers

def determine_centroids(surface_patch_points, average_positions, points_in_space):
    centroids = []
    centroid_indices = []
    for i in range(len(surface_patch_points)):
        if len(surface_patch_points[i]) == 0:
            centroids.append([-1, -1])
            continue
        point = surface_patch_points[i][np.argmin(np.sum(np.square(surface_patch_points[i] - average_positions[i]), axis=-1))]
        centroids.append(points_in_space[point[1]][point[0]])
        centroid_indices.append([point[1], point[0]])
    return np.asarray(centroids), centroid_indices

@njit()
def extract_points(surface_patches, number_of_surfaces):
    points = []
    for i in range(number_of_surfaces):
        points.append([[0, 0]])

    for y in range(height):
        for x in range(width):
            points[surface_patches[y][x]].append([x, y])

    for i in range(number_of_surfaces):
        del points[i][0]
    return [np.asarray(p, dtype="int32") for p in points]

def prepare_border_centers(neighbors, border_centers):
    result = []
    for i in range(len(neighbors)):
        current_list = []
        for neighbor in neighbors[i]:
            current_list.append(border_centers[i][neighbor])
        if len(current_list) == 0:
            current_list.append(np.asarray([0, 0]))
        result.append(np.asarray(current_list, dtype="int32"))
    return result

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
    return final_positions, counter

def swap_values(vec):
    return np.asarray([vec[1], vec[0], vec[2]])

@njit()
def determine_convexity_with_above_and_below_line(normal_1, normal_2, space_point_1, space_point_2, c_1, c_2, points_in_space):
    normal_direction = normal_1 + normal_2 / 2
    normal_direction = normal_direction / np.linalg.norm(normal_direction)
    d_1 = np.dot(space_point_1, normal_direction)
    d_2 = np.dot(space_point_2, normal_direction)
    d_dif = d_2 - d_1

    direction = c_2 - c_1
    length = np.max(np.abs(direction))
    direction = direction / max(length, 1)
    above = 0
    below = 0
    index_dif = 1 / length * d_dif
    for m in range(length + 1):
        point = c_1 + m * direction
        if np.dot(points_in_space[int(point[1])][int(point[0])], normal_direction) > m * index_dif + d_1:
            above += 1
        else:
            below += 1
    return above, below

def determine_convexity_with_closest_points(angles, closest_points, neighbors, surfaces, points_in_space, coplanarity):
    convex = np.zeros((number_of_surfaces, number_of_surfaces))
    concave = np.zeros((number_of_surfaces, number_of_surfaces))
    new_surfaces = surfaces.copy()
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
            space_point = points_in_space[c_1[1]][c_1[0]]
            s = surfaces[c_1[1]][c_1[0]]
            different_surface = False
            threshold = 2.5
            for m in range(length + 1):
                point = [int(p) for p in np.round(c_1 + m * direction)]
                new_s = surfaces[point[1]][point[0]]
                if new_s != s:
                    different_surface = not different_surface
                new_space_point = points_in_space[point[1]][point[0]]
                if np.abs(new_space_point[2] - space_point[2]) > threshold*np.linalg.norm(space_point[:2] - new_space_point[:2]):
                    if s != new_s:
                        connected = False
                        break
                space_point = new_space_point
                s = new_s

            space_point_1 = points_in_space[c_1[1]][c_1[0]]
            space_point_2 = points_in_space[c_2[1]][c_2[0]]
            diff = space_point_1 - space_point_2
            diff = diff/np.linalg.norm(diff)
            normal_1, normal_2 = angles_to_normals(np.asarray([angles[c_1[1]][c_1[0]], angles[c_2[1]][c_2[0]]]))
            v1 = np.dot(diff, normal_1)
            v2 = np.dot(diff, normal_2)

            above, below = determine_convexity_with_above_and_below_line(np.asarray(normal_1, dtype="float32"), np.asarray(normal_2, dtype="float32"),
                                                                         space_point_1, space_point_2, c_1, c_2, points_in_space)

            if (v1 - v2 > 0.05 and above > below*1.3) or (v1 - v2 > 0.2 and above >= below):
                if not connected:
                    continue
                convex[i][surface_2] = 1
                convex[surface_2][i] = 1
                direction = c_2 - c_1
                length = np.max(np.abs(direction))
                direction = direction / max(length, 1)
                for m in range(length + 1):
                    point = c_1 + m * direction
                    new_surfaces[int(point[1])][int(point[0])] = 70

            elif v1 - v2 < -0.13 or (coplanarity[i][surface_2] == 0 and (above*1.6 < below or (abs(v1-v2) < 0.05 and abs(1-above/below) < 0.2))):
                concave[i][surface_2] = 1
                concave[surface_2][i] = 1
                direction = c_2 - c_1
                length = np.max(np.abs(direction))
                direction = direction / max(length, 1)
                for m in range(length + 1):
                    point = c_1 + m * direction
                    new_surfaces[int(point[1])][int(point[0])] = 0
    return convex, concave, new_surfaces

def determine_similar_patches(texture_similarities, color_similarities, normal_similarities, thresholds, neighbors, planes):
    similar_patches_1 = np.zeros_like(texture_similarities)
    similar_patches_1[texture_similarities < thresholds[0][0]] = 1
    similar_patches_1[color_similarities > thresholds[0][1]] = 0

    similar_patches_2 = np.zeros_like(texture_similarities)
    similar_patches_2[texture_similarities < thresholds[1][0]] = 1
    similar_patches_2[color_similarities > thresholds[1][1]] = 0

    similar_patches_3 = np.zeros_like(texture_similarities)
    similar_patches_3[color_similarities < thresholds[2][1]] = 1
    similar_patches_3[texture_similarities > thresholds[2][0]] = 0
    similar_patches_3[normal_similarities > thresholds[2][2]] = 0

    similar_patches_4 = np.zeros_like(texture_similarities)
    similar_patches_4[color_similarities < thresholds[3][1]] = 1
    similar_patches_4[normal_similarities > thresholds[3][2]] = 0
    similar_patches_4[texture_similarities > thresholds[3][0]] = 0

    similar_patches_5 = np.zeros_like(texture_similarities)
    similar_patches_5[color_similarities < thresholds[4][1]] = 1
    similar_patches_5[texture_similarities > thresholds[4][0]] = 0
    similar_patches_5[texture_similarities + color_similarities > thresholds[4][2]] = 0
    similar_patches_5[planes == 1, :] = 0
    similar_patches_5[:, planes == 1] = 0

    for i in range(np.shape(texture_similarities)[0]):
        similar_patches_3[i][i] = 0
        similar_patches_4[i][i] = 0
        similar_patches_5[i][i] = 0

    return similar_patches_1, similar_patches_2, similar_patches_3, similar_patches_4, similar_patches_5

#@njit()
def join_similar_neighbors(candidates_if_not_concave, candidates_if_concave, neighbors, concave):
    similar_neighbors = np.zeros((number_of_surfaces, number_of_surfaces))
    for s_1 in range(number_of_surfaces):
        for s_2 in range(s_1 + 1, number_of_surfaces):
            if s_2 in neighbors[s_1]:
                if candidates_if_not_concave[s_1][s_2] == 1 and concave[s_1][s_2] == 0:
                    similar_neighbors[s_1][s_2] = 1
                    similar_neighbors[s_2][s_1] = 1
                elif candidates_if_concave[s_1][s_2]:
                    similar_neighbors[s_1][s_2] = 1
                    similar_neighbors[s_2][s_1] = 1
    return similar_neighbors

@njit()
def join_surfaces_according_to_join_matrix(join_matrix, surfaces, number_of_surfaces):
    labels = np.arange(number_of_surfaces)
    for i in range(number_of_surfaces):
        for j in range(i + 1, number_of_surfaces):
            if join_matrix[i][j] >= 1:
                l = labels[j]
                new_l = labels[i]
                for k in range(number_of_surfaces):
                    if labels[k] == l:
                        labels[k] = new_l
    for y in range(height):
        for x in range(width):
            surfaces[y][x] = labels[surfaces[y][x]]
    return surfaces, labels

@njit()
def color_and_angle_histograms(lab_image, angle_image, surfaces, texture_image, patches, number_of_surfaces):
    histograms_color = np.zeros((256, 256, number_of_surfaces))
    histograms_angles = np.zeros((40, 40, number_of_surfaces))
    histograms_texture = np.zeros((number_of_surfaces, 256))

    normalized_angles = (angle_image / (2*np.pi) + 0.5) * 39

    counts = np.zeros(number_of_surfaces)
    all_angles_sum = np.zeros((number_of_surfaces, 2))
    patch_counter = np.zeros(number_of_surfaces)
    to_int = lambda x: (int(x[0]), int(x[1]))
    to_rounded_int = lambda x: (round(x[0]), round(x[1]), round(x[2]))
    for y in range(height):
        for x in range(width):

            s = surfaces[y][x]
            if s < 0.0001:
                continue

            l, a, b = to_rounded_int(lab_image[y][x] + 127)

            histograms_color[int(a)][int(b)][s] += 1

            s = patches[y][x]
            if s < 0.0001:
                continue

            a_1, a_2 = to_int(normalized_angles[y][x])
            histograms_angles[a_1][a_2][s] += 1
            all_angles_sum[s] += angle_image[y][x]
            patch_counter[s] += 1

            if y % 4 == 0 and x % 4 == 0:
                new_x = int(x/4)
                new_y = int(y/4)
                texture = texture_image[new_y][new_x]
                histograms_texture[s] += texture
                counts[s] += 1
    norm = np.expand_dims(np.expand_dims(np.sum(np.sum(histograms_color, axis=0), axis=0)/10, axis=0), axis=0)
    for i in range(number_of_surfaces):
        if norm[0][0][i] == 0:
            norm[0][0][i] = 1
    histograms_color = histograms_color/norm

    patch_counter[patch_counter == 0] = 1
    all_angles_sum = all_angles_sum / np.expand_dims(patch_counter, axis=-1)
    histograms_angles = histograms_angles / (patch_counter/10)

    counts[counts == 0] = 1
    histograms_texture = histograms_texture / np.expand_dims(counts, axis=-1) * 10

    return histograms_color, histograms_angles, histograms_texture, all_angles_sum

def chi_squared_distances_model(pool_size=(5, 5), strides=(2, 2)):
    input = layers.Input(shape=(None, None, None))
    number_of_surfaces = tf.shape(input)[-1]
    pool = layers.AveragePooling2D(pool_size=pool_size, strides=strides)(input)
    pool = tf.transpose(pool, (0, 3, 1, 2))
    expansion_1 = tf.repeat(tf.expand_dims(pool, axis=1), repeats=[number_of_surfaces], axis=1)
    expansion_2 = tf.repeat(tf.expand_dims(pool, axis=2), repeats=[number_of_surfaces], axis=2)
    squared_difference = tf.square(tf.subtract(expansion_1, expansion_2))
    addition = tf.add(expansion_1, expansion_2)
    distance = tf.reduce_sum(tf.reduce_sum(tf.math.divide_no_nan(squared_difference, addition), axis=-1), axis=-1)[0]
    model = Model(inputs=[input], outputs=distance)
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


def determine_occlusion_line_points(similarities, centroids, surface_points):
    centroid_inputs = []
    points_inputs = []

    for i in range(number_of_surfaces):
        current_list = []
        for j in range(number_of_surfaces):
            if similarities[i][j] > 0:
                current_list.append(centroids[j])
        if len(current_list) > 0:
            points_inputs.append(surface_points[i])
            centroid_inputs.append(np.asarray(current_list, dtype="int32"))

    return points_inputs, centroid_inputs

def depth_extend_distance(i, j, depth_extends):
    d_1 = depth_extends[i]
    d_2 = depth_extends[j]
    if d_2[0] <= d_1[0] <= d_2[1] or d_2[0] <= d_1[1] <= d_2[1] or d_1[0] <= d_2[0] <= d_1[1]:
        return 0
    if d_1[0] > d_2[1]:
        return d_1[0] - d_2[1]
    return d_2[0] - d_1[1]

def determine_occlusion(candidates, candidates_occlusion, closest_points, surfaces, points_in_space, relabeling, candidates_curved, coplanarity, norm_image, depth_extend):
    join_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
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
            index = 0
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
            if (not other_index and (candidates_curved[i][j] == 0) and coplanarity[i][j] == 0):
                continue

            d_1 = norm_image[pos_1[1]][pos_1[0]]
            d_2 = norm_image[pos_2[1]][pos_2[0]]
            d_dif = d_2 - d_1

            occluded = True

            if index_1 != index_2:
                index_dif = 1 / (index_2 - index_1) * d_dif

                for k in range(index_2 - index_1 - 1):
                    p = positions[index_1 + k + 1]
                    v_1 = np.linalg.norm(points_in_space[p[1]][p[0]])
                    v_2 = k * index_dif + d_1
                    if norm_image[p[1]][p[0]] > k * index_dif + d_1:
                        occluded = False
                        break

            if (coplanarity[i][j] == 0 and candidates_occlusion[i][j] == 0 and candidates_curved[i][j] == 1) and\
                    (abs(index_2 - index_1) > 8 or depth_extend_distance(i, j, depth_extend) > 30):
                occluded = False
            if occluded:
                join_matrix[i][j] = 1
                join_matrix[j][i] = 1

            # for pos1, pos2 in positions:
            #     if occluded:
            #         new_surfaces[pos2][pos1] = 0
            #     else:
            #         new_surfaces[pos2][pos1] = 30
    return join_matrix, new_surfaces

def nan_and_inf_to_zero(array):
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    return array

def find_even_planes(angle_histogram):
    angle_histogram_sum = np.sum(np.sum(angle_histogram, axis=-1, keepdims=True), axis=-2, keepdims=True)
    angle_histogram_sum[angle_histogram_sum == 0] = 1
    angle_histogram_norm = angle_histogram/angle_histogram_sum
    angle_histogram_norm_without_zero = angle_histogram_norm.copy()
    angle_histogram_norm_without_zero[angle_histogram_norm == 0] = 1
    entropy = -np.sum(np.sum(angle_histogram_norm * np.log(angle_histogram_norm_without_zero), axis=-1), axis=-1)
    max = np.max(np.max(angle_histogram, axis=-1), axis=-1)

    planes = np.zeros(number_of_surfaces)
    planes[max >= 1.5] = 1.0
    planes[entropy > 2] = 0.0
    planes[max >= 3] = 1.0
    return planes

@njit()
def determine_coplanarity(similarities, centroids, average_normals, planes, number_of_surfaces):
    coplanarity = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(number_of_surfaces):
        if planes[i] > 0:
            for j in range(i+1, number_of_surfaces):
                if planes[j] > 0 and similarities[i][j] > 0:
                    diff = centroids[i] - centroids[j]
                    dist = float(np.linalg.norm(diff))
                    max_arc = max(0.02, min(0.1, np.arcsin((math.sqrt(dist*2))/dist)))

                    diff = diff / dist
                    arc_1 = np.abs(np.pi/2 - np.arccos(np.dot(diff, average_normals[i])))
                    arc_2 = np.abs(np.pi/2 - np.arccos(np.dot(diff, average_normals[j])))
                    val = np.abs(1 - np.dot(average_normals[i], average_normals[j]))
                    if val < 0.008 and arc_1 < max_arc and arc_2 < max_arc:
                        coplanarity[i][j] = 1
                        coplanarity[j][i] = 1
    return coplanarity

def remove_convex_connections_after_occlusion_reasoning(join_matrix_occlusion, join_matrix_before_occlusion, similarity,
                                                        coplanarity, concave, centroids, depth_image):
    groups = []
    for i in range(number_of_surfaces):
        groups.append(set())
        groups[-1].add(i)

    for i in range(number_of_surfaces):
        for j in range(i+1, number_of_surfaces):
            if join_matrix_before_occlusion[i][j] > 0:
                groups[i].update(groups[j])
                groups[j].update(groups[i])

    pairs = []
    distances = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(number_of_surfaces):
        for j in range(i+1, number_of_surfaces):
            if join_matrix_occlusion[i][j] > 0 and join_matrix_before_occlusion[i][j] == 0:
                pairs.append((i, j))
                d = np.linalg.norm(centroids[i] - centroids[j])
                distances[i][j] = d
                distances[j][i] = d

    indices = depth_image > 0
    average_depth = np.sum(depth_image[indices]) / np.sum(np.asarray(indices, dtype="int32"))
    n_1 = average_depth * math.tan(viewing_angle_x) * 2
    n_2 = np.max(distances)*5

    joins = {}
    for i, j in pairs:
        distance_factor_1 = distances[i][j]/n_2
        distance_factor_2 = distances[i][j]/n_1
        similarity_factor = similarity[i][j]
        coplanar_factor = -0.15 if coplanarity[i][j] else 0
        join_probability_value = distance_factor_1 + distance_factor_2 + similarity_factor + coplanar_factor
        joins[join_probability_value] = (i, j)

    final_join_matrix = join_matrix_before_occlusion.copy()
    l = sorted(joins.items(), key=lambda x: float(x[0]))
    for value, (x, y) in sorted(joins.items(), key=lambda x: float(x[0])):
        if x in groups[y]:
            continue
        join = True
        for i in groups[x]:
            for j in groups[y]:
                if concave[i][j]:
                    join = False
                    break
            if not join:
                break
        if join:
            complete_set = set()
            complete_set.update(groups[x])
            complete_set.update(groups[y])
            for index in complete_set:
                groups[index].update(complete_set)
            final_join_matrix[x][y] = 1
            final_join_matrix[y][x] = 1

    return final_join_matrix

@njit()
def relabel_surfaces(surfaces, patches):
    for y in range(height):
        for x in range(width):
            if patches[y][x] != 0:
                surfaces[y][x] = patches[y][x]
    return surfaces

@njit()
def calculate_depth_extend(surfaces, norm_image, number_of_surfaces):
    values = np.zeros((number_of_surfaces, 2))
    values[:, 0] = np.inf
    values[:, 1] = -np.inf
    for y in range(height):
        for x in range(width):
            n = norm_image[y][x]
            if n == 0:
                continue
            s = surfaces[y][x]
            if n < values[s][0]:
                values[s][0] = n
            if n > values[s][1]:
                values[s][1] = n
    return values

def extract_information(rgb_image, texture_model, surfaces, patches, normal_angles, lab_image, depth_image, points_in_space, depth_edges):
    surfaces = relabel_surfaces(surfaces, patches)
    surface_patch_points = extract_points(patches, number_of_surfaces)
    average_positions, counts = calc_average_position_and_counts(surfaces, number_of_surfaces)
    centroids, centroid_indices = determine_centroids(surface_patch_points, average_positions, points_in_space)
    surfaces = remove_disconnected_components(surfaces, np.asarray(centroid_indices, dtype="int64"))
    texture = texture_model(rgb_image)
    histogram_color, histogram_angles, histogram_texture, average_normals = color_and_angle_histograms(lab_image, normal_angles,
                                                                                      surfaces, texture, patches, number_of_surfaces)

    planes = find_even_planes(np.swapaxes(histogram_angles, 2, 0))
    neighbors, border_centers = determine_neighbors_with_border_centers(surfaces, depth_edges, number_of_surfaces)
    norm_image = np.linalg.norm(points_in_space, axis=-1)
    depth_extend = calculate_depth_extend(surfaces, norm_image, number_of_surfaces)
    return average_positions, histogram_color, histogram_angles, histogram_texture, centroids, average_normals,\
           centroid_indices, surfaces, planes, surface_patch_points, neighbors, border_centers, norm_image, depth_extend

def determine_convexly_connected_surfaces(nearest_points_func, surface_patch_points, neighbors, border_centers, normal_angles, surfaces, points_in_space, coplanarity):
    nearest_points = nearest_points_func(surface_patch_points, prepare_border_centers(neighbors, border_centers))
    convex, concave, new_surfaces = determine_convexity_with_closest_points(normal_angles, nearest_points, neighbors, surfaces, points_in_space, coplanarity)
    return convex, concave, new_surfaces

def join_disconnected_patches(nearest_points_func, similar_patches_occlusion, similar_patches_curved, coplanarity,
                              positions, surface_patch_points, surfaces, points_in_space, relabeling, norm_image, depth_extend):
    candidates = similar_patches_occlusion + coplanarity + similar_patches_curved
    candidates[candidates > 1] = 1
    input = determine_occlusion_line_points(candidates, positions, surface_patch_points)
    if len(input[0]) == 0: return np.zeros((number_of_surfaces, number_of_surfaces)), surfaces
    nearest_points_for_occlusion = nearest_points_func(*input)
    join_matrix, new_surfaces = determine_occlusion(candidates, similar_patches_occlusion, nearest_points_for_occlusion, surfaces,
                                      points_in_space, relabeling, similar_patches_curved, coplanarity, norm_image, depth_extend)
    return join_matrix, new_surfaces

@njit()
def calculate_nearest_points_calc(surface_points, border_centers, max_num):
    results = np.zeros((len(border_centers), max_num, 2), dtype="int32")
    for i in range(len(surface_points)):
        diffs = np.subtract(np.expand_dims(surface_points[i], axis=0), np.expand_dims(border_centers[i], axis=1))
        diffs = np.sum(np.square(diffs), axis=-1)
        for j in range(len(diffs)):
            results[i][j] = surface_points[i][np.argmin(diffs[j])]
    return results

def calculate_nearest_points(surface_points, border_centers):
    max_num = max([len(a) for a in border_centers])
    results = calculate_nearest_points_calc(surface_points, border_centers, max_num)
    return [results[i][:len(border_centers[i])] for i in range(len(border_centers))]

def get_GPU_models():
    return chi_squared_distances_model((10, 10), (4, 4)), \
           chi_squared_distances_model((4, 4), (1, 1)), \
           chi_squared_distances_model_1D(), \
           extract_texture_function(), \
           calculate_nearest_points

def texture_similarity_calc(texture_vecs):
    vecs_1 = np.tile(np.expand_dims(texture_vecs, axis=0), [number_of_surfaces, 1, 1])
    vecs_2 = np.tile(np.expand_dims(texture_vecs, axis=1), [1, number_of_surfaces, 1])
    diff = vecs_1 - vecs_2
    return np.linalg.norm(diff, axis=-1)/256

def assemble_surfaces(surfaces, normal_angles, rgb_image, lab_image, depth_image, number_of_surfaces_in, patches, models, vectors, points_in_space, depth_edges):
    t = time.time()
    global number_of_surfaces
    number_of_surfaces = number_of_surfaces_in
    color_similarity_model, angle_similarity_model, texture_similarity_model, texture_model, nearest_points_func = models

    average_positions, histogram_color, histogram_angles, histogram_texture, centroids,\
    average_normals, centroid_indices, surfaces, planes, surface_patch_points, neighbors, border_centers, norm_image, depth_extend \
        = extract_information(rgb_image, texture_model, surfaces, patches, normal_angles, lab_image, depth_image, points_in_space, depth_edges)
    similarities_color = color_similarity_model(histogram_color)
    similarities_angle = angle_similarity_model(histogram_angles)
    similarities_texture = texture_similarity_calc(histogram_texture)
    sim_neighbors, sim_neighbors_concave, sim_occlusion, sim_occlusion_coplanar, sim_curved\
        = determine_similar_patches(similarities_texture, similarities_color, similarities_angle, [(0.6, 0.6), (0, 0), (0.5, 0.5, 5), (0.8, 1.0, 5.5), (0.8, 0.7, 1.05)], neighbors, planes)
    coplanarity = determine_coplanarity(sim_occlusion_coplanar, centroids, angles_to_normals(average_normals).astype("float32"), planes, number_of_surfaces)

    join_matrix_convexity, concave, new_surfaces = determine_convexly_connected_surfaces(nearest_points_func, surface_patch_points, neighbors,
                                                                 border_centers, normal_angles, surfaces, points_in_space, coplanarity)
    #plot_surfaces(new_surfaces, False)
    _, relabeling = join_surfaces_according_to_join_matrix(join_matrix_convexity, surfaces.copy(), number_of_surfaces)
    join_matrix_occlusion, surfaces = join_disconnected_patches(nearest_points_func, sim_occlusion, sim_curved, coplanarity, average_positions,
                                                                surface_patch_points, surfaces, points_in_space, relabeling, norm_image, depth_extend)
    join_matrix_final = remove_convex_connections_after_occlusion_reasoning(join_matrix_occlusion, join_matrix_convexity,
                                                                      similarities_color + similarities_texture, coplanarity, concave,
                                                                      centroids, depth_image)

    surfaces, _ = join_surfaces_according_to_join_matrix(join_matrix_final, surfaces, number_of_surfaces)
    print(time.time() - t)
    plot_surfaces(surfaces, False)

def main():
    models = get_GPU_models()
    for index in list(range(108, 111)):
        print(index)
        depth, rgb, annotation = load_image(index)
        lab = rgb_to_Lab(rgb)
        Q = np.load(f"out/{index}/Q.npy")
        depth_image = np.load(f"out/{index}/depth.npy")
        angles = np.load(f"out/{index}/angles.npy")
        patches = np.load(f"out/{index}/patches.npy")
        vectors = np.load(f"out/{index}/vectors.npy")
        points_in_space = np.load(f"out/{index}/points.npy")
        depth_edges = np.load(f"out/{index}/edges.npy")
        Q = np.argmax(Q, axis=-1)
        number_of_surfaces = int(np.max(Q) + 1)
        assemble_surfaces(Q, angles, rgb, lab, depth_image, number_of_surfaces, patches, models, vectors, points_in_space, depth_edges)
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