import numpy as np
import tensorflow as tf
import calculate_normals
import find_edges
import find_planes
import plot_image
import time
from standard_values import *
from image_operations import rgb_to_Lab
from tensorflow.keras import layers, Model
from scipy.spatial.transform import Rotation
from numba import njit
from load_images import *
from find_planes import plot_surfaces
from image_processing_models_GPU import calculate_nearest_point_function

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

def prepare_occlusion_closest_point_inputs(occlusion_candidates, centroids, number_of_surfaces, points):
    list = []
    for i in range(number_of_surfaces):
        list.append([])
    for i_1, i_2 in occlusion_candidates:
        list[i_1].append(centroids[i_2])
        list[i_2].append(centroids[i_1])
    i = 0
    index = -1
    missing = []
    while i < len(list):
        index += 1
        if len(list[i]) == 0:
            del list[i]
            missing.append(index)
            continue
        i += 1
    return list, missing

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
def calc_average_normal_and_position_and_counts(angle_image, index_image, number_of_surfaces):
    height, width = np.shape(index_image)
    counter = np.zeros((number_of_surfaces, 1))
    counter_with_edges = np.zeros((number_of_surfaces, 1))
    angles = np.zeros((number_of_surfaces, 2))
    positions = []
    for _ in range(number_of_surfaces):
        positions.append([[0, 0]])
    for y in range(1, height-1):
        for x in range(1, width-1):
            surface = index_image[y][x]
            counter_with_edges[surface][0] += 1
            for y2 in range(y-1, y+2):
                for x2 in range(x-1, x+2):
                    if index_image[y2][x2] != surface:
                        continue
            counter[surface][0] += 1
            angles[surface] += angle_image[y][x]
            positions[surface].append([x, y])
    div_counter = counter.copy()
    for i in range(number_of_surfaces):
        if div_counter[i] == 0:
            div_counter[i] = 1
    angles = angles/div_counter
    return angles, positions, counter, counter_with_edges

def determine_centroid(depth_image, positions):
    for i in range(len(positions)):
        del positions[i][0]
    positions = [np.asarray(p) for p in positions]
    sums = [np.sum(p, axis=0) for p in positions]
    sums = sums / np.asarray([[len(p), len(p)] for p in positions])
    centroids = []
    for i in range(len(positions)):
        if np.any(np.isnan(sums[i])):
            centroids.append([-1, -1])
            continue
        point = positions[i][np.argmin(np.sum(np.square(positions[i] - sums[i]), axis=-1))]
        centroids.append([point[0], point[1], depth_image[point[1]][point[0]]])
    return np.asarray(centroids)

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

def swap_values(vec):
    return np.asarray([vec[1], vec[0], vec[2]])

def determine_convexity_with_closest_points(angles, closest_points, neighbors, number_of_surfaces, surfaces, depth_image, join_matrix):
    convexity = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(number_of_surfaces):
        for j in range(len(neighbors[i])):
            surface_2 = neighbors[i][j]
            index_1 = neighbors[surface_2].index(i)
            if surface_2 < i:
                continue
            c_1 = closest_points[i][j]
            c_2 = closest_points[surface_2][index_1]
            depth_1 = depth_image[c_1[1]][c_1[0]]
            depth_2 = depth_image[c_2[1]][c_2[0]]
            middle_depth = (depth_1 + depth_2) / 2
            diff = c_1 - c_2
            diff = [diff[0]*factor_x*middle_depth, -diff[1]*factor_y*middle_depth, depth_1-depth_2]
            diff = diff/np.linalg.norm(diff)
            normal_1, normal_2 = angles_to_normals(np.asarray([angles[c_1[1]][c_1[0]], angles[c_2[1]][c_2[0]]]))
            v1 = np.dot(diff, normal_1)
            v2 = np.dot(diff, normal_2)
            convex = False
            if v1 - v2 > 0:
                convex = True
                direction = c_2 - c_1
                length = np.max(np.abs(direction))
                direction = direction / max(length, 1)
                height = depth_image[c_1[1]][c_1[0]]
                factor = np.dot(np.abs(direction), [factor_x, factor_y]) / np.sum(direction) * 6
                for m in range(length+1):
                    point = [int(p) for p in np.round(c_1 + m*direction)]
                    new_height = depth_image[point[1]][point[0]]
                    if np.abs(new_height - height) > factor*height:
                        convex = False
                        break
                    height = new_height

            if convex:
                convexity[i][surface_2] = 1
                convexity[surface_2][i] = 1
                direction = c_2 - c_1
                length = np.max(np.abs(direction))
                direction = direction / max(length, 1)
                for m in range(length + 1):
                    point = c_1 + m * direction
                    surfaces[int(point[1])][int(point[0])] = 70

    join_matrix = join_matrix + convexity
    join_matrix[join_matrix > 1] = 1
    return join_matrix

def determine_color_similar_neighbors(color_similarities, neighbors, join_matrix, number_of_surfaces):
    similar_neighbors = np.zeros((number_of_surfaces, number_of_surfaces))
    threshold = 0.6
    for s_1 in range(number_of_surfaces):
        for s_2 in range(s_1 + 1, number_of_surfaces):
            if s_2 in neighbors[s_1]:
                if color_similarities[s_1][s_2] < threshold:
                    similar_neighbors[s_1][s_2] = 1
                    similar_neighbors[s_2][s_1] = 1

    join_matrix = join_matrix + similar_neighbors
    join_matrix[join_matrix > 1] = 1

def determine_occlusion_candidates(surface_similarities, number_of_surfaces, neighbors):
    pairs = []
    threshold = 0.5
    for i_1 in range(number_of_surfaces):
        for i_2 in range(i_1 + 1, number_of_surfaces):
            if i_2 not in neighbors[i_1]:
                if surface_similarities[i_1][i_2] < threshold:
                    pairs.append((i_1, i_2))
    return pairs

def join_surfaces(join_matrix, surfaces, number_of_surfaces):
    labels = np.arange(number_of_surfaces)
    print(labels)
    for i in range(number_of_surfaces):
        for j in range(i + 1, number_of_surfaces):
            if join_matrix[i][j] == 1:
                surfaces[surfaces == j] = labels[i]
                labels[j] = labels[i]
    return surfaces

def color_and_angle_histograms(lab_image, angle_image, surfaces, number_of_surfaces, pixels_per_surface):
    histograms_color = np.zeros((256, 256, number_of_surfaces))
    histograms_angles = np.zeros((40, 40, number_of_surfaces))
    normalized_angles = np.asarray((angle_image / (2*np.pi) + 0.5) * 39, dtype="uint8")
    inverse_pixels = 1 / pixels_per_surface * 10
    height, width = np.shape(surfaces)
    for y in range(height):
        for x in range(width):

            s = surfaces[y][x]
            if s < 0.0001:
                continue

            a_1, a_2 = normalized_angles[y][x]
            l, a, b = np.round(lab_image[y][x] + 127)

            histograms_angles[a_1][a_2][s] += inverse_pixels[s]
            histograms_color[int(a)][int(b)][s] += inverse_pixels[s]

    return histograms_color, histograms_angles

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
    model = Model(inputs=[input], outputs=distance)
    return lambda histogram: model.predict(np.asarray([histogram]))

def determine_occlusion(number_of_surfaces, centroids, depth_image, surfaces):
    occlusions = np.zeros((number_of_surfaces, number_of_surfaces))
    for p_1 in range(number_of_surfaces):
        p_1 = 33
        for p_2 in range(1, number_of_surfaces):
            if p_1 == p_2:
                continue
            c_1 = centroids[p_1]
            c_2 = centroids[p_2]

            p_1_actual = surfaces[c_1[1]][c_1[0]]
            p_2_actual = surfaces[c_1[1]][c_1[0]]

            dif = c_2[:2] - c_1[:2]

            length = np.max(np.abs(dif))

            direction = dif/length

            positions = np.swapaxes(np.expand_dims(c_1[:2], axis=-1) + np.ndarray.astype(np.round(np.expand_dims(direction, axis=-1) *
                                                                    (lambda x: np.stack([x, x]))(np.arange(1, int(np.floor(length))))), dtype="int32"), axis1=0, axis2=1)
            pos_1 = c_1.copy()
            pos_2 = c_2.copy()
            index_1 = 0
            index_2 = len(positions)
            index = -1
            p_2_actual_already_used = False
            for p in positions:
                index += 1
                s = surfaces[p[1]][p[0]]
                if s == p_1_actual:
                    index_1 = index
                    pos_1 = p.copy()
                    continue
                elif s == p_1:
                    p_1_actual = p_1
                    index_1 = index
                    pos_1 = p.copy()
                    continue
                elif s == p_2:
                    index_2 = index
                    pos_2 = p.copy()
                    break
                elif s == p_2_actual and not p_2_actual_already_used:
                    index_2 = index
                    pos_2 = p.copy()
                    p_2_actual_already_used = True
                    continue

            d_1 = depth_image[pos_1[1]][pos_1[0]]
            d_2 = depth_image[pos_2[1]][pos_2[0]]
            d_dif = d_2 - d_1

            occluded = True
            index_dif = 1/(index_2-index_1) * d_dif

            for i in range(index_2-index_1-1):
                p = positions[index_1+i+1]
                if depth_image[p[1]][p[0]] > i*index_dif + d_1:
                    occluded = False
                    break
            if occluded:
                occlusions[p_1][p_2] = 1
                occlusions[p_2][p_1] = 1

            for pos1, pos2 in positions:
                if occluded:
                    surfaces[pos2][pos1] = 0
                else:
                    surfaces[pos2][pos1] = 50

        break
    return surfaces

def assemble_surfaces(surfaces, normal_angles, normal_vectors, rgb_image, lab_image, depth_image, number_of_surfaces, patches):
    color_similarity_model = chi_squared_distances_model((8, 8), (4, 4))
    angle_similarity_model = chi_squared_distances_model((4, 4))

    #average_normals, positions, counts_without_edges, counts = calc_average_normal_and_position_and_counts(angles, surfaces, number_of_surfaces)

    #histogram_color, histogram_angles = color_and_angle_histograms(lab_image, angles, surfaces, number_of_surfaces, counts)
    #color_similarities = color_similarity_model(histogram_color)
    #angle_similarities = angle_similarity_model(histogram_angles)

    depth_edges = find_edges.find_edges_from_depth_image(depth_image)
    new_surfaces = surfaces.copy() * (np.ones_like(depth_edges) - depth_edges)
    #plot_surfaces(new_surfaces, False)
    neighbors, border_centers = determine_neighbors_with_border_centers(surfaces, number_of_surfaces, depth_edges)
    surface_patch_points = extract_points(patches, number_of_surfaces)

    nearest_points_func = calculate_nearest_point_function()
    nearest_points = nearest_points_func(surface_patch_points, prepare_border_centers(neighbors, border_centers))
    #nearest_points = [print(p) for p in nearest_points]
    #quit()
    nearest_points = [np.asarray(p.numpy(), dtype="int32") for p in nearest_points]
    convexity, surfaces = determine_convexity_with_closest_points(normal_angles, nearest_points, neighbors, number_of_surfaces, surfaces, depth_image)
    print(nearest_points)
    plot_surfaces(surfaces, False)
    quit()
    #centroids = determine_centroid(depth_image, positions)

    #show_centroids(surfaces, centroids)

def show_centroids(surfaces, centroids):
    for c in centroids:
        #for i in range(-1, 2):
        #    for j in range()
        surfaces[c[1]][c[0]] = 99
    plot_surfaces(surfaces, False)

def main():
    #Q, depth_image, angles = find_planes.test_model_on_image([110])
    depth, rgb, annotation = load_image(110)
    lab = rgb_to_Lab(rgb)
    Q = np.load("Q.npy")
    depth_image = np.load("depth.npy")
    angles = np.load("angles.npy")
    patches = np.load("patches.npy")
    vectors = np.load("vectors.npy")
    Q = np.argmax(Q, axis=-1)
    number_of_surfaces = int(np.max(Q) + 1)
    assemble_surfaces(Q, angles, vectors, rgb, lab, depth_image, number_of_surfaces, patches)
    quit()
    avg_angles, positions, counts = calc_average_normal_and_position_and_counts(angles, Q, number_of_surfaces)
    centroids = determine_centroid(depth_image, positions)
    occlusions = determine_occlusion(number_of_surfaces, centroids, depth_image, Q)
    plot_surfaces(occlusions, False)

if __name__ == '__main__':
    main()