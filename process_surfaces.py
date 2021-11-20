import numba
import warnings

import numpy as np

import plot_image
from standard_values import *
from numba import njit
from load_images import *

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
def determine_neighboring_surfaces_and_border_centers(surfaces, depth_edges, num_surfaces):
    height, width = np.shape(surfaces)
    neighbors = np.zeros((num_surfaces, num_surfaces))
    border_centers = np.zeros((num_surfaces, num_surfaces, 2))
    new_surfaces = surfaces.copy() * depth_edges
    #plot_image.plot_surfaces(new_surfaces)
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
    for i in range(num_surfaces):
        neighbors[i][i] = 0
        border_centers[i][i] = 0
        for j in range(num_surfaces):
            if neighbors[i][j] < 4:
                neighbors[i][j] = 1
                neighbors[j][i] = 1

    border_centers = np.divide(border_centers, 2*np.expand_dims(neighbors, axis=-1))

    neighbors_list = []
    for i in range(num_surfaces):
        current_list = []
        for j in range(1, num_surfaces):
            if neighbors[i][j] > 3:
                current_list.append(j)

        neighbors_list.append(current_list)
    return neighbors_list, border_centers

def determine_surface_patch_centroids(data):
    surface_patch_points, average_positions, points_3d = data["patch_points"], data["avg_pos"], data["points_3d"]
    centroids = []
    centroid_indices = []
    to_int = lambda x: (int(x[0]), int(x[1]))
    for i in range(len(surface_patch_points)):
        if len(surface_patch_points[i]) == 0:
            centroids.append([-1, -1])
            continue
        point = to_int(surface_patch_points[i][np.argmin(np.sum(np.square(surface_patch_points[i] - average_positions[i]), axis=-1))])
        centroids.append(points_3d[point[1]][point[0]])
        centroid_indices.append([point[1], point[0]])
    return np.asarray(centroids), centroid_indices

@njit()
def extract_individual_points_from_surfaces(surface_patches, num_surfaces):
    points = []
    for i in range(num_surfaces):
        points.append([[0, 0]])

    for y in range(height):
        for x in range(width):
            points[surface_patches[y][x]].append([x, y])

    for i in range(num_surfaces):
        del points[i][0]
    return [np.asarray(p, dtype="float32") for p in points]


@njit()
def determine_pixel_counts_and_average_position_for_patches(patches, num_surfaces):
    height, width = np.shape(patches)
    counter = np.zeros((num_surfaces, 1))
    positions = []
    for _ in range(num_surfaces):
        positions.append([[0, 0]])
    for y in range(1, height-1):
        for x in range(1, width-1):
            surface = patches[y][x]
            counter[surface] += 1
            positions[surface].append([x, y])
    final_positions = np.zeros((num_surfaces, 2))
    for i in range(num_surfaces):
        if counter[i] > 0:
            final_positions[i] = np.sum(np.asarray(positions[i]), axis=0) / counter[i]
    for i in range(num_surfaces):
        del positions[i][0]
    return final_positions, counter

def swap_values(vec):
    return np.asarray([vec[1], vec[0], vec[2]])

@njit()
def determine_convexity_with_above_and_below_line(normal_1, normal_2, space_point_1, space_point_2, c_1, c_2, points_3d):
    normal_direction = (normal_1 + normal_2) / 2
    normal_direction = normal_direction / np.linalg.norm(normal_direction)

    vec_1 = space_point_2 - space_point_1
    vec_2 = np.cross(normal_direction, vec_1)

    plane_normal = np.cross(vec_1, vec_2)
    plane_val = np.dot(space_point_1, plane_normal)

    direction = c_2 - c_1
    length = np.max(np.abs(direction))
    direction = direction / max(length, 1)
    above = 0
    below = 0

    for m in range(0, length):
        point = c_1 + m * direction
        if np.dot(points_3d[int(point[1])][int(point[0])], plane_normal) > plane_val:
            above += 1
        else:
            below += 1
    return above, below


def determine_convexity_for_candidates(data, candidates, closest_points):
    angles, neighbors, surfaces, points_3d, coplanarity, norm_image, num_surfaces = data["angles"], \
                     data["neighbors"], data["surfaces"],  data["points_3d"], data["coplanarity"], data["norm"], data["num_surfaces"]

    convex = np.zeros((num_surfaces, num_surfaces))
    concave = np.zeros((num_surfaces, num_surfaces))
    new_surfaces = surfaces.copy()
    for i in range(num_surfaces):
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
            space_point = np.asarray([norm_image[c_1[1]][c_1[0]], *list(points_3d[c_1[1]][c_1[0]])])
            new_space_point = np.zeros(4)
            s = surfaces[c_1[1]][c_1[0]]
            threshold = 0.005
            diff_counter = 0
            for m in range(length + 1):
                point = [int(p) for p in np.round(c_1 + m * direction)]
                new_s = surfaces[point[1]][point[0]]
                new_space_point[0] = norm_image[point[1]][point[0]]
                new_space_point[1:] = points_3d[point[1]][point[0]]
                if np.max(np.abs(new_space_point - space_point)) / (new_space_point[0] + space_point[0]) > threshold and s != new_s and new_s != 0:
                    connected = False
                    break
                if new_s not in [i, surface_2]:
                    if diff_counter >= 1:
                        connected = False
                        break
                    diff_counter += 1
                    continue
                space_point = new_space_point.copy()
                s = new_s
            if not connected:
                continue
            space_point_1 = points_3d[c_1[1]][c_1[0]]
            space_point_2 = points_3d[c_2[1]][c_2[0]]
            diff = utils.normed(space_point_1 - space_point_2)
            normal_1, normal_2 = utils.angles_to_normals(np.asarray([angles[c_1[1]][c_1[0]], angles[c_2[1]][c_2[0]]]))
            v1 = np.dot(diff, normal_1)
            v2 = np.dot(diff, normal_2)

            above, below = determine_convexity_with_above_and_below_line(np.asarray(normal_1, dtype="float32"), np.asarray(normal_2, dtype="float32"),
                                                                         space_point_1, space_point_2, c_1, c_2, points_3d)

            if (candidates[i][surface_2] == 1 and v1 - v2 > 0.05 and above > below*2) or (v1 - v2 > 0 and below <= max(2, (above + below) * 0.2)) or (v1-v2 > -0.04 and above < below * 0.1):
                convex[i][surface_2] = 1
                convex[surface_2][i] = 1
                direction = c_2 - c_1
                length = np.max(np.abs(direction))
                direction = direction / max(length, 1)
                for m in range(length + 1):
                    point = c_1 + m * direction
                    new_surfaces[int(point[1])][int(point[0])] = 70

            elif v1 - v2 < -0.16 or (coplanarity[i][surface_2] == 0 and (above*1.6 < below or (abs(v1-v2) < 0.05 and abs(1-above/below) < 0.2))):
                concave[i][surface_2] = 1
                concave[surface_2][i] = 1
                direction = c_2 - c_1
                length = np.max(np.abs(direction))
                direction = direction / max(length, 1)
                for m in range(length + 1):
                    point = c_1 + m * direction
                    new_surfaces[int(point[1])][int(point[0])] = 0
    return convex, concave, new_surfaces

@njit()
def relabel_surfaces(surfaces, patches):
    for y in range(height):
        for x in range(width):
            if patches[y][x] != 0:
                surfaces[y][x] = patches[y][x]
    return surfaces

@njit()
def determine_histograms_and_average_normals(lab_image, angle_image, texture_image, surfaces, patches, num_surfaces):
    histograms_color = np.zeros((256, 256, num_surfaces))
    histograms_angles = np.zeros((40, 40, num_surfaces))
    histograms_texture = np.zeros((num_surfaces, 256))

    normalized_angles = (angle_image / (2*np.pi) + 0.5) * 39

    counts = np.zeros(num_surfaces)
    all_angles_sum = np.zeros((num_surfaces, 2))
    patch_counter = np.zeros(num_surfaces)
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
    for i in range(num_surfaces):
        if norm[0][0][i] == 0:
            norm[0][0][i] = 1
    histograms_color = histograms_color/norm

    patch_counter[patch_counter == 0] = 1
    all_angles_sum = all_angles_sum / np.expand_dims(patch_counter, axis=-1)
    histograms_angles = histograms_angles / (patch_counter/10)

    counts[counts == 0] = 1
    histograms_texture = histograms_texture / np.expand_dims(counts, axis=-1) * 10

    return histograms_color, histograms_angles, histograms_texture, all_angles_sum

def determine_occlusion_line_points(candidates, point_sets, target_points, num_surfaces):
    centroid_inputs = []
    points_inputs = []

    for i in range(num_surfaces):
        current_list = []
        for j in range(num_surfaces):
            if candidates[i][j] > 0:
                current_list.append(target_points[j])
        if len(current_list) > 0:
            points_inputs.append(point_sets[i])
            centroid_inputs.append(np.asarray(current_list, dtype="float32"))

    return points_inputs, centroid_inputs

def calc_depth_extend_distance(i, j, depth_extends):
    d_1 = depth_extends[i]
    d_2 = depth_extends[j]
    zero_indices = np.zeros(4)
    zero_indices[np.logical_or(np.logical_or(np.logical_and(d_2[:, 0] <= d_1[:, 0], d_1[:, 0] <= d_2[:, 1]),
                 np.logical_and(d_2[:, 0] <= d_1[:, 1], d_1[:, 1] <= d_2[:, 1])),
                 np.logical_and(d_1[:, 0] <= d_2[:, 0], d_2[:, 1] <= d_1[:, 1]))] = 1
    greater_indices = d_1[:, 0] > d_2[:, 1]
    greater_values = d_1[:, 0] - d_2[:, 1]
    smaller_values = d_2[:, 0] - d_1[:, 1]
    smaller_values[greater_indices] = greater_values[greater_indices]
    smaller_values[zero_indices == 1] = 0
    return np.max(smaller_values)

@njit()
def calculate_depth_extend(surfaces, norm_image, points_3d, num_surfaces):
    values = np.zeros((num_surfaces, 4, 2), dtype="float32")
    values[:, :, 0] = np.inf
    values[:, :, 1] = -np.inf
    for y in range(height):
        for x in range(width):
            n = norm_image[y][x]
            if n == 0:
                continue
            new_values = np.asarray([n, points_3d[y][x][0], points_3d[y][x][1], points_3d[y][x][2]], dtype="float32")
            s = surfaces[y][x]
            values[s, :, 0] = np.minimum(values[s, :, 0], new_values)
            values[s, :, 1] = np.maximum(values[s, :, 1], new_values)
    return values

def determine_occlusion(candidates_all, candidates, closest_points, relabeling, data):
    candidates_occlusion, candidates_curved = candidates["occlusion"], candidates["curved"]
    surfaces, coplanarity, norm_image, depth_extend, num_surfaces, points_3d, neighbors = \
        data["surfaces"], data["coplanarity"], data["norm"], data["depth_extend"], data["num_surfaces"], data["points_3d"], data["neighbors"]
    join_matrix = np.zeros((num_surfaces, num_surfaces))
    closest_points_array = np.zeros((num_surfaces, num_surfaces, 2), dtype="int32")
    new_surfaces = surfaces.copy()
    number_of_candidates = np.sum(candidates_all, axis=1)

    index_1 = 0
    for i in range(num_surfaces):
        if number_of_candidates[i] == 0:
            continue
        index_2 = 0
        for j in range(num_surfaces):
            if candidates_all[i][j] == 0:
                continue
            closest_points_array[i][j] = closest_points[index_1][index_2]
            index_2 += 1
        index_1 += 1

    for i in range(num_surfaces):
        for j in range(i+1, num_surfaces):
            if candidates_all[i][j] == 0:
                continue
            l_1 = relabeling[i]
            l_2 = relabeling[j]
            if l_1 == l_2:
                continue
            p_1 = closest_points_array[i][j]
            p_2 = closest_points_array[j][i]

            dif = p_2 - p_1
            length = np.max(np.abs(dif))
            direction = dif / length

            positions = np.swapaxes(np.expand_dims(p_1, axis=-1) + np.ndarray.astype(np.round(np.expand_dims(direction, axis=-1) *
                                    (lambda x: np.stack([x, x]))(np.arange(1, length))), dtype="int32"), axis1=0, axis2=1)
            pos_1 = p_1.copy()
            pos_2 = p_2.copy()
            index_1 = 0
            index_2 = len(positions)
            index = 0
            other_index = False
            other_surface_counter = 0
            zero_counter = 0
            for p in positions:
                s = surfaces[p[1]][p[0]]
                l = relabeling[s]
                if s == i:
                    index_1 = index
                    pos_1 = p.copy()
                elif s == j:
                    index_2 = index
                    pos_2 = p.copy()
                    break
                elif l == l_1 or l == l_2:
                    other_index = False
                    break
                elif s != 0:
                    other_surface_counter += 1
                elif s == 0:
                    zero_counter += 1
                index += 1
            if zero_counter >= 45 or other_surface_counter >= 3:
                other_index = True
            if coplanarity[i][j] == 1 and zero_counter < 2 and other_surface_counter == 0 and j not in neighbors[i]:
                continue
            if (not other_index and (candidates_curved[i][j] == 0) and coplanarity[i][j] == 0) or (not other_index and coplanarity[i][j] == 0 and zero_counter > 8):
                continue

            d_1 = norm_image[pos_1[1]][pos_1[0]]
            d_2 = norm_image[pos_2[1]][pos_2[0]]

            if d_1 > d_2:
                pos_tmp = pos_1.copy()
                pos_1 = pos_2.copy()
                pos_2 = pos_tmp.copy()

            point_1 = points_3d[pos_1[1]][pos_1[0]]
            point_2 = points_3d[pos_2[1]][pos_2[0]]

            line = utils.normed(point_2 - point_1)

            occluded = True
            if index_1 != index_2:
                for k in range(index_2 - index_1 - 1):
                    p = positions[index_1 + k + 1]
                    s = surfaces[p[1]][p[0]]
                    if s in [0, i, j]:
                        continue
                    current_point = points_3d[p[1]][p[0]]
                    proj = point_1 + line * np.dot(current_point - point_1, line)
                    d = current_point - proj
                    len_proj_dist = np.linalg.norm(d)
                    cos_proj = np.dot(utils.normed(current_point), d/len_proj_dist)
                    if cos_proj <= np.pi/2 and cos_proj >= 0:
                        dist = len_proj_dist / cos_proj
                        if dist > 1.3:
                            occluded = False
                            break
            if (coplanarity[i][j] == 0 and candidates_occlusion[i][j] == 0 and candidates_curved[i][j] == 1) and\
                    (abs(index_2 - index_1) > 8 or calc_depth_extend_distance(i, j, depth_extend) > 34):
                occluded = False
            if occluded:
                join_matrix[i][j] = 1
                join_matrix[j][i] = 1

            for pos1, pos2 in positions:
                if occluded:
                    new_surfaces[pos2][pos1] = 100
                else:
                    new_surfaces[pos2][pos1] = 0
    return join_matrix, new_surfaces

def determine_even_planes(data):
    angle_histogram, num_surfaces = np.swapaxes(data["hist_angle"], 2, 0), data["num_surfaces"]
    angle_histogram_sum = np.sum(np.sum(angle_histogram, axis=-1, keepdims=True), axis=-2, keepdims=True)
    angle_histogram_sum[angle_histogram_sum == 0] = 1
    angle_histogram_norm = angle_histogram/angle_histogram_sum
    angle_histogram_norm_without_zero = angle_histogram_norm.copy()
    angle_histogram_norm_without_zero[angle_histogram_norm == 0] = 1
    entropy = -np.sum(np.sum(angle_histogram_norm * np.log(angle_histogram_norm_without_zero), axis=-1), axis=-1)
    max = np.max(np.max(angle_histogram, axis=-1), axis=-1)

    planes = np.zeros(num_surfaces)
    planes[max >= 1.5] = 1.0
    planes[entropy > 2] = 0.0
    planes[max >= 3] = 1.0
    return planes

@njit()
def determine_coplanarity(candidates, centroids, average_normals, planes, num_surfaces):
    coplanarity = np.zeros((num_surfaces, num_surfaces))
    for i in range(num_surfaces):
        if planes[i] > 0:
            for j in range(i+1, num_surfaces):
                if planes[j] > 0 and candidates[i][j] > 0:
                    diff = centroids[i] - centroids[j]
                    dist = float(np.linalg.norm(diff))
                    max_arc = 0.01 + min(0.09, np.arcsin((math.sqrt(dist*1.1))/dist))

                    diff = diff / dist
                    arc_1 = np.abs(np.pi/2 - np.arccos(np.dot(diff, average_normals[i])))
                    arc_2 = np.abs(np.pi/2 - np.arccos(np.dot(diff, average_normals[j])))
                    val = np.abs(1 - np.dot(average_normals[i], average_normals[j]))
                    if val < 0.0095 and arc_1 < max_arc and arc_2 < max_arc:
                        coplanarity[i][j] = 1
                        coplanarity[j][i] = 1
    return coplanarity

def calc_box_and_surface_overlap(data):
    bboxes, surfaces, num_surfaces = data["bboxes"], data["surfaces"], data["num_surfaces"]
    n_bboxes = np.shape(bboxes)[0]
    overlap_counter = np.zeros((n_bboxes, num_surfaces))

    for i in range(n_bboxes):
        box = np.round(bboxes[i]).astype("int32")
        for y in range(box[1], box[3]+1):
            for x in range(box[0], box[2]+1):
                overlap_counter[i][surfaces[y][x]] += 1

    counts = np.zeros(num_surfaces)
    for y in range(height):
        for x in range(width):
            counts[surfaces[y][x]] += 1

    for i in range(num_surfaces):
        if counts[i] == 0: counts[i] = 1

    overlap_counter = overlap_counter / np.expand_dims(counts, axis=0)
    return overlap_counter

def create_bbox_similarity_matrix_from_box_surface_overlap(bbox_overlap_matrix, bbox_data):
    confidence_scores = np.array([x[4] for x in bbox_data])
    num_boxes, num_labels = np.shape(bbox_overlap_matrix)
    t = np.transpose(bbox_overlap_matrix * np.expand_dims(np.sqrt(confidence_scores), axis=-1))
    ext_1 = np.tile(np.expand_dims(t, axis=0), [num_labels, 1, 1])
    ext_2 = np.tile(np.expand_dims(t, axis=1), [1, num_labels, 1])
    similarity = np.sum(ext_1 * ext_2, axis=-1)
    return similarity

def create_bbox_CRF_matrix(bbox_overlap_matrix):
    num_boxes, num_labels = np.shape(bbox_overlap_matrix)
    similarity_matrix = np.zeros((num_boxes + num_labels, num_boxes + num_labels))
    similarity_matrix[num_labels:, :num_labels] = bbox_overlap_matrix
    return similarity_matrix

def neighborhood_list_to_matrix(data):
    neighbors_list, num_surfaces = data["neighbors"], data["num_surfaces"]
    neighbor_matrix = np.zeros((num_surfaces, num_surfaces))
    for i in range(len(neighbors_list)):
        for neighbor in neighbors_list[i]:
            neighbor_matrix[i][neighbor] = 1

    return neighbor_matrix

def create_depth_extend_distance_matrix(data):
    depth_extend, num_surfaces = data["depth_extend"], data["num_surfaces"]
    distances = np.zeros((num_surfaces, num_surfaces))
    for i in range(num_surfaces):
        for j in range(i+1, num_surfaces):
            d = calc_depth_extend_distance(i, j, depth_extend)
            distances[i, j] = d
            distances[j, i] = d
    return distances

def determine_occlusion_candidates_and_connection_info(closest_points, data):
    surfaces, norm_image, num_surfaces = data["surfaces"], data["norm"], data["num_surfaces"]
    occlusion_matrix = np.zeros((num_surfaces, num_surfaces))
    close_matrix = np.zeros((num_surfaces, num_surfaces))
    closest_points_array = np.zeros((num_surfaces, num_surfaces, 2), dtype="int32")

    index_1 = 0
    for i in range(num_surfaces):
        index_2 = 0
        for j in range(num_surfaces):
            closest_points_array[i][j] = closest_points[index_1][index_2]
            index_2 += 1
        index_1 += 1

    for i in range(num_surfaces):
        for j in range(i + 1, num_surfaces):
            p_1 = closest_points_array[i][j]
            p_2 = closest_points_array[j][i]

            dif = p_2 - p_1
            length = np.max(np.abs(dif))
            direction = dif / length

            positions = np.swapaxes(np.expand_dims(p_1, axis=-1) + np.ndarray.astype(np.round(np.expand_dims(direction, axis=-1) *
                                                                          (lambda x: np.stack([x, x]))(np.arange(1, length))), dtype="int32"),axis1=0, axis2=1)
            pos_1 = p_1.copy()
            pos_2 = p_2.copy()
            index_1 = 0
            index_2 = len(positions)
            index = 0
            other_index = False
            other_surface_counter = 0
            zero_counter = 0
            for p in positions:
                s = surfaces[p[1]][p[0]]
                if s == i:
                    index_1 = index
                    pos_1 = p.copy()
                elif s == j:
                    index_2 = index
                    pos_2 = p.copy()
                    break
                elif s != 0:
                    other_surface_counter += 1
                elif s == 0:
                    zero_counter += 1
                index += 1
            if zero_counter >= 45 or other_surface_counter >= 3:
                other_index = True
            if not other_index and abs(index_2 - index_1) < 8:
                close_matrix[i, j] = 1
                close_matrix[j, i] = 1
                continue

            d_1 = norm_image[pos_1[1]][pos_1[0]]
            d_2 = norm_image[pos_2[1]][pos_2[0]]
            d_dif = d_2 - d_1

            occluded = True

            if index_1 != index_2:
                index_dif = 1 / (index_2 - index_1) * d_dif

                for k in range(index_2 - index_1 - 1):
                    p = positions[index_1 + k + 1]
                    s = surfaces[p[1]][p[0]]
                    if norm_image[p[1]][p[0]] > k * index_dif + d_1 + 1 and s != 0:
                        occluded = False
                        break

            if occluded:
                occlusion_matrix[i, j] = 1
                occlusion_matrix[j, i] = 1
    return occlusion_matrix, close_matrix, closest_points_array

@njit()
def calculate_nearest_points_calc(patch_points, border_centers, max_num):
    results = np.zeros((len(border_centers), max_num, 2), dtype="int32")
    for i in range(len(patch_points)):
        diffs = np.subtract(np.expand_dims(patch_points[i], axis=0), np.expand_dims(border_centers[i], axis=1))
        diffs = np.sum(np.square(diffs), axis=-1)
        for j in range(len(diffs)):
            results[i][j] = patch_points[i][np.argmin(diffs[j])]
    return results

def calculate_nearest_points(point_sets, target_points):
    max_num = max([len(a) for a in target_points])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=numba.NumbaPendingDeprecationWarning)
        results = calculate_nearest_points_calc(point_sets, target_points, max_num)
    return [results[i][:len(target_points[i])] for i in range(len(target_points))]

def calculate_texture_similarities(texture_vecs, num_surfaces):
    vecs_1 = np.tile(np.expand_dims(texture_vecs, axis=0), [num_surfaces, 1, 1])
    vecs_2 = np.tile(np.expand_dims(texture_vecs, axis=1), [1, num_surfaces, 1])
    diff = vecs_1 - vecs_2
    return np.linalg.norm(diff, axis=-1)/256

def calc_depth_extend_to_depth_extend_distance_ratio(data):
    average_depth_extends_objects = np.sum(np.abs(np.subtract(data["depth_extend"][:,:,0], data["depth_extend"][:,:,1])), axis=-1, keepdims=True)/4
    average_depth_extends_pairs = (np.tile(average_depth_extends_objects, [1, data["num_surfaces"]]) +
                                  np.tile(np.swapaxes(average_depth_extends_objects, 0, 1), [data["num_surfaces"], 1])) / 2
    ratio = data["depth_extend_distances"] / average_depth_extends_pairs
    ratio[np.isnan(ratio)] = 0
    return ratio


def extract_information_from_surface_data_and_preprocess_surfaces(data, texture_model):
    data["patch_points"] = extract_individual_points_from_surfaces(data["patches"], data["num_surfaces"])
    data["surfaces"] = relabel_surfaces(data["surfaces"], data["patches"])
    data["avg_pos"], data["counts"] = determine_pixel_counts_and_average_position_for_patches(data["surfaces"], data["num_surfaces"])
    data["centroids"], centroid_indices = determine_surface_patch_centroids(data)
    data["surfaces"] = remove_disconnected_components(data["surfaces"], np.asarray(centroid_indices, dtype="int64"))
    data["texture"] = texture_model(data["rgb"])
    if "lab" not in data.keys():
        data["lab"] = utils.rgb_to_Lab(data["rgb"])
    data["hist_color"], data["hist_angle"], data["hist_texture"], data["avg_normals"] = determine_histograms_and_average_normals(data["lab"], data["angles"],
                                                                                      data["texture"], data["surfaces"], data["patches"], data["num_surfaces"])
    data["avg_normals"] = utils.angles_to_normals(data["avg_normals"])
    data["planes"] = determine_even_planes(data)
    data["neighbors"], data["border_centers"] = determine_neighboring_surfaces_and_border_centers(data["surfaces"], data["depth_edges"], data["num_surfaces"])
    data["norm"] = np.linalg.norm(data["points_3d"], axis=-1)
    data["depth_extend"] = calculate_depth_extend(data["surfaces"], data["norm"], data["points_3d"], data["num_surfaces"])
    data["depth_extend_distances"] = create_depth_extend_distance_matrix(data)/1000
    data["depth_extend_distance_ratio"] = calc_depth_extend_to_depth_extend_distance_ratio(data)

def find_optimal_surface_joins_from_annotation(data):
    annotation, surfaces, num_surfaces, num_boxes = data["annotation"], data["surfaces"], data["num_surfaces"], data["num_bboxes"]
    num_annotations = np.max(annotation)
    annotation_counter = np.zeros((num_surfaces, num_annotations+1))
    annotation_counter[:, 0] = np.ones(num_surfaces)
    counts = np.zeros(num_surfaces)
    for y in range(height):
        for x in range(width):
            if (s := surfaces[y][x]) != 0:
                counts[s] += 1
                if (a := annotation[y][x]) != 0:
                    annotation_counter[s][a] += 1

    counts[counts == 0] = 1
    annotation_counter[np.divide(annotation_counter, np.expand_dims(counts, axis=-1)) < 0.6] = 0
    annotation_correspondence = np.argmax(annotation_counter, axis=-1)
    annotation_correspondence[np.max(annotation_counter, axis=-1) == 0] = 0

    Y_1 = np.zeros((num_surfaces, num_surfaces))
    Y_2 = np.zeros((num_surfaces, num_surfaces))
    for s_1 in range(1, num_surfaces):
        for s_2 in range(s_1+1, num_surfaces):
            if (a := annotation_correspondence[s_1]) == (b := annotation_correspondence[s_2]) and a != 0:
                Y_1[s_1][s_2] = 1
            elif a > 0 or b > 0:
                Y_2[s_1][s_2] = 1

    Y = np.pad(np.stack([Y_1, Y_2], axis=0), ((0, 0), (0, num_boxes), (0, num_boxes)))
    return np.asarray([Y[:, 1:, 1:]])

def get_position_and_occlusion_info(data):
    avg_positions, patch_points, surfaces, norm_image, num_surfaces = data["avg_pos"], data["patch_points"], data["surfaces"], data["norm"], data["num_surfaces"]
    input = determine_occlusion_line_points(np.ones((num_surfaces, num_surfaces)), patch_points, avg_positions, num_surfaces)
    nearest_points_for_occlusion = calculate_nearest_points(*input)
    join_matrix, close_matrix, closest_points = determine_occlusion_candidates_and_connection_info(nearest_points_for_occlusion, data)
    return join_matrix, close_matrix, closest_points

def determine_convexly_connected_surfaces(candidates, data):
    patch_points, neighbors, border_centers, normal_angles, surfaces, points_3d, coplanarity, norm_image = data["patch_points"],\
                 data["neighbors"], data["border_centers"], data["angles"], data["surfaces"], data["points_3d"], data["coplanarity"], data["norm"]
    nearest_points = calculate_nearest_points(patch_points, prepare_border_centers(neighbors, border_centers))
    convex, concave, new_surfaces = determine_convexity_for_candidates(data, candidates, nearest_points)
    return convex, concave, new_surfaces

def prepare_border_centers(neighbors, border_centers):
    result = []
    for i in range(len(neighbors)):
        current_list = []
        for neighbor in neighbors[i]:
            current_list.append(border_centers[i][neighbor])
        if len(current_list) == 0:
            current_list.append(np.asarray([0, 0]))
        result.append(np.asarray(current_list, dtype="float32"))
    return result

def calculate_pairwise_similarity_features_for_surfaces(data, models):
    color_similarity_model, angle_similarity_model, texture_model, object_detector = models[:4]

    extract_information_from_surface_data_and_preprocess_surfaces(data, texture_model)

    candidates = {"convexity": np.ones((data["num_surfaces"], data["num_surfaces"]))}
    data["sim_color"] = color_similarity_model(data["hist_color"])
    data["sim_texture"] = calculate_texture_similarities(data["hist_texture"], data["num_surfaces"])
    data["sim_angle"] = angle_similarity_model(data["hist_angle"])/20

    data["planes"] = determine_even_planes(data)
    data["same_plane_type"] = (np.tile(np.expand_dims(data["planes"], axis=0), [data["num_surfaces"], 1]) == np.tile(np.expand_dims(data["planes"], axis=-1), [1, data["num_surfaces"]])).astype("int32")
    data["both_curved"] = (lambda x: np.dot(np.transpose(x), x))(np.expand_dims(1-data["planes"], axis=0))
    data["coplanarity"] = determine_coplanarity(candidates["convexity"], data["centroids"].astype("float64"), data["avg_normals"], data["planes"], data["num_surfaces"])
    data["convex"], data["concave"], _ = determine_convexly_connected_surfaces(candidates["convexity"], data)
    data["neighborhood_mat"] = neighborhood_list_to_matrix(data)
    data["bboxes"] = object_detector(data["rgb"])
    data["bbox_overlap"] = calc_box_and_surface_overlap(data)
    data["bbox_crf_matrix"] = create_bbox_CRF_matrix(data["bbox_overlap"])
    data["bbox_similarity_matrix"] = create_bbox_similarity_matrix_from_box_surface_overlap(data["bbox_overlap"], data["bboxes"])
    data["num_bboxes"] = np.shape(data["bbox_overlap"])[0]
    data["occlusion_mat"], data["close_mat"], data["closest_points"] = get_position_and_occlusion_info(data)
    data["close_mat"][data["neighborhood_mat"] == 1] = 0
    data["distances"] = np.sqrt(np.sum(np.square(data["closest_points"] - np.swapaxes(data["closest_points"], axis1=0, axis2=1)), axis=-1)) / 500

def create_similarity_feature_matrix(data):
    keys = ["bbox_similarity_matrix", "sim_texture", "sim_color", "sim_angle", "convex", "coplanarity", "neighborhood_mat",
            "concave", "distances", "close_mat", "occlusion_mat", "depth_extend_distances", "same_plane_type", "both_curved"]#, "depth_extend_distance_ratio"]
    matrix = np.stack([data[key][1:,1:] for key in keys], axis=-1)
    #matrix[:, :, 11] = matrix[:, :, 11] / 1000
    #matrix[:, : , 8] = matrix[:, : , 8] / 500
    return matrix

@njit()
def join_surfaces_according_to_join_matrix(join_matrix, surfaces, num_surfaces):
    labels = np.arange(num_surfaces)
    for i in range(num_surfaces):
        for j in range(i + 1, num_surfaces):
            if join_matrix[i][j] >= 1:
                l = labels[j]
                new_l = labels[i]
                for k in range(num_surfaces):
                    if labels[k] == l:
                        labels[k] = new_l
    for y in range(height):
        for x in range(width):
            surfaces[y][x] = labels[surfaces[y][x]]
    return surfaces, labels

def determine_disconnected_join_candidates(relabeling, candidates, data):
    candidates_all = candidates["occlusion"] + candidates["occlusion_coplanar"] + candidates["curved"]
    candidates_all[candidates_all > 1] = 1
    input = determine_occlusion_line_points(candidates_all, data["patch_points"], data["avg_pos"], data["num_surfaces"])
    if len(input[0]) == 0: return np.zeros((data["num_surfaces"], data["num_surfaces"])), data["surfaces"]
    closest_points = calculate_nearest_points(*input)
    join_matrix, new_surfaces = determine_occlusion(candidates_all, candidates, closest_points, relabeling, data)
    return join_matrix, new_surfaces