import time
import process_surfaces as ps
from standard_values import *
from numba import njit
from load_images import *
from find_planes import plot_surfaces
from image_processing_models_GPU import extract_texture_function, chi_squared_distances_model_1D, chi_squared_distances_model_2D

parameters_candidates = [(0.6, 0.5, 7, 2), (0.8, 1.0, 7), (0.8, 0.7, 1.15)]

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

def determine_join_candidates(data, thresholds):
    texture_similarities, color_similarities, normal_similarities, planes, num_surfaces = data["sim_texture"], data["sim_color"],\
                      data["sim_angle"], data["planes"], data["num_surfaces"]
    similar_patches_2 = np.zeros_like(texture_similarities)
    similar_patches_2[color_similarities < thresholds[1][1]] = 1
    similar_patches_2[normal_similarities > thresholds[1][2]] = 0
    similar_patches_2[texture_similarities > thresholds[1][0]] = 0
    
    similar_patches_1 = np.zeros_like(texture_similarities)
    similar_patches_1[color_similarities < thresholds[0][1]] = 1
    similar_patches_1[texture_similarities > thresholds[0][0]] = 0
    similar_patches_1[normal_similarities > thresholds[0][2]] = 0
    for i in range(num_surfaces):
        if planes[i] == 1:
            continue
        for j in range(i + 1, num_surfaces):
            if planes[j] == 1 or similar_patches_2[i][j] == 0 or thresholds[0][3] < normal_similarities[i][j]:
                continue
            else:
                similar_patches_1[i][j] = 1
                similar_patches_1[j][i] = 1

    similar_patches_3 = np.zeros_like(texture_similarities)
    similar_patches_3[color_similarities < thresholds[2][1]] = 1
    similar_patches_3[texture_similarities > thresholds[2][0]] = 0
    similar_patches_3[texture_similarities + color_similarities > thresholds[2][2]] = 0
    similar_patches_3[planes == 1, :] = 0
    similar_patches_3[:, planes == 1] = 0

    for i in range(np.shape(texture_similarities)[0]):
        similar_patches_1[i][i] = 0
        similar_patches_2[i][i] = 0
        similar_patches_3[i][i] = 0

    values = texture_similarities + color_similarities
    similar_patches_4 = np.zeros_like(texture_similarities)
    similar_patches_4[values < 1.2] = 1
    
    candidates = {}
    candidates["occlusion"] = similar_patches_1
    candidates["occlusion_coplanar"] = similar_patches_2
    candidates["curved"] = similar_patches_3
    candidates["convexity"] = similar_patches_4
    return candidates

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

def remove_convex_connections_after_occlusion_reasoning(join_matrices, data):
    similarity = data["sim_color"] + data["sim_texture"]
    join_matrix_occlusion, join_matrix_before_occlusion = join_matrices["convex"], join_matrices["occlusion"]
    centroids, depth_image, coplanarity, concave, num_surfaces = data["centroids"], data["depth"], data["coplanarity"], data["concave"], data["num_surfaces"]

    groups = []
    for i in range(num_surfaces):
        groups.append(set())
        groups[-1].add(i)

    for i in range(num_surfaces):
        for j in range(i+1, num_surfaces):
            if join_matrix_before_occlusion[i][j] > 0:
                groups[i].update(groups[j])
                groups[j].update(groups[i])

    pairs = []
    distances = np.zeros((num_surfaces, num_surfaces))
    for i in range(num_surfaces):
        for j in range(i+1, num_surfaces):
            if join_matrix_occlusion[i][j] > 0 and join_matrix_before_occlusion[i][j] == 0:
                pairs.append((i, j))
                d = np.linalg.norm(centroids[i] - centroids[j])
                distances[i][j] = d
                distances[j][i] = d

    indices = depth_image > 0
    average_depth = np.sum(depth_image[indices]) / np.sum(np.asarray(indices, dtype="int32"))
    n_1 = average_depth * math.tan(viewing_angle_x) * 1.5
    n_2 = np.max(distances)*4

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

def determine_convexly_connected_surfaces(candidates, data):
    patch_points, neighbors, border_centers, normal_angles, surfaces, points_3d, coplanarity, norm_image = data["patch_points"],\
                 data["neighbors"], data["border_centers"], data["angles"], data["surfaces"], data["points_3d"], data["coplanarity"], data["norm"]
    nearest_points = ps.calculate_nearest_points(patch_points, prepare_border_centers(neighbors, border_centers))
    convex, concave, new_surfaces = ps.determine_convexity_for_candidates(data, candidates, nearest_points)
    return convex, concave, new_surfaces

def determine_disconnected_join_candidates(relabeling, candidates, data):
    candidates_all = candidates["occlusion"] + candidates["coplanar"] + candidates["curved"]
    candidates_all[candidates_all > 1] = 1
    input = ps.determine_occlusion_line_points(candidates_all, data["patch_points"], data["centroids"], data["num_surfaces"])
    if len(input[0]) == 0: return np.zeros((data["num_surfaces"], data["num_surfaces"])), data["surfaces"]
    closest_points = ps.calculate_nearest_points(*input)
    join_matrix, new_surfaces = ps.determine_occlusion(candidates_all, candidates, closest_points, relabeling, data)
    return join_matrix, new_surfaces

def get_GPU_models():
    return chi_squared_distances_model_2D((10, 10), (4, 4)), \
           chi_squared_distances_model_2D((4, 4), (1, 1)), \
           extract_texture_function(), \


def assemble_surfaces(data, models):
    color_similarity_model, angle_similarity_model, texture_model = models

    ps.extract_information_from_surface_data_and_preprocess_surfaces(data, texture_model)

    data["sim_color"] = color_similarity_model(data["hist_color"])
    data["sim_angle"] = angle_similarity_model(data["hist_angles"])
    data["sim_texture"] = ps.calculate_texture_similarities(data["hist_texture"], data["num_surfaces"])

    join_candidates = determine_join_candidates(data, parameters_candidates)

    data["coplanarity"] = ps.determine_coplanarity(join_candidates["occlusion_coplanar"], data)

    join_surfaces = {}
    join_surfaces["convex"], data["concave"], new_surfaces = determine_convexly_connected_surfaces(join_candidates["convexity"], data)
    _, relabeling = join_surfaces_according_to_join_matrix(join_surfaces["convex"], data["surfaces"].copy(), data["num_surfaces"])
    join_surfaces["disconnected"], new_surfaces = determine_disconnected_join_candidates(relabeling, join_candidates, data)
    join_surfaces["final"] = remove_convex_connections_after_occlusion_reasoning(join_surfaces, data)

    surfaces, _ = join_surfaces_according_to_join_matrix(join_surfaces["final"], data["surfaces"], data["num_surfaces"])
    plot_surfaces(surfaces, False)

def main():
    models = get_GPU_models()
    for index in list(range(0, 111)):
        print(index)
        data = load_image_and_surface_information(index)
        assemble_surfaces(data, models)
    quit()

if __name__ == '__main__':
    main()