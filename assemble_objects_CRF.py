import numpy as np
import assemble_objects
import detect_objects
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
from standard_values import *

def mean_field_iteration(unary_potentials, pairwise_potentials, Q):
    tf.multiply(pairwise_potentials, Q)



def get_neighborhood_matrix(neighbors_list, number_of_surfaces):
    neighbor_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(len(neighbors_list)):
        for neighbor in neighbors_list:
            neighbor_matrix[i][neighbor] = 1

    return neighbor_matrix, neighbors_list

def calc_box_and_surface_overlap(bboxes, surfaces, number_of_surfaces):
    n_bboxes = np.shape(bboxes)[0]
    overlap_counter = np.zeros((n_bboxes, number_of_surfaces))

    for i in range(n_bboxes):
        box = bboxes[i]
        for y in range(box[1], box[3]+1):
            for x in range(box[0], box[2]+1):
                overlap_counter[i][surfaces[y][x]] += 1

    counts = np.zeros(number_of_surfaces)
    for y in range(height):
        for x in range(width):
            counts[surfaces[y][x]] += 1

    for i in range(number_of_surfaces):
        if counts[i] == 0: counts[i] = 1

    overlap_counter = overlap_counter / np.expand_dims(counts, axis=0)
    return overlap_counter

def get_similarity_data_for_CRF(surfaces, depth_edges, rgb_image, lab_image, patches, models, normal_angles, points_in_space, depth_image):
    number_of_surfaces = int(np.max(surfaces) + 1)
    color_similarity_model, texture_similarity_model, texture_model, nearest_points_func, object_detector = models

    average_positions, histogram_color, histogram_angles, histogram_texture, centroids, \
    average_normals, centroid_indices, surfaces, planes, surface_patch_points, neighbors, border_centers, norm_image, depth_extend \
        = assemble_objects.extract_information(rgb_image, texture_model, surfaces, patches, normal_angles, lab_image, depth_image,
                              points_in_space, depth_edges)

    similarities_color = color_similarity_model(histogram_color)
    similarities_texture = texture_similarity_model(histogram_texture)

    planes = assemble_objects.find_even_planes(np.swapaxes(histogram_angles, 2, 0))
    coplanarity_matrix = assemble_objects.determine_coplanarity(np.ones(number_of_surfaces, number_of_surfaces), centroids,
                                                         assemble_objects.angles_to_normals(average_normals).astype("float32"), planes, number_of_surfaces)

    convexity_matrix, _, _ = assemble_objects.determine_convexly_connected_surfaces(nearest_points_func, surface_patch_points, neighbors, border_centers,
                                                                normal_angles, surfaces, points_in_space, coplanarity_matrix, norm_image, np.ones(number_of_surfaces, number_of_surfaces))

    neighborhood_matrix = get_neighborhood_matrix(neighbors, number_of_surfaces)
    bboxes = object_detector(rgb_image)
    bbox_overlap_matrix = calc_box_and_surface_overlap(bboxes, surfaces, number_of_surfaces)

    return bbox_overlap_matrix, similarities_texture, similarities_color, convexity_matrix, coplanarity_matrix, neighborhood_matrix