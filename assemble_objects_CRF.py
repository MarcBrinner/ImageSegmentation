import numpy as np
import assemble_objects
import detect_objects
import tensorflow as tf
from image_operations import rgb_to_Lab
from load_images import load_image
from image_processing_models_GPU import Variable2, Print_Tensor
from tensorflow.keras import layers, optimizers, Model, losses
from standard_values import *

initial_guess_parameters = [1, 2, 2, 1, 3, 3, 0, -1, -1, 0, 0, 0, -0.5, 1, 0.001]

def print_tensor(input):
    p = Print_Tensor()(input)
    return p

def get_Y_value(annotation, surfaces, number_of_surfaces):
    num_annotations = np.max(annotation)
    annotation_counter = np.zeros((number_of_surfaces, num_annotations+1))
    annotation_counter[:, 0] = np.ones(number_of_surfaces)
    counts = np.zeros(number_of_surfaces)
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

    Y = np.zeros((number_of_surfaces, number_of_surfaces))
    Y_2 = np.zeros((number_of_surfaces, number_of_surfaces))
    for s_1 in range(1, number_of_surfaces):
        for s_2 in range(s_1+1, number_of_surfaces):
            if (a := annotation_correspondence[s_1]) == (b := annotation_correspondence[s_2]) and a != 0:
                Y[s_1][s_2] = 1
            if a > 0 or b > 0:
                Y_2[s_1][s_2] = 1

    return np.asarray([np.stack([Y, Y_2], axis=0)[:, 1:, 1:]])

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(pairwise_potentials, Q_mult), matrix_1), axis=1)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-2)
    new_Q = tf.exp(unary_potentials - tf.multiply(compatibility, weight))
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_CRF(w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, weight, num_iterations=3):
    Q_in = layers.Input(shape=(None, None))
    #unary_in = layers.Input(shape=(None, None))
    num_labels = tf.shape(Q_in)[1]

    convexity_in = layers.Input(shape=(None, None))
    color_in = layers.Input(shape=(None, None))
    texture_in = layers.Input(shape=(None, None))
    coplanarity_in = layers.Input(shape=(None, None))
    boxes_in = layers.Input(shape=(None, None))
    neighborhood_in = layers.Input(shape=(None, None))

    boxes = tf.sigmoid(layers.Add()([Variable2(w_13, name="w_13")(boxes_in), layers.Multiply()([Variable2(w_14, name="w_14")(boxes_in), boxes_in])]))
    boxes_1 = tf.tile(tf.expand_dims(boxes, axis=-2), [1, 1, num_labels, 1])
    boxes_2 = tf.tile(tf.expand_dims(boxes, axis=-3), [1, num_labels, 1, 1])
    box_similarity = tf.reduce_sum(tf.multiply(boxes_1, boxes_2), axis=-1)

    pairwise_potentials = layers.Add()([layers.Add()([Variable2(w_7, name="w_7")(convexity_in), layers.Multiply()([-Variable2(w_1, name="w_1")(convexity_in), convexity_in])]),
                                        layers.Add()([Variable2(w_8, name="w_8")(color_in), layers.Multiply()([Variable2(w_2, name="w_2")(color_in), color_in])]),
                                        layers.Add()([Variable2(w_9, name="w_9")(texture_in), layers.Multiply()([Variable2(w_3, name="w_3")(texture_in), texture_in])]),
                                        layers.Add()([Variable2(w_10, name="w_10")(coplanarity_in), layers.Multiply()([-Variable2(w_4, name="w_4")(coplanarity_in), coplanarity_in])]),
                                        layers.Add()([Variable2(w_11, name="w_11")(box_similarity), layers.Multiply()([-Variable2(w_5, name="w_5")(box_similarity), box_similarity])]),
                                        layers.Add()([Variable2(w_12, name="w_12")(neighborhood_in), layers.Multiply()([-Variable2(w_6, name="w_6")(neighborhood_in), neighborhood_in])])])

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)
    weight = Variable2(weight, name="weight")(Q_in)
    Q = Q_in
    for _ in range(num_iterations):
        Q = mean_field_iteration(Q_in, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight)
    output = Q

    model = Model(inputs=[Q_in, convexity_in, color_in, texture_in, coplanarity_in, boxes_in, neighborhood_in], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[], run_eagerly=True)
    return model

def custom_loss(y_actual, y_predicted):
    join_matrix = tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1)
    mask = tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1)

    num_labels = tf.shape(y_predicted)[2]

    prediction_expanded_1 = tf.tile(tf.expand_dims(y_predicted, axis=1), [1, num_labels, 1, 1])
    prediction_expanded_2 = tf.tile(tf.expand_dims(y_predicted, axis=2), [1, 1, num_labels, 1])
    mult_out = tf.multiply(prediction_expanded_1, prediction_expanded_2)
    sum_out = tf.reduce_sum(mult_out, axis=-1)
    mask_out = tf.multiply(sum_out, mask)

    positive_error = tf.reduce_sum(tf.reduce_sum(tf.multiply(join_matrix, mask_out), axis=-1),axis=-1)
    negative_error = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.ones_like(join_matrix) - join_matrix, mask_out), axis=-1),axis=-1)
    error = -positive_error + 0.1 * negative_error
    print(error)
    return error

def get_initial_probabilities(num_labels):
    potentials = np.eye(num_labels) * 10
    potentials = potentials + np.random.uniform(size=(num_labels, num_labels))
    potentials = potentials / np.sum(potentials, axis=-1, keepdims=True)
    return potentials

def get_neighborhood_matrix(neighbors_list, number_of_surfaces):
    neighbor_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(len(neighbors_list)):
        for neighbor in neighbors_list:
            neighbor_matrix[i][neighbor] = 1

    return neighbor_matrix

def calc_box_and_surface_overlap(bboxes, surfaces, number_of_surfaces):
    n_bboxes = np.shape(bboxes)[0]
    overlap_counter = np.zeros((n_bboxes, number_of_surfaces))

    for i in range(n_bboxes):
        box = np.round(bboxes[i]).astype("int32")
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
    assemble_objects.set_number_of_surfaces(number_of_surfaces)
    color_similarity_model, texture_similarity_model, texture_model, nearest_points_func, object_detector, _ = models

    average_positions, histogram_color, histogram_angles, histogram_texture, centroids, \
    average_normals, centroid_indices, surfaces, planes, surface_patch_points, neighbors, border_centers, norm_image, depth_extend \
        = assemble_objects.extract_information(rgb_image, texture_model, surfaces, patches, normal_angles, lab_image, depth_image,
                              points_in_space, depth_edges)

    similarities_color = color_similarity_model(histogram_color)
    similarities_texture = assemble_objects.texture_similarity_calc(histogram_texture)

    planes = assemble_objects.find_even_planes(np.swapaxes(histogram_angles, 2, 0))
    coplanarity_matrix = assemble_objects.determine_coplanarity(np.ones((number_of_surfaces, number_of_surfaces)), centroids,
                                                         assemble_objects.angles_to_normals(average_normals).astype("float32"), planes, number_of_surfaces)

    convexity_matrix, _, _ = assemble_objects.determine_convexly_connected_surfaces(nearest_points_func, surface_patch_points, neighbors, border_centers,
                                                                normal_angles, surfaces, points_in_space, coplanarity_matrix, norm_image, np.ones((number_of_surfaces, number_of_surfaces)))

    neighborhood_matrix = get_neighborhood_matrix(neighbors, number_of_surfaces)
    bboxes = object_detector(rgb_image)
    bbox_overlap_matrix = calc_box_and_surface_overlap(bboxes, surfaces, number_of_surfaces)

    return np.transpose(bbox_overlap_matrix)[1:], similarities_texture[1:, 1:], similarities_color[1:, 1:], convexity_matrix[1:, 1:], coplanarity_matrix[1:, 1:], neighborhood_matrix[1:, 1:]

def plot_prediction(prediction, surfaces):
    max_vals = np.argmax(prediction[0], axis=-1)
    for i in range(len(max_vals)):
        if max_vals[i] != i:
            surfaces[surfaces == i] = max_vals[i]+1
    assemble_objects.plot_surfaces(surfaces)

def assemble_objects_CRF(surfaces, depth_edges, rgb_image, lab_image, patches, models, normal_angles, points_in_space, depth_image, annotation, train=False):
    bbox_overlap_matrix, similarities_texture, similarities_color, convexity_matrix, coplanarity_matrix, neighborhood_matrix =\
        get_similarity_data_for_CRF(surfaces, depth_edges, rgb_image, lab_image, patches, models, normal_angles, points_in_space, depth_image)

    CRF_model = models[-1]

    num_labels = int(np.max(surfaces))
    Q_in = get_initial_probabilities(num_labels)

    input = [np.asarray([e]) for e in [Q_in, convexity_matrix, similarities_color, similarities_texture,
                                                              coplanarity_matrix, bbox_overlap_matrix, neighborhood_matrix]]

    if not train:
        prediction = CRF_model.predict(input)
        plot_prediction(prediction, surfaces)
    else:
        Y = get_Y_value(annotation, surfaces, num_labels+1)
        #s, l = assemble_objects.join_surfaces_according_to_join_matrix(join_matrix, surfaces, num_labels+1)
        #assemble_objects.plot_surfaces(s)
        CRF_model.fit(input, Y)
        prediction = CRF_model.predict(input)
        plot_prediction(prediction, surfaces)

def get_GPU_models():
    return assemble_objects.chi_squared_distances_model((10, 10), (4, 4)), \
           assemble_objects.chi_squared_distances_model_1D(), \
           assemble_objects.extract_texture_function(), \
           assemble_objects.calculate_nearest_points,\
           detect_objects.get_object_detector(weights="parameters/yolov3.weights"),\
           mean_field_CRF(*initial_guess_parameters)

def main():
    models = get_GPU_models()
    for index in list(range(0, 111)):
        print(index)
        depth, rgb, annotation = load_image(index)
        lab = rgb_to_Lab(rgb)
        Q = np.load(f"out/{index}/Q.npy")
        depth_image = np.load(f"out/{index}/depth.npy")
        angles = np.load(f"out/{index}/angles.npy")
        patches = np.load(f"out/{index}/patches.npy")
        points_in_space = np.load(f"out/{index}/points.npy")
        depth_edges = np.load(f"out/{index}/edges.npy")
        Q = np.argmax(Q, axis=-1)
        assemble_objects_CRF(Q, depth_edges, rgb, lab, patches, models, angles, points_in_space, depth_image, annotation, True)
    quit()

if __name__ == '__main__':
    main()