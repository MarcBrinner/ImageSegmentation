import numpy as np
import assemble_objects_rules
import detect_objects
import tensorflow as tf

import load_images
import process_surfaces as ps
from utils import rgb_to_Lab
from load_images import load_image
from image_processing_models_GPU import Variable2, Print_Tensor, extract_texture_function, chi_squared_distances_model_1D,\
                                        chi_squared_distances_model_2D
from tensorflow.keras import layers, optimizers, Model, losses
from standard_values import *
from find_planes import plot_surfaces

#initial_guess_parameters = [1.1154784, 2, 2, 1, 1.1, 3, -1, 2, 1.0323303, 0.24122038]
initial_guess_parameters = [1, 2, 2, 1, 1.1, 3, -1, 2, 1, 0.02]

def print_tensor(input):
    p = Print_Tensor()(input)
    return p

variable_names = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7", "w_8", "w_9", "weight"]

def print_parameters(model):
    for name in variable_names:
        try:
            print(f"{name}: {model.get_layer(name).weights}")
        except:
            print(f"{name}: Not available.")

def get_Y_value(data):
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

    Y = np.pad(np.stack([Y_1, Y_2, np.zeros_like(Y_1)], axis=0), ((0, 0), (0, num_boxes), (0, num_boxes)))
    for i in range(num_surfaces, num_surfaces + num_boxes):
        Y[2][i][i] = 1.0
    return np.asarray([Y[:, 1:, 1:]])

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(pairwise_potentials, Q_mult), matrix_1), axis=1)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-2)
    new_Q = tf.exp(unary_potentials - tf.multiply(compatibility, weight))
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_CRF(w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, weight, num_iterations=10):
    Q_in = layers.Input(shape=(None, None))
    #unary_in = layers.Input(shape=(None, None))
    num_labels = tf.shape(Q_in)[1]

    convexity_in = layers.Input(shape=(None, None))
    color_in = layers.Input(shape=(None, None))
    texture_in = layers.Input(shape=(None, None))
    coplanarity_in = layers.Input(shape=(None, None))
    boxes_in = layers.Input(shape=(None, None))
    neighborhood_in = layers.Input(shape=(None, None))

    boxes = (tf.sigmoid(layers.Add()([Variable2(w_7, name="w_7")(boxes_in), layers.Multiply()([Variable2(w_8, name="w_8")(boxes_in), boxes_in])])) - 0.5) * 2

    pairwise_potentials = layers.Add()([layers.Multiply()([-Variable2(w_1, name="w_1")(convexity_in), convexity_in]),
                                        #layers.Multiply()([Variable2(w_2, name="w_2")(color_in), color_in]),
                                        #layers.Multiply()([Variable2(w_3, name="w_3")(texture_in), texture_in]),
                                        #layers.Multiply()([-Variable2(w_4, name="w_4")(coplanarity_in), coplanarity_in]),
                                        #layers.Multiply()([-Variable2(w_6, name="w_6")(neighborhood_in), neighborhood_in]),
                                        Variable2(w_9, name="w_9")(neighborhood_in)])
    num_boxes = num_labels - tf.shape(pairwise_potentials)[1]
    pairwise_potentials = tf.pad(pairwise_potentials, [[0, 0], [0, num_boxes], [0, num_boxes]]) #+ layers.Multiply()([-Variable2(w_5, name="w_5")(boxes), boxes])

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)
    weight = Variable2(weight, name="weight")(Q_in)
    Q = Q_in
    for _ in range(num_iterations):
        Q = mean_field_iteration(Q_in, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight)
    output = Q

    model = Model(inputs=[Q_in, convexity_in, color_in, texture_in, coplanarity_in, boxes_in, neighborhood_in], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-3), metrics=[], run_eagerly=True)
    return model

def custom_loss(y_actual, y_predicted):
    join_matrix = tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1)
    not_join_matrix = tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1)
    entropy_mask = tf.reduce_sum(tf.squeeze(tf.gather(y_actual, [2], axis=1), axis=1), axis=-1)

    num_labels = tf.shape(y_predicted)[2]

    prediction_expanded_1 = tf.tile(tf.expand_dims(y_predicted, axis=1), [1, num_labels, 1, 1])
    prediction_expanded_2 = tf.tile(tf.expand_dims(y_predicted, axis=2), [1, 1, num_labels, 1])
    mult_out = tf.multiply(prediction_expanded_1, prediction_expanded_2)
    sum_out = tf.reduce_sum(mult_out, axis=-1)

    positive_error = tf.reduce_sum(tf.reduce_sum(tf.multiply(join_matrix, sum_out), axis=-1),axis=-1)
    negative_error = tf.reduce_sum(tf.reduce_sum(tf.multiply(not_join_matrix, sum_out), axis=-1),axis=-1)
    entropy = -tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(y_predicted, tf.math.log(y_predicted)), axis=-1), entropy_mask), axis=-1)

    num_joins = tf.reduce_sum(tf.reduce_sum(join_matrix, axis=-1), axis=-1)
    num_not_joins = tf.reduce_sum(tf.reduce_sum(not_join_matrix, axis=-1), axis=-1)
    num_entropies = tf.reduce_sum(entropy_mask, axis=-1)

    error = -print_tensor(tf.math.divide_no_nan(positive_error, num_joins))\
            + print_tensor(tf.math.divide_no_nan(negative_error, num_not_joins))#\
            #+ 0.3 * entropy/num_entropies
    return error

def get_initial_probabilities(data):
    num_labels, bbox_overlap = data["num_surfaces"]-1, data["bbox_overlap"]
    num_boxes = np.shape(bbox_overlap)[0]
    potentials_surfaces = np.eye(num_labels, num_labels + num_boxes) * 1
    potentials_surfaces = potentials_surfaces + np.random.uniform(size=np.shape(potentials_surfaces))
    potentials_surfaces = potentials_surfaces / np.sum(potentials_surfaces, axis=-1, keepdims=True)

    potentials_boxes = bbox_overlap.copy()
    potentials_boxes = potentials_boxes + np.random.uniform(size=np.shape(potentials_boxes)) * 0.1
    potentials_boxes = potentials_boxes / np.sum(potentials_boxes, axis=-1, keepdims=True)
    potentials_boxes[np.isnan(potentials_boxes)] = 0
    potentials_boxes = np.pad(potentials_boxes, ((0, 0), (0, num_boxes)))

    potentials = np.concatenate([potentials_surfaces, potentials_boxes], axis=0)
    return potentials

def get_position_and_occlusion_infos(data):
    avg_positions, patch_points, surfaces, norm_image, num_surfaces = data["avg_pos"], data["patch_points"], data["surfaces"], data["norm"], data["num_surfaces"]
    input = ps.determine_occlusion_line_points(np.ones((num_surfaces, num_surfaces)), patch_points, avg_positions, num_surfaces)
    nearest_points_for_occlusion = ps.calculate_nearest_points(*input)
    join_matrix, close_matrix, closest_points = ps.determine_occlusion_candidates_and_connection_infos(nearest_points_for_occlusion, surfaces, norm_image, num_surfaces)
    return join_matrix, close_matrix, closest_points

def calculate_pairwise_similarity_features_for_surfaces(data, models):
    color_similarity_model, angle_similarity_model, texture_model, object_detector, _ = models

    ps.extract_information_from_surface_data_and_preprocess_surfaces(data, texture_model)

    data["sim_color"] = color_similarity_model(data["hist_color"])
    data["sim_texture"] = ps.calculate_texture_similarities(data["hist_texture"], data["num_surfaces"])
    data["sim_angles"] = angle_similarity_model(data["hist_angles"])/20

    data["planes"] = ps.determine_even_planes(data)
    data["coplanarity"] = ps.determine_coplanarity(np.ones((data["num_surfaces"], data["num_surfaces"])), data)
    data["convex"], data["concave"], _ = assemble_objects_rules.determine_convexly_connected_surfaces(np.ones((data["num_surfaces"], data["num_surfaces"])), data)
    data["neighborhood_mat"] = ps.neighborhood_list_to_matrix(data)
    data["bboxes"] = object_detector(data["rgb"])
    data["bbox_overlap"] = ps.calc_box_and_surface_overlap(data)
    data["bbox_similarities"] = ps.create_bbox_similarity_matrix_from_box_surface_overlap(data["bbox_overlap"])
    data["num_bboxes"] = np.shape(data["bbox_overlap"])[0]
    #detect_objects.show_bounding_boxes(rgb_image, bboxes)
    #plot_surfaces(surfaces)
    #centroid_distances = np.sqrt(np.sum(np.square(np.tile(np.expand_dims(centroids, axis=0), [num_surfaces, 1, 1]) - np.tile(np.expand_dims(centroids, axis=1), [1, num_surfaces, 1])), axis=-1))/1000
    data["occlusion_mat"], data["close_mat"], data["closest_points"] = get_position_and_occlusion_infos(data)
    data["close_mat"][data["neighborhood_mat"] == 1] = 0
    data["distances"] = np.sqrt(np.sum(np.square(data["closest_points"] - np.swapaxes(data["closest_points"], axis1=0, axis2=1)), axis=-1))/500
    data["depth_extend_distances"] = ps.create_depth_extend_distance_matrix(data)/1000

def plot_prediction(prediction, surfaces):
    max_vals = np.argmax(prediction[0], axis=-1)
    for i in range(len(max_vals)):
        if max_vals[i] != i:
            surfaces[surfaces == i+1] = max_vals[i]+1
    assemble_objects_rules.plot_surfaces(surfaces)

def assemble_objects_CRF(data, models, train=False):
    calculate_pairwise_similarity_features_for_surfaces(data, models)

    CRF_model = models[-1]

    Q_in = get_initial_probabilities(data)
    unary_in = Q_in.copy()
    unary_in[data["num_surfaces"]-1:] = -1000

    input = [np.asarray([e]) for e in [Q_in, data["convexity"], data["sim_color"], data["sim_texture"],
                                       data["coplanarity"], data["sim_bboxes"], data["neighborhood_mat"]]]

    if not train:
        prediction = CRF_model.predict(input)
        plot_prediction(prediction, data["surfaces"])
    else:
        Y = get_Y_value(data)
        #s, l = assemble_objects.join_surfaces_according_to_join_matrix(join_matrix, surfaces, num_labels+1)
        #assemble_objects.plot_surfaces(s)
        CRF_model.fit(input, Y, epochs=700)
        print_parameters(CRF_model)
        prediction = CRF_model.predict(input)
        plot_prediction(prediction, data["surfaces"])

def get_GPU_models():
    return chi_squared_distances_model_2D((10, 10), (4, 4)), \
           chi_squared_distances_model_2D((4, 4), (1, 1)), \
           extract_texture_function(), \
           detect_objects.get_object_detector(), \
           mean_field_CRF(*initial_guess_parameters)

def main():
    models = get_GPU_models()
    train = True
    index = 4
    while True:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        assemble_objects_CRF(data, models, train)
        train = False
        index += 1

if __name__ == '__main__':
    #create_training_set()
    #train_unary_classifier()
    #quit()
    #assemble_objects_with_unary_classifier()
    #quit()
    #train_unary_classifier()
    quit()
    main()