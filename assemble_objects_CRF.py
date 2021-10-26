import pickle
import random

import numpy as np

import assemble_objects_rules
import detect_objects
import tensorflow as tf
import load_images
import plot_image
import process_surfaces as ps
import standard_values
from image_processing_models_GPU import Variable2, extract_texture_function, chi_squared_distances_model_1D,\
                                        chi_squared_distances_model_2D, print_tensor, Variable
from find_planes import get_find_surface_function
from tensorflow.keras import layers, optimizers, Model, losses, regularizers, initializers
from standard_values import *

#initial_guess_parameters = [1.1154784, 2, 2, 1, 1.1, 3, -1, 2, 1.0323303, 0.24122038]
initial_guess_parameters = [1, 2, 2, 1, 1.1, 0.02]

variable_names = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7", "w_8", "w_9", "weight"]
clf_types = ["LR", "Neural"]

def create_CRF_training_set():
    models = get_GPU_models()
    bbox_overlap_list = []
    features_list = []
    Y_list = []
    for index in standard_values.train_indices:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        calculate_pairwise_similarity_features_for_surfaces(data, models)
        similarity_matrix = create_similarity_feature_matrix(data)

        Y = get_Y_value(data)
        bbox_overlap = data["bbox_overlap"][:, 1:]

        bbox_overlap_list.append(bbox_overlap)
        Y_list.append(Y)
        features_list.append(similarity_matrix)

    pickle.dump(features_list, open("data/train_object_assemble_CRF_dataset/features.pkl", "wb"))
    pickle.dump(Y_list, open("data/train_object_assemble_CRF_dataset/Y.pkl", "wb"))
    pickle.dump(bbox_overlap_list, open("data/train_object_assemble_CRF_dataset/bbox.pkl", "wb"))

def LR_model(num_features=standard_values.num_pairwise_features):
    input = layers.Input(shape=(None, None, num_features))
    out = tf.squeeze(layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0001),  name="dense")(input), axis=-1)
    model = Model(input, out)
    return model

def neural_model(num_features = standard_values.num_pairwise_features):
    input = layers.Input(shape=(None, None, num_features))
    d_1 = layers.Dropout(0)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.0001), activation="relu", name="dense_1")(input))
    d_2 = layers.Dropout(0)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.0001), activation="relu", name="dense_2")(d_1))
    out = tf.squeeze(layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0001), name="dense_3")(d_2), axis=-1)
    model = Model(input, out)
    return model

def get_pairwise_model(clf_type):
    if clf_type not in clf_types:
        print("Clf type not recognized.")
        quit()
    if clf_type == "LR":
        return LR_model()
    elif clf_type == "Neural":
        return neural_model()
    return None

def print_parameters(model, variable_names):
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

    Y = np.pad(np.stack([Y_1, Y_2], axis=0), ((0, 0), (0, num_boxes), (0, num_boxes)))
    return np.asarray([Y[:, 1:, 1:]])

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(pairwise_potentials, axis=-1), Q_mult), matrix_1), axis=2)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-1)
    compatibility = compatibility - tf.reduce_min(compatibility, axis=-1, keepdims=True)
    compatibility = tf.math.divide_no_nan(compatibility, tf.reduce_max(compatibility)) * 15
    new_Q = tf.exp(tf.multiply(unary_potentials, 0.5) - compatibility)
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_CRF(box_weight_1, boxes_weight_2, box_bias, weight, LR_weights, LR_bias, num_iterations=10):
    Q_in = layers.Input(shape=(None, None))
    counts = layers.Input(shape=(2,), dtype=tf.int32)
    features = layers.Input(shape=(None, None, None))
    boxes_in = layers.Input(shape=(None, None))
    boxes_sim = (tf.sigmoid(layers.Add()([Variable2(box_bias, name="box_bias")(boxes_in), layers.Multiply()([Variable2(box_weight_1, name="box_weight_1")(boxes_in), boxes_in])])) - 0.5)

    linear_LR = tf.reduce_sum(layers.Multiply()([Variable2(LR_weights, name="LR_weights")(features), features]), axis=-1)
    features_LR = tf.sigmoid(layers.Add()([Variable2(LR_bias, name="LR_bias")(linear_LR), linear_LR]))

    num_labels = counts[0, 0]
    num_boxes = counts[0, 1]

    pairwise_potentials = tf.pad(features_LR, [[0, 0], [0, num_boxes], [0, num_boxes]]) + layers.Multiply()([Variable2(boxes_weight_2, name="box_weight_2")(boxes_sim), boxes_sim])

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)
    weight = Variable2(weight, name="weight")(Q_in)
    Q = Q_in
    for _ in range(num_iterations):
        Q = mean_field_iteration(Q_in, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight)
    output = Q

    model = Model(inputs=[Q_in, features, boxes_in, counts], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=3e-3), metrics=[], run_eagerly=True)
    return model

def mean_field_CRF_test(feature_processing_model, num_iterations=60, num_features=standard_values.num_pairwise_features):
    features = layers.Input(shape=(None, None, num_features), name="features")
    boxes = layers.Input(shape=(None, None), name="boxes")

    num_surfaces = tf.shape(boxes)[2]
    num_boxes = tf.shape(boxes)[1]
    num_labels = num_surfaces + num_boxes

    boxes_dense = tf.squeeze(layers.Dense(1, kernel_initializer=initializers.constant(1), bias_initializer=initializers.constant(-0.2))(tf.expand_dims(boxes, axis=-1)), axis=-1)
    boxes_pad = tf.pad(boxes_dense, [[0, 0], [num_surfaces, 0], [0, num_boxes]])

    pairwise_potentials_pred = feature_processing_model(features)
    initial_Q = pairwise_potentials_pred * (1 - tf.eye(num_surfaces)) + tf.eye(num_surfaces)
    initial_Q = tf.math.divide_no_nan(initial_Q, tf.reduce_sum(initial_Q, axis=-1, keepdims=True))
    initial_Q = tf.pad(initial_Q, [[0, 0], [0, num_boxes], [0, num_boxes]])

    pairwise_potentials = pairwise_potentials_pred - 0.5
    pairwise_potentials = tf.pad(pairwise_potentials, [[0, 0], [0, num_boxes], [0, num_boxes]])
    pairwise_potentials_pred = tf.pad(pairwise_potentials_pred, [[0, 0], [0, num_boxes], [0, num_boxes]])
    pairwise_potentials_pred = pairwise_potentials_pred + boxes_pad

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)

    Q = initial_Q
    Q_acc = tf.stack([pairwise_potentials_pred, Q], axis=1)
    for _ in range(num_iterations):
        Q = 0.9 * Q + 0.1 * mean_field_iteration(initial_Q, pairwise_potentials, Q, num_labels, matrix_1, matrix_2)
        Q_acc = tf.concat([Q_acc, tf.expand_dims(Q, axis=1)], axis=1)

    model = Model(inputs=[features, boxes], outputs=Q_acc)
    model.compile(loss=custom_loss_new_last, optimizer=optimizers.Adam(learning_rate=3e-2), metrics=[], run_eagerly=True)

    return model

def cross_entropy(x, y, axis=-1):
  safe_y = tf.where(tf.equal(x, 0), tf.ones_like(y), y)
  return -tf.reduce_sum(x * tf.math.log(safe_y), axis)

def entropy(x, axis=-1):
  return cross_entropy(x, x, axis)

def KL_divergence(Q, M):
    return cross_entropy(Q, tf.math.divide_no_nan(Q, M))

def custom_loss_new_last(y_actual, y_predicted):
    pred_pot = y_predicted[:, 0, :, :]
    pred_Q = y_predicted[:, 25, :, :]
    join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1))
    not_join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1))
    #num_outputs = tf.cast(tf.shape(pred_Q)[1], dtype=tf.float32)
    #weights = tf.range(num_outputs, dtype=tf.float32)/num_outputs
    #sum_weights = tf.reduce_sum(weights, axis=-1)
    num_labels = tf.shape(pred_Q)[2]

    LR_labels = tf.ones_like(pred_pot) * (1-not_join_matrix)
    LR_weights = 1- (tf.ones_like(pred_pot) * (1- (join_matrix + not_join_matrix)))

    LR_loss = losses.BinaryCrossentropy(from_logits=False)(tf.expand_dims(LR_labels, axis=-1), tf.expand_dims(pred_pot, axis=-1), LR_weights)

    prediction_expanded_1 = tf.tile(tf.expand_dims(pred_Q, axis=1), [1, num_labels, 1, 1])
    prediction_expanded_2 = tf.tile(tf.expand_dims(pred_Q, axis=2), [1, 1, num_labels, 1])
    mult_out = tf.multiply(prediction_expanded_1, prediction_expanded_2)
    sum_out = tf.reduce_sum(mult_out, axis=-1)

    label_0_diff = 1 - sum_out
    label_0_loss = tf.math.log(tf.where(tf.equal(label_0_diff, 0), tf.ones_like(label_0_diff), label_0_diff))

    label_1_loss = tf.math.log(tf.where(tf.equal(sum_out, 0), tf.ones_like(sum_out), sum_out))


    positive_error = -tf.reduce_sum(tf.reduce_sum(tf.math.multiply_no_nan(join_matrix, label_1_loss), axis=-1), axis=-1)
    negative_error = -tf.reduce_sum(tf.reduce_sum(tf.math.multiply_no_nan(not_join_matrix, label_0_loss), axis=-1),axis=-1)

    num_joins = tf.reduce_sum(tf.reduce_sum(join_matrix, axis=-1), axis=-1)
    num_not_joins = tf.reduce_sum(tf.reduce_sum(not_join_matrix, axis=-1), axis=-1)

    Q_loss = (tf.math.divide_no_nan(positive_error, 1)
            + tf.math.divide_no_nan(negative_error, 1))
    return LR_loss

def custom_loss_new(y_actual, y_predicted):
    pred_pot = y_predicted[:, 0, :, :]
    pred_Q = y_predicted[:, 1:, :, :]
    join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1))
    not_join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1))
    num_outputs = tf.cast(tf.shape(pred_Q)[1], dtype=tf.float32)
    weights = tf.range(num_outputs, dtype=tf.float32)/num_outputs
    sum_weights = tf.reduce_sum(weights, axis=-1)
    num_labels = tf.shape(pred_Q)[2]

    LR_labels = tf.ones_like(pred_pot) * (1-not_join_matrix)
    LR_weights = 1- (tf.ones_like(pred_pot) * (1- (join_matrix + not_join_matrix)))

    LR_loss = losses.BinaryCrossentropy(from_logits=False)(tf.expand_dims(LR_labels, axis=-1), tf.expand_dims(pred_pot, axis=-1), LR_weights)

    prediction_expanded_1 = tf.tile(tf.expand_dims(pred_Q, axis=2), [1, 1, num_labels, 1, 1])
    prediction_expanded_2 = tf.tile(tf.expand_dims(pred_Q, axis=3), [1, 1, 1, num_labels, 1])
    mult_out = tf.multiply(prediction_expanded_1, prediction_expanded_2)
    sum_out = tf.reduce_sum(mult_out, axis=-1)

    label_1_diff = 1 - sum_out
    label_1_diff_safe = tf.where(tf.equal(label_1_diff, 0), tf.ones_like(label_1_diff), label_1_diff)
    loss_label_1 = tf.math.multiply_no_nan(tf.math.log(label_1_diff_safe), label_1_diff)

    label_0_diff_safe = tf.where(tf.equal(sum_out, 0), tf.ones_like(sum_out), sum_out)
    loss_label_0 = tf.math.multiply_no_nan(tf.math.log(label_0_diff_safe), sum_out)


    positive_error = -tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.multiply(join_matrix, loss_label_1), axis=-1), axis=-1)*weights, axis=-1)
    negative_error = -tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.multiply(not_join_matrix, loss_label_0), axis=-1),axis=-1)*weights, axis=-1)

    num_joins = tf.reduce_sum(tf.reduce_sum(join_matrix, axis=-1), axis=-1)
    num_not_joins = tf.reduce_sum(tf.reduce_sum(not_join_matrix, axis=-1), axis=-1)

    Q_loss = (tf.math.divide_no_nan(positive_error, num_joins * sum_weights)
            + tf.math.divide_no_nan(negative_error, num_not_joins * sum_weights)) * 10
    return Q_loss# + LR_loss*5

def custom_loss(y_actual, y_predicted):
    pred_pot = y_predicted[:, 0, :, :]
    pred_Q = y_predicted[:, 1:, :, :]
    join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1))
    not_join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1))
    num_outputs = tf.cast(tf.shape(pred_Q)[1], dtype=tf.float32)
    weights = tf.range(num_outputs, dtype=tf.float32)/num_outputs
    sum_weights = tf.reduce_sum(weights, axis=-1)
    num_labels = tf.shape(pred_Q)[2]

    LR_labels = tf.ones_like(pred_pot) * (1-not_join_matrix)
    LR_weights = 1- (tf.ones_like(pred_pot) * (1- (join_matrix + not_join_matrix)))

    LR_loss = losses.BinaryCrossentropy(from_logits=False)(tf.expand_dims(LR_labels, axis=-1), tf.expand_dims(pred_pot, axis=-1), LR_weights)

    prediction_expanded_1 = tf.tile(tf.expand_dims(pred_Q, axis=2), [1, 1, num_labels, 1, 1])
    prediction_expanded_2 = tf.tile(tf.expand_dims(pred_Q, axis=3), [1, 1, 1, num_labels, 1])
    mult_out = tf.multiply(prediction_expanded_1, prediction_expanded_2)
    sum_out = tf.reduce_sum(mult_out, axis=-1)

    positive_error = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(tf.multiply(join_matrix, 1 - sum_out)), axis=-1), axis=-1)*weights, axis=-1)
    negative_error = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(tf.multiply(not_join_matrix, sum_out)), axis=-1),axis=-1)*weights, axis=-1)

    num_joins = tf.reduce_sum(tf.reduce_sum(join_matrix, axis=-1), axis=-1)
    num_not_joins = tf.reduce_sum(tf.reduce_sum(not_join_matrix, axis=-1), axis=-1)

    Q_loss = (tf.math.divide_no_nan(positive_error, num_joins * sum_weights)
            + tf.math.divide_no_nan(negative_error, num_not_joins * sum_weights)*10)
    return Q_loss + LR_loss*5

def get_position_and_occlusion_infos(data):
    avg_positions, patch_points, surfaces, norm_image, num_surfaces = data["avg_pos"], data["patch_points"], data["surfaces"], data["norm"], data["num_surfaces"]
    input = ps.determine_occlusion_line_points(np.ones((num_surfaces, num_surfaces)), patch_points, avg_positions, num_surfaces)
    nearest_points_for_occlusion = ps.calculate_nearest_points(*input)
    join_matrix, close_matrix, closest_points = ps.determine_occlusion_candidates_and_connection_infos(nearest_points_for_occlusion, data)
    return join_matrix, close_matrix, closest_points

def calculate_pairwise_similarity_features_for_surfaces(data, models):
    color_similarity_model, angle_similarity_model, texture_model, object_detector = models[:4]

    ps.extract_information_from_surface_data_and_preprocess_surfaces(data, texture_model)

    candidates = {"convexity": np.ones((data["num_surfaces"], data["num_surfaces"]))}
    data["sim_color"] = color_similarity_model(data["hist_color"])
    data["sim_texture"] = ps.calculate_texture_similarities(data["hist_texture"], data["num_surfaces"])
    data["sim_angle"] = angle_similarity_model(data["hist_angle"])/20

    data["planes"] = ps.determine_even_planes(data)
    data["same_plane_type"] = (np.tile(np.expand_dims(data["planes"], axis=0), [data["num_surfaces"], 1]) == np.tile(np.expand_dims(data["planes"], axis=-1), [1, data["num_surfaces"]])).astype("int32")
    data["both_curved"] = (lambda x: np.dot(np.transpose(x), x))(np.expand_dims(1-data["planes"], axis=0))
    data["coplanarity"] = ps.determine_coplanarity(candidates["convexity"], data["centroids"].astype("float64"), data["avg_normals"], data["planes"], data["num_surfaces"])
    data["convex"], data["concave"], _ = assemble_objects_rules.determine_convexly_connected_surfaces(candidates["convexity"], data)
    data["neighborhood_mat"] = ps.neighborhood_list_to_matrix(data)
    data["bboxes"] = object_detector(data["rgb"])
    data["bbox_overlap"] = ps.calc_box_and_surface_overlap(data)
    data["bbox_crf_matrix"] = ps.create_bbox_CRF_matrix(data["bbox_overlap"])
    data["bbox_similarity_matrix"] = ps.create_bbox_similarity_matrix_from_box_surface_overlap(data["bbox_overlap"], data["bboxes"])
    data["num_bboxes"] = np.shape(data["bbox_overlap"])[0]
    #detect_objects.show_bounding_boxes(rgb_image, bboxes)
    #plot_surfaces(surfaces)
    #centroid_distances = np.sqrt(np.sum(np.square(np.tile(np.expand_dims(centroids, axis=0), [num_surfaces, 1, 1]) - np.tile(np.expand_dims(centroids, axis=1), [1, num_surfaces, 1])), axis=-1))/1000
    data["occlusion_mat"], data["close_mat"], data["closest_points"] = get_position_and_occlusion_infos(data)
    data["close_mat"][data["neighborhood_mat"] == 1] = 0
    data["distances"] = np.sqrt(np.sum(np.square(data["closest_points"] - np.swapaxes(data["closest_points"], axis1=0, axis2=1)), axis=-1))/500
    data["depth_extend_distances"] = ps.create_depth_extend_distance_matrix(data)/1000

def create_similarity_feature_matrix(data):
    keys = ["bbox_similarity_matrix", "sim_texture", "sim_color", "sim_angle", "convex", "coplanarity", "neighborhood_mat",
            "concave", "distances", "close_mat", "occlusion_mat", "depth_extend_distances", "same_plane_type", "both_curved"]
    matrix = np.stack([data[key][1:,1:] for key in keys], axis=-1)
    return matrix

def create_surfaces_from_crf_output(prediction, surfaces):
    # num_labels = np.shape(prediction)[-1]
    # t_1 = np.tile(np.expand_dims(prediction, axis=0), [num_labels, 1, 1])
    # t_2 = np.tile(np.expand_dims(prediction, axis=1), [1, num_labels, 1])
    # m = np.multiply(t_1, t_2)
    # s = np.sum(m, axis=-1)
    # join_matrix = np.zeros_like(s)
    # join_matrix[s > 0.5] = 1
    # new_surfaces, _ = assemble_objects_rules.join_surfaces_according_to_join_matrix(join_matrix, surfaces, num_labels+1)
    max_vals = np.argmax(prediction, axis=-1)
    new_surfaces = surfaces.copy()
    for i in range(len(max_vals)):
        if max_vals[i] != i:
            new_surfaces[new_surfaces == i+1] = max_vals[i]+1
    return new_surfaces

def train_CRF(clf_type="Neural"):
    features = pickle.load(open("data/train_object_assemble_CRF_dataset/features.pkl", "rb"))
    Y = pickle.load(open("data/train_object_assemble_CRF_dataset/Y.pkl", "rb"))
    boxes = pickle.load(open("data/train_object_assemble_CRF_dataset/bbox.pkl", "rb"))

    # for i in range(len(train_indices)):
    #     y = Y[i]
    #     y[0, 0, :, :] = y[0, 0, :, :] + np.transpose(y[0, 0, :, :])
    #     y[0, 1, :, :] = y[0, 1, :, :] + np.transpose(y[0, 1, :, :])
    #     Y[i] = y
    def gen():
        n = len(train_indices)
        #indices = list(range(n))
        indices = list(range(n))# + list(range(n-15, n)) + list(range(n-7, n))
        random.shuffle(indices)
        for i in indices:
            yield {"features": features[i], "boxes": boxes[i]}, tf.squeeze(Y[i], axis=0)

    dataset = tf.data.Dataset.from_generator(gen, output_types=({"features": tf.float32, "boxes": tf.float32}, tf.float32))

    pw_clf = get_pairwise_model(clf_type)
    #lr_clf = pickle.load(open(f"parameters/pairwise_surface_clf/clf_LR.pkl", "rb"))
    #pw_clf.get_layer("dense").set_weights([lr_clf.coef_.reshape(14, 1), lr_clf.intercept_])
    #print(lr_clf.coef_.reshape(14, 1), lr_clf.intercept_)
    #nn = assemble_objects_pairwise_clf.get_nn_clf()
    #pw_clf.load_weights("parameters/pairwise_surface_clf/clf_Neural.ckpt")
    #for name in ["dense_1", "dense_2", "dense_3"]:
    #    pw_clf.get_layer(name).set_weights(nn.get_layer(name).get_weights())
    CRF = mean_field_CRF_test(pw_clf)
    #CRF.load_weights("test.ckpt")
    CRF.fit(dataset.batch(1), verbose=1, epochs=15)
    print(CRF.trainable_weights)

    # CRF = mean_field_CRF_test(LR_model())
    # print(CRF.evaluate(dataset.batch(1), verbose=1))
    #
    CRF.save_weights(f"data/CRF_pairwise_clf/{clf_type}.ckpt")
    # print(CRF.trainable_weights)
    quit()
    for x, y in dataset:
        h = CRF.predict([np.asarray([k]) for k in x.values()], batch_size=1)
        print()
        quit()
    quit()

def assemble_objects_crf(crf, models, data):
    calculate_pairwise_similarity_features_for_surfaces(data, models)
    similarity_matrix = create_similarity_feature_matrix(data)

    inputs = [np.asarray([x]) for x in [similarity_matrix, data["bbox_overlap"][:, 1:]]]

    pw_clf_2 = LR_model()
    lr_clf = pickle.load(open(f"parameters/pairwise_surface_clf/clf_LR.pkl", "rb"))
    pw_clf_2.get_layer("dense").set_weights([lr_clf.coef_.reshape(14, 1), lr_clf.intercept_])
    crf_2 = mean_field_CRF_test(pw_clf_2)

    prediction = crf.predict(inputs)
    prediction2 = crf_2.predict(inputs)
    data["final_surfaces"] = create_surfaces_from_crf_output(prediction[0, -1], data["surfaces"])
    return data

def get_GPU_models():
    return chi_squared_distances_model_2D((10, 10), (4, 4)), \
           chi_squared_distances_model_2D((4, 4), (1, 1)), \
           extract_texture_function(), \
           detect_objects.get_object_detector(), \
           mean_field_CRF_test(LR_model(14), 60)

def get_prediction_model(clf_type="LR"):
    surface_model = get_find_surface_function()
    assemble_surface_models = get_GPU_models()
    pw_clf = get_pairwise_model(clf_type)

    crf = mean_field_CRF_test(pw_clf)
    crf.load_weights(f"data/CRF_pairwise_clf/{clf_type}.ckpt")
    print(crf.get_layer("boxes_sub").weights)
    return lambda x, y: assemble_objects_crf(crf, assemble_surface_models, load_images.load_image_and_surface_information(y))

def predict_indices(indices=standard_values.test_indices, clf_type="LR"):
    models = get_GPU_models()

    pw_clf = get_pairwise_model(clf_type)

    #lr_clf = pickle.load(open(f"parameters/pairwise_surface_clf/clf_LR.pkl", "rb"))
    #pw_clf.get_layer("dense").set_weights([lr_clf.coef_.reshape(14, 1), lr_clf.intercept_])

    crf = mean_field_CRF_test(pw_clf)
    crf.load_weights(f"data/CRF_pairwise_clf/{clf_type}.ckpt")
    for index in indices:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        data = assemble_objects_crf(crf, models, data)
        plot_image.plot_surfaces(data["surfaces"])
        plot_image.plot_surfaces(data["final_surfaces"])

if __name__ == '__main__':
    train_CRF("Neural")