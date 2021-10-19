import pickle
import numpy as np
import assemble_objects_rules
import detect_objects
import tensorflow as tf
import load_images
import process_surfaces as ps
import standard_values
from image_processing_models_GPU import Variable2, extract_texture_function, chi_squared_distances_model_1D,\
                                        chi_squared_distances_model_2D, print_tensor, Variable
from tensorflow.keras import layers, optimizers, Model, losses
from standard_values import *

#initial_guess_parameters = [1.1154784, 2, 2, 1, 1.1, 3, -1, 2, 1.0323303, 0.24122038]
initial_guess_parameters = [1, 2, 2, 1, 1.1, 0.02]

variable_names = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7", "w_8", "w_9", "weight"]

def create_CRF_training_set():
    models = get_GPU_models()
    Q_list = []
    features_list = []
    counts_list = []
    Y_list = []
    for index in standard_values.train_indices:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        calculate_pairwise_similarity_features_for_surfaces(data, models)
        similarity_matrix = create_similarity_feature_matrix(data)

        Q_in = get_initial_probabilities(data)
        Y = get_Y_value(data)

        Q_list.append(Q_in)
        Y_list.append(Y)
        features_list.append(similarity_matrix)
        counts_list.append(np.asarray([data["num_surfaces"]-1 + data["num_bboxes"], data["num_bboxes"]]))

    pickle.dump(features_list, open("data/train_object_assemble_CRF_dataset/features.pkl", "wb"))
    pickle.dump(Y_list, open("data/train_object_assemble_CRF_dataset/Y.pkl", "wb"))
    pickle.dump(counts_list, open("data/train_object_assemble_CRF_dataset/counts.pkl", "wb"))
    pickle.dump(Q_list, open("data/train_object_assemble_CRF_dataset/Q.pkl", "wb"))


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

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(pairwise_potentials, axis=-1), Q_mult), matrix_1), axis=2)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-1)
    new_Q = tf.exp(tf.multiply(unary_potentials, 0.5) - tf.multiply(compatibility, 0.2))
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
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-3), metrics=[], run_eagerly=True)
    return model

def mean_field_CRF_test(feature_weight, LR_bias, weight, num_iterations=1):
    Q_in = layers.Input(shape=(None, None), name="Q")
    features = layers.Input(shape=(None, None, None), name="features")
    counts = layers.Input(shape=(2,), dtype=tf.int32, name="counts")

    linear_LR = tf.reduce_sum(layers.Multiply()([Variable2(feature_weight, name="feature_weight")(features), features]), axis=-1)
    features_LR = tf.sigmoid(layers.Add()([Variable2(LR_bias, name="LR_bias")(linear_LR), linear_LR])) - 0.5

    num_labels = counts[0, 0]
    num_boxes = counts[0, 1]

    pairwise_potentials = tf.pad(features_LR, [[0, 0], [0, num_boxes], [0, num_boxes]])

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels) *
                                                                                                tf.cast(num_labels-num_boxes+ 1, dtype=tf.float32) *
                                                                                                tf.squeeze(Variable(np.asarray([[1]]), name="weight_1")(features_LR), axis=1), axis=0), axis=0)

    weight = Variable2(weight, name="weight_2")(Q_in)
    Q = Q_in
    for _ in range(num_iterations):
        Q = mean_field_iteration(Q_in, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight)

    model = Model(inputs=[Q_in, features, counts], outputs=Q)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=3e-2), metrics=[], run_eagerly=True)

    return model

def custom_loss(y_actual, y_predicted):
    join_matrix = tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1)
    not_join_matrix = tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1)
    num_outputs = tf.cast(tf.shape(y_predicted)[1], dtype=tf.float32)

    num_labels = tf.shape(y_predicted)[2]

    prediction_expanded_1 = tf.tile(tf.expand_dims(y_predicted, axis=2), [1, 1, num_labels, 1, 1])
    prediction_expanded_2 = tf.tile(tf.expand_dims(y_predicted, axis=3), [1, 1, 1, num_labels, 1])
    mult_out = tf.multiply(prediction_expanded_1, prediction_expanded_2)
    sum_out = tf.reduce_sum(mult_out, axis=-1)

    positive_error = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(layers.ELU(alpha=0.1)(0.4 - tf.multiply(join_matrix, sum_out)), axis=-1),axis=-1), axis=-1)
    negative_error = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.multiply(layers.ELU(alpha=0.1)(not_join_matrix - 0.05), sum_out), axis=-1),axis=-1), axis=-1)

    num_joins = tf.reduce_sum(tf.reduce_sum(join_matrix, axis=-1), axis=-1) * num_outputs
    num_not_joins = tf.reduce_sum(tf.reduce_sum(not_join_matrix, axis=-1), axis=-1) * num_outputs

    error = tf.math.divide_no_nan(positive_error, num_joins)\
            + tf.math.divide_no_nan(negative_error, num_not_joins)
    return error

def get_initial_probabilities(data):
    num_labels, bbox_overlap = data["num_surfaces"]-1, data["bbox_overlap"]
    num_boxes = np.shape(bbox_overlap)[0]
    potentials_surfaces = np.eye(num_labels, num_labels + num_boxes)
    potentials_surfaces = potentials_surfaces + np.random.uniform(size=np.shape(potentials_surfaces)) * 0.02
    potentials_surfaces = potentials_surfaces / np.sum(potentials_surfaces, axis=-1, keepdims=True)

    potentials_boxes = bbox_overlap[:, 1:].copy()
    potentials_boxes = potentials_boxes + np.random.uniform(size=np.shape(potentials_boxes)) * 0.02
    potentials_boxes = potentials_boxes / np.sum(potentials_boxes, axis=-1, keepdims=True)
    potentials_boxes[np.isnan(potentials_boxes)] = 0
    potentials_boxes = np.pad(potentials_boxes, ((0, 0), (0, num_boxes)))

    potentials = np.concatenate([potentials_surfaces, potentials_boxes], axis=0)
    return potentials

def get_position_and_occlusion_infos(data):
    avg_positions, patch_points, surfaces, norm_image, num_surfaces = data["avg_pos"], data["patch_points"], data["surfaces"], data["norm"], data["num_surfaces"]
    input = ps.determine_occlusion_line_points(np.ones((num_surfaces, num_surfaces)), patch_points, avg_positions, num_surfaces)
    nearest_points_for_occlusion = ps.calculate_nearest_points(*input)
    join_matrix, close_matrix, closest_points = ps.determine_occlusion_candidates_and_connection_infos(nearest_points_for_occlusion, data)
    return join_matrix, close_matrix, closest_points

def calculate_pairwise_similarity_features_for_surfaces(data, models):
    color_similarity_model, angle_similarity_model, texture_model, object_detector, _ = models

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
    data["bbox_similarity_matrix"] = ps.create_bbox_similarity_matrix_from_box_surface_overlap(data["bbox_overlap"])
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

def plot_prediction(prediction, surfaces):
    max_vals = np.argmax(prediction, axis=-1)
    for i in range(len(max_vals)):
        if max_vals[i] != i:
            surfaces[surfaces == i+1] = max_vals[i]+1
    assemble_objects_rules.plot_surfaces(surfaces)

def train_CRF():
    features = pickle.load(open("data/train_object_assemble_CRF_dataset/features.pkl", "rb"))
    Y = pickle.load(open("data/train_object_assemble_CRF_dataset/Y.pkl", "rb"))
    counts = pickle.load(open("data/train_object_assemble_CRF_dataset/counts.pkl", "rb"))
    Q = pickle.load(open("data/train_object_assemble_CRF_dataset/Q.pkl", "rb"))

    def gen():
        for i in range(len(standard_values.train_indices)):
            yield {"Q": Q[i], "features": features[i], "counts": counts[i]}, tf.squeeze(Y[i], axis=0)

    dataset = tf.data.Dataset.from_generator(gen, output_types=({"Q": tf.float32, "features": tf.float32, "counts": tf.int32}, tf.float32))
    #p = assemble_objects_pairwise_clf.load_clf("LR")
    # CRF = mean_field_CRF_test(np.reshape(p.coef_, (1, 12)), np.asarray([[p.intercept_]]), np.asarray([[0.1]]), 1)
    # print(CRF.evaluate(dataset.batch(1), verbose=1))
    #
    # CRF = mean_field_CRF_test(np.ones((1, 12)), np.asarray([[0]]), np.asarray([[0.1]]), 1)
    # print(CRF.evaluate(dataset.batch(1), verbose=1))
    #print_parameters(CRF, ["feature_weight", "LR_bias", "weight_1", "weight_2"])
    CRF = mean_field_CRF_test(np.ones((1, 12)), np.asarray([[0]]), np.asarray([[0.1]]), 3)
    CRF.fit(dataset.batch(1), verbose=1, epochs=20)
    print_parameters(CRF, ["feature_weight", "LR_bias", "weight_1", "weight_2"])

    quit()
    for x, y in dataset:
        h = CRF.predict([np.asarray([k]) for k in x.values()], batch_size=1)
        print()
        quit()
    quit()

def assemble_objects_CRF(data, models, train=False):
    calculate_pairwise_similarity_features_for_surfaces(data, models)
    similarity_matrix = create_similarity_feature_matrix(data)

    CRF_model = models[-1]

    Q_in = get_initial_probabilities(data)
    unary_in = Q_in.copy()
    unary_in[data["num_surfaces"]-1:] = -1000

    input = [np.asarray([e]) for e in [Q_in, data["convexity"], data["sim_color"], data["sim_texture"],
                                       data["coplanarity"], data["bbox_crf_matrix"], data["neighborhood_mat"]]]

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
    create_CRF_training_set()
    quit()
    train_CRF()
    quit()
    main()