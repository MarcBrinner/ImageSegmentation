import pickle
import random
import numpy as np
import detect_objects
import tensorflow as tf
import load_images
import plot_image
import process_surfaces as ps
import standard_values
from image_processing_models_GPU import extract_texture_function, chi_squared_distances_model_2D
from find_planes import find_surface_model
from tensorflow.keras import layers, optimizers, Model, losses, regularizers, initializers
from standard_values import *

variable_names = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7", "w_8", "w_9", "weight"]
clf_types = ["LR", "Neural"]

def get_GPU_models():
    return chi_squared_distances_model_2D((10, 10), (4, 4)), \
           chi_squared_distances_model_2D((4, 4), (1, 1)), \
           extract_texture_function(), \
           detect_objects.get_object_detector()

def create_CRF_training_set():
    models = get_GPU_models()
    bbox_overlap_list = []
    features_list = []
    Y_list = []
    for index in standard_values.train_indices:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        ps.calculate_pairwise_similarity_features_for_surfaces(data, models)
        similarity_matrix = create_similarity_feature_matrix(data)

        Y = ps.find_optimal_surface_joins(data)
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

def neural_network_model(num_features = standard_values.num_pairwise_features):
    input = layers.Input(shape=(None, None, num_features))
    d_1 = layers.Dropout(0)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.0001), activation="relu", name="dense_1")(input))
    d_2 = layers.Dropout(0)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.0001), activation="relu", name="dense_2")(d_1))
    out = tf.squeeze(layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0001), name="dense_3")(d_2), axis=-1)
    model = Model(input, out)
    return model

def get_pairwise_clf(clf_type):
    if clf_type not in clf_types:
        print("Clf type not recognized.")
        quit()
    if clf_type == "LR":
        return LR_model()
    elif clf_type == "Neural":
        return neural_network_model()
    return None

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(pairwise_potentials, axis=-1), Q_mult), matrix_1), axis=2)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-1)
    compatibility = compatibility - tf.reduce_min(compatibility, axis=-1, keepdims=True)
    compatibility = tf.math.divide_no_nan(compatibility, tf.reduce_max(compatibility)) * 15
    new_Q = tf.exp(tf.multiply(unary_potentials, 0.5) - compatibility)
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_CRF(feature_processing_model, num_iterations=60, num_features=standard_values.num_pairwise_features):
    features = layers.Input(shape=(None, None, num_features), name="features")
    boxes = layers.Input(shape=(None, None), name="boxes")

    num_surfaces = tf.shape(boxes)[2]
    num_boxes = tf.shape(boxes)[1]
    num_labels = num_surfaces + num_boxes

    boxes_dense = tf.squeeze(layers.Dense(1, kernel_initializer=initializers.constant(value=0), bias_initializer=initializers.constant(value=0))(tf.expand_dims(boxes, axis=-1)), axis=-1)
    boxes_pad = tf.pad(boxes_dense, [[0, 0], [num_surfaces, 0], [0, num_boxes]])
    boxes_pad = boxes_pad + tf.transpose(boxes_pad, perm=[0, 2, 1])

    pairwise_potentials_pred = feature_processing_model(features)
    initial_Q = pairwise_potentials_pred * (1 - tf.eye(num_surfaces)) + tf.eye(num_surfaces)
    initial_Q = tf.math.divide_no_nan(initial_Q, tf.reduce_sum(initial_Q, axis=-1, keepdims=True))
    initial_Q = tf.pad(initial_Q, [[0, 0], [0, num_boxes], [0, num_boxes]])

    pairwise_potentials = pairwise_potentials_pred - 0.5
    pairwise_potentials = tf.pad(pairwise_potentials, [[0, 0], [0, num_boxes], [0, num_boxes]])  + boxes_pad
    pairwise_potentials_pred = tf.pad(pairwise_potentials_pred, [[0, 0], [0, num_boxes], [0, num_boxes]])

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)

    Q = initial_Q
    Q_acc = tf.stack([pairwise_potentials_pred, Q], axis=1)
    for _ in range(num_iterations):
        Q = 0.9 * Q + 0.1 * mean_field_iteration(initial_Q, pairwise_potentials, Q, num_labels, matrix_1, matrix_2)
        Q_acc = tf.concat([Q_acc, tf.expand_dims(Q, axis=1)], axis=1)

    model = Model(inputs=[features, boxes], outputs=Q_acc)
    model.compile(loss=custom_loss_new_last, optimizer=optimizers.Adam(learning_rate=3e-2), metrics=[], run_eagerly=False)

    return model

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

def train_CRF(clf_type="Neural"):
    features = pickle.load(open("data/train_object_assemble_CRF_dataset/features.pkl", "rb"))
    Y = pickle.load(open("data/train_object_assemble_CRF_dataset/Y.pkl", "rb"))
    boxes = pickle.load(open("data/train_object_assemble_CRF_dataset/bbox.pkl", "rb"))

    def gen():
        indices = list(range(len(features)))
        random.shuffle(indices)
        for i in indices:
            yield {"features": features[i], "boxes": boxes[i]}, tf.squeeze(Y[i], axis=0)

    dataset = tf.data.Dataset.from_generator(gen, output_types=({"features": tf.float32, "boxes": tf.float32}, tf.float32))

    pw_clf = get_pairwise_clf(clf_type)
    #lr_clf = pickle.load(open(f"parameters/pairwise_surface_clf/clf_LR.pkl", "rb"))
    #pw_clf.get_layer("dense").set_weights([lr_clf.coef_.reshape(14, 1), lr_clf.intercept_])
    #print(lr_clf.coef_.reshape(14, 1), lr_clf.intercept_)
    #nn = assemble_objects_pairwise_clf.get_nn_clf()
    pw_clf.load_weights("parameters/pairwise_surface_clf/clf_Neural.ckpt")
    #for name in ["dense_1", "dense_2", "dense_3"]:
    #    pw_clf.get_layer(name).set_weights(nn.get_layer(name).get_weights())
    CRF = mean_field_CRF(pw_clf)
    #CRF.load_weights("test.ckpt")
    #print(CRF.trainable_weights)
    #CRF.fit(dataset.batch(1), verbose=1, epochs=15)
    #print(CRF.trainable_weights)

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
    ps.calculate_pairwise_similarity_features_for_surfaces(data, models)
    similarity_matrix = ps.create_similarity_feature_matrix(data)

    inputs = [np.asarray([x]) for x in [similarity_matrix, data["bbox_overlap"][:, 1:]]]

    prediction = crf.predict(inputs)
    data["final_surfaces"] = ps.create_surfaces_from_crf_output(prediction[0, -1], data["surfaces"])
    return data

def assemble_objects_for_indices(indices=standard_values.test_indices, clf_type="LR", plot=True):
    models = get_GPU_models()

    pw_clf = get_pairwise_clf(clf_type)

    crf = mean_field_CRF(pw_clf)
    crf.load_weights(f"data/CRF_pairwise_clf/{clf_type}.ckpt")
    results = []
    for index in indices:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        data = assemble_objects_crf(crf, models, data)

        if plot:
            plot_image.plot_surfaces(data["final_surfaces"])

        results.append(data["final_surfaces"])
    return results

def get_full_prediction_model(clf_type="LR"):
    surface_model = find_surface_model()
    assemble_surface_models = get_GPU_models()
    pw_clf = get_pairwise_clf(clf_type)

    crf = mean_field_CRF(pw_clf)
    crf.load_weights(f"data/CRF_pairwise_clf/{clf_type}.ckpt")
    return lambda x, y: assemble_objects_crf(crf, assemble_surface_models, load_images.load_image_and_surface_information(y))


if __name__ == '__main__':
    #predict_indices([100], clf_type="Neural")
    #quit()
    train_CRF("Neural")