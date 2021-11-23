import load_images
import plot_image
import post_processing
import process_surfaces as ps
import config
import numpy as np
import tensorflow as tf
import CRF_tools as crf_tools
from find_surfaces import find_surface_model
from tensorflow.keras import layers, optimizers, Model, regularizers, initializers

variable_names = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7", "w_8", "w_9", "weight"]
clf_types = ["LR", "Neural"]

def LR_model(num_features=config.num_pairwise_features):
    input = layers.Input(shape=(None, None, num_features))
    out = tf.squeeze(layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0001),  name="dense")(input), axis=-1)
    model = Model(input, out)
    return model

def neural_network_model(num_features = config.num_pairwise_features):
    input = layers.Input(shape=(None, None, num_features))
    d_1 = layers.Dropout(0.2)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.005), activation="relu", name="dense_1")(input))
    d_2 = layers.Dropout(0.2)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.005), activation="relu", name="dense_2")(d_1))
    out = tf.squeeze(layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.005), name="dense_3")(d_2), axis=-1)
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

def mean_field_CRF(feature_processing_model, num_iterations=60, num_features=config.num_pairwise_features, use_boxes=True):
    features = layers.Input(shape=(None, None, num_features), name="features")
    boxes = layers.Input(shape=(None, None), name="boxes")

    num_surfaces = tf.shape(boxes)[2]
    num_boxes = tf.shape(boxes)[1]
    num_labels = num_surfaces + num_boxes

    boxes_dense = tf.squeeze(layers.Dense(1, kernel_initializer=initializers.constant(value=1), bias_initializer=initializers.constant(value=-0.5))(tf.expand_dims(boxes, axis=-1)), axis=-1)
    boxes_pad = tf.pad(boxes_dense, [[0, 0], [num_surfaces, 0], [0, num_boxes]])
    boxes_pad = boxes_pad + tf.transpose(boxes_pad, perm=[0, 2, 1])

    pairwise_potentials_pred = feature_processing_model(features)
    pairwise_potentials = pairwise_potentials_pred * 0.99 + 0.005
    pairwise_potentials = tf.math.log(tf.math.divide_no_nan(pairwise_potentials, 1-pairwise_potentials))
    pairwise_potentials = tf.pad(pairwise_potentials, [[0, 0], [0, num_boxes], [0, num_boxes]])

    initial_Q = pairwise_potentials_pred * (1 - tf.eye(num_surfaces)) + tf.eye(num_surfaces)
    initial_Q = tf.math.divide_no_nan(initial_Q, tf.reduce_sum(initial_Q, axis=-1, keepdims=True))
    initial_Q = tf.pad(initial_Q, [[0, 0], [0, num_boxes], [0, num_boxes]])

    if use_boxes:
        pairwise_potentials = pairwise_potentials + boxes_pad
    else:
        pairwise_potentials = pairwise_potentials + 0 * boxes_pad
    pairwise_potentials_pred = tf.pad(pairwise_potentials_pred, [[0, 0], [0, num_boxes], [0, num_boxes]])

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)

    Q = initial_Q
    Q_acc = tf.stack([pairwise_potentials_pred, Q], axis=1)
    for _ in range(num_iterations):
        Q = 0.9 * Q + 0.1 * mean_field_iteration(initial_Q, pairwise_potentials, Q, num_labels, matrix_1, matrix_2)
        Q_acc = tf.concat([Q_acc, tf.expand_dims(Q, axis=1)], axis=1)

    model = Model(inputs=[features, boxes], outputs=Q_acc)
    model.compile(loss=crf_tools.custom_loss_CRF, optimizer=optimizers.Adam(learning_rate=optimizers.schedules.ExponentialDecay(3e-2, decay_steps=50, decay_rate=0.95)), metrics=[], run_eagerly=False)

    return model

def train_CRF(clf_type, use_boxes=True):
    pw_clf = get_pairwise_clf(clf_type)
    CRF = mean_field_CRF(pw_clf, use_boxes=use_boxes)
    crf_tools.train_CRF(CRF, f"parameters/CRF_trained_with_clf/{clf_type}.ckpt")

def assemble_objects(crf, models, data):
    ps.calculate_pairwise_similarity_features_for_surfaces(data, models)
    similarity_matrix = ps.create_similarity_feature_matrix(data)

    inputs = [np.asarray([x]) for x in [similarity_matrix, data["bbox_overlap"][:, 1:]]]

    prediction = crf.predict(inputs)
    data["final_surfaces"] = crf_tools.create_surfaces_from_crf_output(prediction[0, -1], data["surfaces"])
    return data

def assemble_objects_for_indices(indices=config.test_indices, clf_type="LR", plot=True):
    models = crf_tools.get_GPU_models()

    pw_clf = get_pairwise_clf(clf_type)

    crf = mean_field_CRF(pw_clf)
    crf.load_weights(f"data/CRF_pairwise_clf/{clf_type}.ckpt")
    results = []
    for index in indices:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        data = assemble_objects(crf, models, data)

        if plot:
            plot_image.plot_surfaces(data["final_surfaces"])

        results.append(data["final_surfaces"])
    return results

def get_full_prediction_model(clf_type="LR", use_boxes=False, do_post_processing=False):
    surface_model = find_surface_model()
    post_processing_model = post_processing.get_postprocessing_model(do_post_processing)
    assemble_surface_models = crf_tools.get_GPU_models()
    pw_clf = get_pairwise_clf(clf_type)

    crf = mean_field_CRF(pw_clf, use_boxes=use_boxes)
    try:
        crf.load_weights(f"parameters/CRF_trained_with_clf/{clf_type}.ckpt")
    except:
        print("No parameters found. Please train the model first.")

    return lambda x, y: post_processing_model(assemble_objects(crf, assemble_surface_models, load_images.load_image_and_surface_information(y)))

def main():
    assemble_objects_for_indices([7, 88, 102], clf_type="Neural")

if __name__ == '__main__':
    train_CRF("LR", use_boxes=True)