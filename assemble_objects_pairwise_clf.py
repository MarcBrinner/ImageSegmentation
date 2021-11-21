import random

import CRF_tools as crf_tools
import plot_image
import post_processing
import standard_values
import tensorflow as tf
import pickle
import process_surfaces as ps
import numpy as np
from tensorflow.keras import layers, Model, losses, optimizers, regularizers, initializers
from plot_image import plot_surfaces
from find_planes import find_surface_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from load_images import load_image_and_surface_information
from process_surfaces import find_optimal_surface_joins_from_annotation

clf_types = ["LR", "Tree", "Forest", "Neural", "Ensemble"]
model_types = ["Pairs", "Pairs CRF", "Pairs CRF Boxes"]

def check_clf_type_and_model_type(model_type, clf_type):
    if (model_type not in model_types and model_type is not None) or (clf_type not in clf_types and clf_type is not None):
        print("Unknown model type or clf type.")
        quit()

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(pairwise_potentials, axis=-1), Q_mult), matrix_1), axis=2)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-1)
    compatibility = compatibility - tf.reduce_min(compatibility, axis=-1, keepdims=True)
    compatibility = tf.math.divide_no_nan(compatibility, tf.reduce_max(compatibility)) * 15
    new_Q = tf.exp(tf.multiply(unary_potentials, 0.5) - compatibility)
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_CRF(num_iterations=60, use_boxes=True):
    pairwise_potentials_in = layers.Input(shape=(None, None), name="potentials")
    boxes_in = layers.Input(shape=(None, None), name="boxes")

    num_surfaces = tf.shape(boxes_in)[2]
    num_boxes = tf.shape(boxes_in)[1]
    num_labels = num_surfaces + num_boxes

    eye_mat = tf.eye(num_surfaces)
    initial_Q = pairwise_potentials_in * (1-eye_mat) + eye_mat
    initial_Q = tf.math.divide_no_nan(initial_Q, tf.reduce_sum(initial_Q, axis=-1, keepdims=True))
    initial_Q = tf.pad(initial_Q, [[0, 0], [0, num_boxes], [0, num_boxes]])

    boxes_dense = tf.squeeze(layers.Dense(1, kernel_initializer=initializers.constant(value=1.0), bias_initializer=initializers.constant(value=-0.3))(tf.expand_dims(boxes_in, axis=-1)), axis=-1)
    boxes_pad = tf.pad(boxes_dense, [[0, 0], [num_surfaces, 0], [0, num_boxes]])
    boxes_pad = boxes_pad + tf.transpose(boxes_pad, perm=[0, 2, 1])

    pairwise_potentials = tf.math.log(tf.math.divide_no_nan(pairwise_potentials_in, 1-pairwise_potentials_in))
    pairwise_potentials = tf.pad(pairwise_potentials, [[0, 0], [0, num_boxes], [0, num_boxes]])

    if use_boxes:
        pairwise_potentials = pairwise_potentials + boxes_pad
    else:
        pairwise_potentials = pairwise_potentials + 0 * boxes_pad

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)

    Q = initial_Q
    Q_acc = tf.stack([pairwise_potentials, Q], axis=1)
    for _ in range(num_iterations):
        Q = 0.9 * Q + 0.1 * mean_field_iteration(initial_Q, pairwise_potentials, Q, num_labels, matrix_1, matrix_2)
        Q_acc = tf.concat([Q_acc, tf.expand_dims(Q, axis=1)], axis=1)

    model = Model(inputs=[pairwise_potentials_in, boxes_in], outputs=Q_acc)
    model.compile(loss=crf_tools.custom_loss_CRF, optimizer=optimizers.Adam(learning_rate=optimizers.schedules.ExponentialDecay(3e-3, decay_steps=50, decay_rate=0.95)), metrics=[], run_eagerly=False)
    return model

def create_pairwise_clf_training_set():
    models = crf_tools.get_GPU_models()
    inputs = []
    labels = []
    for set_type in ["train", "test"]:
        for index in (standard_values.train_indices if set_type == "train" else standard_values.test_indices):
            print(index)
            data = load_image_and_surface_information(index)
            ps.calculate_pairwise_similarity_features_for_surfaces(data, models)

            Y = find_optimal_surface_joins_from_annotation(data)[0]
            join_matrix = Y[0][:-data["num_bboxes"], :-data["num_bboxes"]]
            not_join_matrix = Y[1][:-data["num_bboxes"], :-data["num_bboxes"]]

            feature_matrix = ps.create_similarity_feature_matrix(data)
            index_matrix = (join_matrix + not_join_matrix) == 1
            num_indices = np.sum(index_matrix)

            labels = labels + list(np.ndarray.flatten(join_matrix[index_matrix]))
            inputs = inputs + list(np.reshape(feature_matrix[index_matrix], (num_indices, np.shape(feature_matrix)[-1])))

        np.save(f"data/assemble_objects_with_clf_datasets/{set_type}_inputs.npy", np.asarray(inputs))
        np.save(f"data/assemble_objects_with_clf_datasets/{set_type}_labels.npy", np.asarray(labels))
        inputs = []
        labels = []

def get_nn_clf(num_features=standard_values.num_pairwise_features):
    input = layers.Input(shape=(num_features,))
    d_1 = layers.Dropout(0.1)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.0001), activation="relu", name="dense_1")(input))
    d_2 = layers.Dropout(0.1)(layers.Dense(50, kernel_regularizer=regularizers.l2(0.0001), activation="relu", name="dense_2")(d_1))
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0001), name="dense_3")(d_2)
    model = Model(inputs=input, outputs=tf.squeeze(out, axis=-1))
    model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam(learning_rate=5e-4))
    return model

def train_pairwise_classifier(type="LR"):
    if type not in clf_types:
        print("Invalid model type.")
        quit()
    inputs_train = np.load("data/assemble_objects_with_clf_datasets/train_inputs.npy")
    labels_train = np.load("data/assemble_objects_with_clf_datasets/train_labels.npy")
    inputs_test = np.load("data/assemble_objects_with_clf_datasets/test_inputs.npy")
    labels_test = np.load("data/assemble_objects_with_clf_datasets/test_labels.npy")

    kwargs = {}
    if type == "LR":
        clf = LogisticRegression(max_iter=1000, penalty="l2", C=0.5)
    elif type == "Tree":
        clf = DecisionTreeClassifier(min_samples_leaf=30)
    elif type == "Forest":
        clf = RandomForestClassifier(max_depth=13, n_estimators=70)
    elif type == "Neural":
        clf = get_nn_clf()
        kwargs["epochs"] = 20
    clf.fit(inputs_train, labels_train, **kwargs)
    if type == "Neural":
        clf.save_weights("parameters/pairwise_surface_clf/clf_Neural.ckpt")
    else:
        pickle.dump(clf, open(f"parameters/pairwise_surface_clf/clf_{type}.pkl", "wb"))
    print(np.sum(np.abs(np.round(clf.predict(inputs_train))-labels_train))/len(labels_train))
    print(np.sum(np.abs(np.round(clf.predict(inputs_test))-labels_test))/len(labels_test))

def evaluate_clf(model_type="Ensemble"):
    if model_type not in clf_types:
        print("Invalid model type.")
        quit()
    clf = load_pw_clf(model_type)
    inputs_train = np.load("data/assemble_objects_with_clf_datasets/train_inputs.npy")
    labels_train = np.load("data/assemble_objects_with_clf_datasets/train_labels.npy")
    inputs_test = np.load("data/assemble_objects_with_clf_datasets/test_inputs.npy")
    labels_test = np.load("data/assemble_objects_with_clf_datasets/test_labels.npy")
    print(np.sum(np.abs(np.round(clf.predict(inputs_train))-labels_train))/len(labels_train))
    print(np.sum(np.abs(np.round(clf.predict(inputs_test))-labels_test))/len(labels_test))

def load_pw_clf(type_str):
    if type_str == "Neural":
        clf_neural = get_nn_clf()
        clf_neural.load_weights("parameters/pairwise_surface_clf/clf_Neural.ckpt")
        clf = type('', (), {})()
        clf.predict = lambda x: np.round(clf_neural.predict(x))
        clf.predict_proba = lambda x: (lambda y: np.stack([1-y, y], axis=-1))(clf_neural.predict(x))
    elif type_str == "Ensemble":
        clfs = [load_pw_clf(t) for t in clf_types if t != "Ensemble"]
        clf = type('', (), {})()
        clf.predict_proba = lambda x: np.sum(np.asarray([clf.predict_proba(x) for clf in clfs]), axis=0) / len(clfs)
        clf.predict = lambda x: np.round(np.sum(np.asarray([clf.predict_proba(x)[:, 1] for clf in clfs]), axis=0) / len(clfs))
    else:
        clf = pickle.load(open(f"parameters/pairwise_surface_clf/clf_{type_str}.pkl", "rb"))
    return clf

def train_CRF_wrapper(clf_type="Forest"):
    pw_clf = load_pw_clf(clf_type)

    CRF = mean_field_CRF(use_boxes=True)
    crf_tools.train_CRF(CRF, f"parameters/CRF_trained_without_clf/{clf_type}.ckpt", pw_clf)
    print(CRF.trainable_weights)

def assemble_objects_with_pairwise_classifier(clf, models, data):
    ps.calculate_pairwise_similarity_features_for_surfaces(data, models)
    feature_matrix = ps.create_similarity_feature_matrix(data)

    predictions = clf.predict(np.reshape(feature_matrix, (-1, np.shape(feature_matrix)[-1])))
    join_matrix = np.zeros((data["num_surfaces"], data["num_surfaces"]))
    join_matrix[1:, 1:] = np.reshape(predictions, (data["num_surfaces"]-1, data["num_surfaces"]-1))

    data["final_surfaces"], _ = ps.join_surfaces_according_to_join_matrix(join_matrix, data["surfaces"], data["num_surfaces"])
    return data

def assemble_objects_crf(clf, models, crf, data):
    ps.calculate_pairwise_similarity_features_for_surfaces(data, models)
    feature_matrix = ps.create_similarity_feature_matrix(data)

    predictions = np.reshape(clf.predict_proba(np.reshape(feature_matrix, (-1, np.shape(feature_matrix)[-1])))[:, 1],
                             (data["num_surfaces"] - 1, data["num_surfaces"] - 1))
    predictions = predictions * 0.99 + 0.005
    crf_out = crf.predict([np.asarray([x]) for x in [predictions, data["bbox_overlap"][:, 1:]]])
    data["final_surfaces"] = crf_tools.create_surfaces_from_crf_output(crf_out[0, -1], data["surfaces"])
    return data

def assemble_objects_for_indices(indices, model_type="Pairs CRF Boxes", clf_type="Forest", plot=True):
    check_clf_type_and_model_type(model_type, clf_type)

    clf = load_pw_clf(clf_type)

    models = crf_tools.get_GPU_models()
    if "CRF" in model_type:
        crf = mean_field_CRF(num_iterations=60, use_boxes=("Boxes" in model_type))

    results = []
    for index in indices:
        data = load_image_and_surface_information(index)

        if "CRF" in model_type:
            data = assemble_objects_crf(clf, models, crf, data)
        else:
            data = assemble_objects_with_pairwise_classifier(clf, models, data)

        if plot:
            plot_surfaces(data["final_surfaces"])
        results.append(data["final_surfaces"])
    return results

def get_full_prediction_model(model_type="Pairs CRF Boxes", clf_type="Forest"):
    check_clf_type_and_model_type(model_type, clf_type)

    surface_model = find_surface_model()
    post_processing_model = post_processing.get_postprocessing_model()
    assemble_surface_models = crf_tools.get_GPU_models()
    clf = load_pw_clf(clf_type)

    if "CRF" in model_type:
        CRF = mean_field_CRF(num_iterations=60, use_boxes=("Boxes" in model_type))
        if "Boxes" in model_type:
            try:
                pass
                #CRF.load_weights(f"parameters/CRF_trained_without_clf/{clf_type}.ckpt")
            except:
                print("No trained parameters found. Using default parameters instead.")
        return lambda x, y: post_processing_model(assemble_objects_crf(clf, assemble_surface_models, CRF, load_image_and_surface_information(y)))
    else:
        return lambda x, y: post_processing_model(assemble_objects_with_pairwise_classifier(clf, assemble_surface_models, load_image_and_surface_information(y)))

def main():
    assemble_objects_for_indices([3, 76, 77], model_type="Pairs CRF")

if __name__ == '__main__':
    #assemble_objects_for_indices(standard_values.test_indices, model_type="Pairs", clf_type="Forest", plot=True)
    train_pairwise_classifier("Forest")
    quit()
    #create_pairwise_clf_training_set()
    #quit()
    assemble_objects_for_indices(standard_values.test_indices, "Pairs CRF", clf_type="Forest")