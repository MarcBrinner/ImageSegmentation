import numpy as np
import random
import pickle
import tensorflow as tf
import load_images
import process_surfaces as ps
from tensorflow.keras import layers, losses
from image_processing_models_GPU import chi_squared_distances_model_2D, extract_texture_function
from detect_objects import get_object_detector
from standard_values import *

def get_GPU_models():
    return chi_squared_distances_model_2D((10, 10), (4, 4)), \
           chi_squared_distances_model_2D((4, 4), (1, 1)), \
           extract_texture_function(), \
           get_object_detector()

def create_CRF_training_set():
    models = get_GPU_models()
    bbox_overlap_list = []
    features_list = []
    Y_list = []
    for index in train_indices:
        print(index)
        data = load_images.load_image_and_surface_information(index)
        ps.calculate_pairwise_similarity_features_for_surfaces(data, models)
        similarity_matrix = ps.create_similarity_feature_matrix(data)

        Y = ps.find_optimal_surface_joins_from_annotation(data)
        bbox_overlap = data["bbox_overlap"][:, 1:]

        bbox_overlap_list.append(bbox_overlap)
        Y_list.append(Y)
        features_list.append(similarity_matrix)

    pickle.dump(features_list, open("data/train_object_assemble_CRF_dataset/features.pkl", "wb"))
    pickle.dump(Y_list, open("data/train_object_assemble_CRF_dataset/Y.pkl", "wb"))
    pickle.dump(bbox_overlap_list, open("data/train_object_assemble_CRF_dataset/bbox.pkl", "wb"))

def create_surfaces_from_crf_output(prediction, surfaces):
    max_vals = np.argmax(prediction, axis=-1)
    new_surfaces = surfaces.copy()
    for i in range(len(max_vals)):
        if max_vals[i] != i:
            new_surfaces[new_surfaces == i+1] = max_vals[i]+1
    return new_surfaces

def train_CRF(CRF, save_path, pw_clf=None):
    features = pickle.load(open("data/train_object_assemble_CRF_dataset/features.pkl", "rb"))
    Y = pickle.load(open("data/train_object_assemble_CRF_dataset/Y.pkl", "rb"))
    boxes = pickle.load(open("data/train_object_assemble_CRF_dataset/bbox.pkl", "rb"))

    def gen():
        indices = list(range(len(features)))
        random.shuffle(indices)
        for i in indices:
            if pw_clf is None:
                yield {"features": features[i], "boxes": boxes[i]}, tf.squeeze(Y[i], axis=0)
            else:
                num_surfaces = np.shape(features[i])[0]
                predictions = np.reshape(pw_clf.predict_proba(np.reshape(features[i], (-1, np.shape(features[i])[-1])))[:, 1],
                           (num_surfaces, num_surfaces))
                yield {"potentials": predictions, "boxes": boxes[i]}, tf.squeeze(Y[i], axis=0)

    if pw_clf is None:
        dataset = tf.data.Dataset.from_generator(gen, output_types=({"features": tf.float32, "boxes": tf.float32}, tf.float32))
    else:
        dataset = tf.data.Dataset.from_generator(gen, output_types=({"potentials": tf.float32, "boxes": tf.float32}, tf.float32))

    CRF.fit(dataset.batch(1), verbose=1, epochs=60)
    print(CRF.trainable_weights)
    CRF.save_weights(save_path)

def custom_loss_CRF(y_actual, y_predicted):
    pred_pot = y_predicted[:, 0, :, :]
    pred_Q = y_predicted[:, 25, :, :]
    join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1))
    not_join_matrix = layers.Dropout(0.2)(tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1))

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

    Q_loss = positive_error + negative_error + LR_loss * 5
    return Q_loss