import gc
import pickle

import keras.losses
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers, layers, optimizers, initializers

import plot_image
from image_processing_models_GPU import normals_model_GPU, get_angle_arrays, Counts,\
    gaussian_filter_with_depth_factor_model_GPU, Variable, Variable2, Components
from load_images import *
from config import *
from plot_image import *
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

similarity_normalizers = [6, 2, 1/50, 1/40]
gauss_CRF_parameters = list(np.reshape([0.8, 0.8, 1.5, 1 / 80, 10000.0, 1 / 100, 10000.0, 1 / 200, 1 / 100, 10000.0, 20.0, 0.01], (12, 1, 1)))
LR_CRF_parameters = [0.01, 15000]
standard_kernel_size = 7
div_x, div_y = 4, 4

def get_GPU_models(mode="LR", kernel_size=standard_kernel_size):
    size_x, size_y = int(width / div_x), int(height / div_y)
    return gaussian_filter_with_depth_factor_model_GPU(), \
           find_surfaces_model_GPU(), \
           normals_model_GPU(), \
           conv_crf_LR(*load_pixel_similarity_parameters(), *LR_CRF_parameters, kernel_size, size_y, size_x) if mode == "LR" else \
           conv_crf_Gauss(*load_Gauss_parameters(), kernel_size, size_y, size_x)

@njit()
def extract_training_samples(diffs, annotation_windows, annotation_centrals, window, w):
    inputs_0 = [np.asarray([0.0, 0.0, 0.0, 0.0])]
    inputs_1 = [np.asarray([0.0, 0.0, 0.0, 0.0])]
    for i_1 in range(np.shape(diffs)[0]):
        for i_2 in range(np.shape(diffs)[1]):
            if annotation_centrals[i_1, i_2] == -1:
                continue
            for i_3 in range(window + 1):
                for i_4 in range(w if i_3 < window else window):
                    if annotation_windows[i_1, i_2, i_3, i_4] == -1:
                        continue
                    r = np.random.rand()
                    if annotation_windows[i_1, i_2, i_3, i_4] == annotation_centrals[i_1, i_2]:
                        if r < 0.006 and annotation_centrals[i_1, i_2] > 0:
                            inputs_1.append(diffs[i_1, i_2, i_3, i_4])
                    elif r < 0.06:
                        inputs_0.append(diffs[i_1, i_2, i_3, i_4])
    del inputs_0[0]
    del inputs_1[0]
    return inputs_0, inputs_1

# Create the training set for the training of the pixel similarity function.
def create_training_dataset(window=7):
    w = 2*window + 1
    pos_diffs = np.sqrt(np.sum(np.square(np.stack(np.meshgrid(np.arange(-window, window+1), np.arange(-window, window+1)), axis=-1)), axis=-1)) * similarity_normalizers[3]

    sliding_window = lambda x: np.swapaxes(sliding_window_view(x, (w, w), (0, 1)), -1, 2)
    central_element = lambda x: np.expand_dims(np.expand_dims(x[:, :, window, window], axis=2), axis=2)
    differences = lambda x: np.sqrt(np.sum(np.square(x - central_element(x)), axis=-1))

    inputs_0 = []
    inputs_1 = []
    save_index = 0
    for count_index in range(len(train_indices)):
        index = train_indices[count_index]
        data = load_image_and_surface_information(index)
        rgb_diffs = differences(sliding_window(data["rgb"])) * similarity_normalizers[2]
        angle_diffs = differences(sliding_window(data["angles"])) * similarity_normalizers[1]

        depth_windows = sliding_window(data["depth"])
        depth_central = central_element(depth_windows)
        depth_diffs = np.abs((depth_windows - depth_central) / ((depth_central + depth_windows)/2)) * similarity_normalizers[0]
        depth_diffs[data["depth"][window:-window, window:-window] == 0] = 0

        annotation = data["annotation"].copy().astype("int32")
        annotation[data["depth"] == 0] = -1
        annotation_windows = sliding_window(annotation)

        diffs = np.stack([depth_diffs, angle_diffs, rgb_diffs, np.broadcast_to(pos_diffs, np.shape(angle_diffs))], axis=-1)
        in_0, in_1 = extract_training_samples(diffs, annotation_windows, annotation[window:-window, window:-window], window, w)
        inputs_0.extend(in_0)
        inputs_1.extend(in_1)
        print(len(inputs_0), len(inputs_1), count_index % 10)
        if (count_index > 0 and count_index%10 == 0) or index == train_indices[-1]:
            np.save(f"data/assign_pixels_dataset/inputs_0_{save_index}.npy", np.asarray(inputs_0))
            np.save(f"data/assign_pixels_dataset/inputs_1_{save_index}.npy", np.asarray(inputs_1))
            save_index += 1
            inputs_0 = []
            inputs_1 = []
            print("SAVED.")
        gc.collect()
    join_datasets()

# Helper function in the process of creating the training set for the similarity classifiers.
def join_datasets():
    load = lambda x, y: np.load(f"data/assign_pixels_dataset/inputs_{x}_{y}.npy")
    save = lambda x, y: np.save(f"data/assign_pixels_dataset/inputs_{x}.npy", y)
    delete = lambda x, y: os.remove(f"data/assign_pixels_dataset/inputs_{x}_{y}.npy")

    for i in range(2):
        array = load(i, 0)
        delete(i, 0)
        for j in range(1, 10):
            array = np.concatenate([array, load(i, j)], axis=0)
            delete(i, j)
        save(i, array)

# Train the logistic regression pixel similarity classifier.
def train_LR_clf():
    inputs_0 = np.load("data/assign_pixels_dataset/inputs_0.npy")
    inputs_1 = np.load("data/assign_pixels_dataset/inputs_1.npy")

    train_0 = inputs_0[:int(len(inputs_0)*0.9)]
    train_1 = inputs_1[:int(len(inputs_1)*0.9)]
    test_0 = inputs_0[int(len(inputs_0) * 0.9):]
    test_1 = inputs_1[int(len(inputs_1) * 0.9):]
    train_X = np.concatenate([train_0, train_1], axis=0)
    test_X = np.concatenate([test_0, test_1], axis=0)
    train_Y = np.concatenate([np.zeros(int(len(inputs_0)*0.9)), np.ones(int(len(inputs_1)*0.9))], axis=0)
    test_Y = np.concatenate([np.zeros(len(inputs_0) - int(len(inputs_0)*0.9)), np.ones(len(inputs_1) - int(len(inputs_1)*0.9))], axis=0)

    poly = PolynomialFeatures(degree=1, interaction_only=False, include_bias=False)
    train_X_p = poly.fit_transform(train_X)
    test_X_p = poly.fit_transform(test_X)

    clf = LogisticRegression(max_iter=1000, penalty="l2")
    clf.fit(train_X_p, train_Y)
    print(clf.coef_)
    print(clf.intercept_)
    print(np.sum(np.abs(clf.predict_proba(train_X_p)[:, 1]-train_Y))/len(train_Y))
    print(np.sum(np.abs(clf.predict_proba(test_X_p)[:, 1]-test_Y))/len(test_Y))
    clf.fit(np.concatenate([train_X_p, test_X_p], axis=0), np.concatenate([train_Y, test_Y], axis=0))
    print(clf.coef_)
    print(clf.intercept_)
    pickle.dump(clf, open("parameters/pixel_similarity_clf/clf_2.pkl", "wb"))

# Create a Gaussian kernel-based pixel similarity function for the use as similarity function in the CRF.
def get_Gauss_model():
    input_in = layers.Input(shape=(4,))
    input = tf.square(input_in)

    one = tf.square(tf.expand_dims(input[:, 3], axis=-1))
    two = tf.square(tf.expand_dims(input[:, 0], axis=-1))
    three = tf.square(tf.expand_dims(input[:, 2], axis=-1))
    four = tf.square(tf.expand_dims(input[:, 1], axis=-1))

    one = tf.repeat(one, 3, axis=-1)
    two = tf.repeat(two, 3, axis=-1)

    vars = tf.concat([one, two, three, four], axis=-1)
    vars = layers.Multiply()([tf.square(Variable2(np.asarray([np.sqrt(x) for x in [100, 100, 100, 100, 100, 100, 100, 5]]), name="theta")(vars)), vars])

    exp_1 = tf.expand_dims(tf.exp(-(vars[:, 0] + vars[:, 3])), axis=-1)
    exp_2 = tf.expand_dims(tf.exp(-(vars[:, 1] + vars[:, 4] + vars[:, 6])), axis=-1)
    exp_3 = tf.expand_dims(tf.exp(-(vars[:, 2] + vars[:, 5] + vars[:, 7])), axis=-1)
    out = layers.Dense(1, activation="sigmoid", kernel_initializer=initializers.constant([1, 1, 1]),
                       bias_initializer=initializers.constant(-0.8), kernel_regularizer=regularizers.l2(0.03), name="weights")(
        tf.concat([exp_1, exp_2, exp_3], axis=-1))
    model = Model(inputs=input_in, outputs=tf.squeeze(out, axis=-1))
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=optimizers.Adam(learning_rate=8e-3), metrics=[],
                  run_eagerly=False)
    return model

# Create a logistic regression model for the use as similarity function in the CRF.
def get_LR_model():
    input = layers.Input(shape=(4,))

    vars = tf.reduce_sum(layers.Multiply()([tf.square(Variable2(np.asarray([1.0, 1.0, 1.0, 1.0]), name="theta")(input)) * tf.constant([-1.0, -1.0, -1.0, 1.0]), input]), axis=-1)
    out = tf.math.sigmoid(layers.Add()([Variable2([-1.0], name="bias")(vars), vars]))

    model = Model(inputs=input, outputs=out)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=optimizers.Adam(learning_rate=8e-3), metrics=[],
                  run_eagerly=False)
    model.summary()
    return model

# Train the similarity function based on the Gaussian kernels.
def train_Gauss_clf():
    try:
        inputs_0 = np.load("data/assign_pixels_dataset/inputs_0.npy")
        inputs_1 = np.load("data/assign_pixels_dataset/inputs_1.npy")
    except:
        print("No data set found, creating new one...")
        create_training_dataset()
        try:
            inputs_0 = np.load("data/assign_pixels_dataset/inputs_0.npy")
            inputs_1 = np.load("data/assign_pixels_dataset/inputs_1.npy")
        except:
            print("Failed to load data.")
            quit()

    train_0 = inputs_0[:int(len(inputs_0)*0.9)]
    train_1 = inputs_1[:int(len(inputs_1)*0.9)]
    test_0 = inputs_0[int(len(inputs_0) * 0.9):]
    test_1 = inputs_1[int(len(inputs_1) * 0.9):]
    train_X = np.concatenate([train_0, train_1], axis=0)
    test_X = np.concatenate([test_0, test_1], axis=0)
    train_Y = np.concatenate([np.zeros(int(len(inputs_0)*0.9)), np.ones(int(len(inputs_1)*0.9))], axis=0)
    test_Y = np.concatenate([np.zeros(len(inputs_0) - int(len(inputs_0)*0.9)), np.ones(len(inputs_1) - int(len(inputs_1)*0.9))], axis=0)

    complete_data = np.concatenate([train_X, test_X], axis=0), np.concatenate([train_Y, test_Y], axis=0)

    clf = get_Gauss_model()
    clf.fit(*complete_data, shuffle=True, epochs=2, batch_size=128)

    parameters = [*[p[0] for p in list(clf.get_layer("weights").weights[0].numpy())], clf.get_layer("weights").weights[1].numpy()[0]]
    parameters += list(clf.get_layer("theta").weights[0].numpy())
    print(parameters)
    pickle.dump(parameters, open("parameters/pixel_similarity_clf/gauss.pkl", "wb"))

# Load the parameters of the similarity function based on the logistic regression model.
def load_pixel_similarity_parameters():
    clf = pickle.load(open("parameters/pixel_similarity_clf/clf.pkl", "rb"))
    return clf.coef_, clf.intercept_ + 4

# Load the parameters of the similarity function based on the Gaussian kernels.
def load_Gauss_parameters():
    p = pickle.load(open("parameters/pixel_similarity_clf/gauss.pkl", "rb"))
    p.append(0.01)
    #p[3] = p[3] + 3
    p[3] = 0
    return list(np.reshape(p, (13, 1, 1)))

# The GPU implementation of the model for detecting smooth surface patches in the depth image.
def find_surfaces_model_GPU(depth=4, threshold=0.007003343675404672 * 10.5, height=height, width=width, alpha=3, s1=2, s2=1, n1=11, n2=5, component_threshold=20):
    # Find curvature score edges
    depth_image_in = layers.Input(batch_shape=(1, height, width))
    depth_image = tf.expand_dims(tf.squeeze(depth_image_in, axis=0), axis=-1)

    angle_values_x, angle_values_y = get_angle_arrays()

    middle = depth_image[1:-1, 1:-1, :]
    distance_factors_x = tf.abs(angle_values_x[1:-1, :-2, :] - angle_values_x[1:-1, 2:, :]) * middle * 0.5
    distance_factors_y = tf.abs(angle_values_y[:-2, 1:-1, :] - angle_values_y[2:, 1:-1, :]) * middle * 0.5
    distance_factors = tf.concat([distance_factors_x, distance_factors_y], axis=-1)[depth - 1: -(depth - 1), depth - 1: -(depth - 1),:]

    shape = (height - 2*depth, width - 2*depth)

    curvature_scores = tf.zeros(shape)
    central_depths = depth_image[depth:-depth, depth:-depth, :]
    central_points = tf.concat([tf.zeros((shape[0], shape[1], 2)), central_depths], axis=-1)

    for direction in [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]:
        current_scores = tf.zeros_like(curvature_scores)
        prev_distances = tf.zeros_like(curvature_scores)
        prev_points = central_points
        for k in range(1, depth+1):
            new_points = tf.concat([distance_factors*k, tf.slice(depth_image, [depth-k*direction[0], depth-k*direction[1], 0], (shape[0], shape[1], 1))], axis=-1)
            dif_1 = tf.sqrt(tf.reduce_sum(tf.square(new_points - prev_points), axis=-1))
            dif_2 = tf.sqrt(tf.reduce_sum(tf.square(new_points - central_points), axis=-1))
            score = tf.math.divide_no_nan((dif_1 + prev_distances), dif_2) - 1
            current_scores = current_scores + score
            curvature_scores = curvature_scores + current_scores
            prev_distances = dif_2
            prev_points = new_points

    pixels = tf.cast(tf.less(curvature_scores, threshold), tf.int32)
    pixels = tf.pad(tf.where(tf.greater(tf.squeeze(tf.gather(central_points, [2], axis=-1), axis=-1), 0.001), pixels, 0), [[depth, depth], [depth, depth]], constant_values=0)

    # Smooth edges
    pixels = tf.where(tf.greater(tf.reduce_sum(tf.image.extract_patches(tf.expand_dims(tf.expand_dims(pixels, 0), -1), [1, 2*s1+1, 2*s1+1, 1], padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), axis=-1, keepdims=True), n1), 1, 0)
    pixels = tf.where(tf.greater(tf.reduce_sum(tf.image.extract_patches(pixels, [1, 2*s2+1, 2*s2+1, 1], padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), axis=-1), n2), 1, 0)
    pixels = tf.squeeze(pixels, axis=0)

    # Find depth edges
    sub_1 = depth_image[:-2, 1:-1, :]
    sub_2 = depth_image[2:, 1:-1, :]
    sub_3 = depth_image[1:-1, 2:, :]
    sub_4 = depth_image[1:-1, :-2, :]

    diffs = tf.abs(tf.concat([sub_3, sub_4, sub_1, sub_2], axis=-1) - middle)

    distance_factors = layers.Concatenate(axis=-1)([distance_factors_x, distance_factors_x, distance_factors_y, distance_factors_y]) * alpha
    edges = tf.pad(tf.reduce_min(tf.where(tf.greater(diffs, distance_factors), 0, 1), axis=-1), [[1, 1], [1, 1]], constant_values=0)
    pixels = tf.multiply(edges, pixels)

    components = Components()(pixels)
    counts = Counts()(components)

    pixels = pixels * tf.reshape(tf.gather(tf.where(tf.greater(counts, component_threshold), 1, 0), tf.reshape(components, (height*width,))), (height, width))
    pixels = Components()(pixels)

    model = Model(inputs=depth_image_in, outputs=[pixels, edges])
    return lambda x: model.predict(np.expand_dims(x, axis=0), batch_size=1)

# The Gaussian kernel-based CRF model for assigning additional pixels to the given surface patches.
def conv_crf_Gauss(w_1, w_2, w_3, bias, theta_1_1, theta_2_1, theta_3_1, theta_1_2, theta_2_2, theta_3_2, theta_2_3, theta_3_3, weight, kernel_size, height, width):
    k = kernel_size*2+1
    Q = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, None), dtype=tf.float32)
    number_of_surfaces = tf.shape(Q)[-1]
    unary_potentials = layers.Input(shape=(height, width, None), dtype=tf.float32)
    val = layers.Input(batch_shape=(1, 1, 1, 1, None))
    matrix = tf.tile(tf.reshape(tf.ones((number_of_surfaces, number_of_surfaces)) - tf.eye(number_of_surfaces, number_of_surfaces),
                               (1, 1, 1, number_of_surfaces, number_of_surfaces)), [1, height, width, 1, 1])

    mask_np = np.ones((1, 1, 1, k, k, 1))
    mask_np[0, 0, 0, kernel_size, kernel_size, 0] = 0
    mask = tf.constant(mask_np, dtype=tf.float32)

    features_1 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 3), dtype=tf.float32)
    features_2 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 6), dtype=tf.float32)
    features_3 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 5), dtype=tf.float32)

    mask_1 = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
    mask_2 = tf.constant([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=tf.float32)
    mask_3 = tf.constant([0.0, 0.0, 1.0, 0.0, 0.0], dtype=tf.float32)

    windows_Q = tf.reshape(tf.image.extract_patches(Q, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, number_of_surfaces))
    windows_Q = tf.multiply(windows_Q, tf.broadcast_to(mask, tf.shape(windows_Q)))
    windows_f_1 = tf.reshape(tf.image.extract_patches(features_1, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 3))
    windows_f_2 = tf.reshape(tf.image.extract_patches(features_2, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 6))
    windows_f_3 = tf.reshape(tf.image.extract_patches(features_3, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 5))

    f_1 = tf.gather(tf.gather(windows_f_1, [kernel_size], axis=3), [kernel_size], axis=4)
    f_2 = tf.gather(tf.gather(windows_f_2, [kernel_size], axis=3), [kernel_size], axis=4)
    f_3 = tf.gather(tf.gather(windows_f_3, [kernel_size], axis=3), [kernel_size], axis=4)

    feature_1 = tf.tile(f_1, [1, 1, 1, k, k, 1])
    feature_2 = tf.tile(f_2, [1, 1, 1, k, k, 1])
    feature_3 = tf.tile(f_3, [1, 1, 1, k, k, 1])

    div_1 = tf.math.pow(f_1, mask_1)
    div_2 = tf.math.pow(f_2, mask_2)
    div_3 = tf.math.pow(f_3, mask_3)

    theta_1_1 = tf.repeat(Variable(theta_1_1, name="theta_1_1")(feature_1), repeats=[2], axis=-1)
    theta_1_2 = tf.repeat(Variable(theta_1_2, name="theta_1_2")(feature_1), repeats=[1], axis=-1)
    theta_2_1 = tf.repeat(Variable(theta_2_1, name="theta_2_1")(feature_1), repeats=[2], axis=-1)
    theta_2_2 = tf.repeat(Variable(theta_2_2, name="theta_2_2")(feature_1), repeats=[1], axis=-1)
    theta_2_3 = tf.repeat(Variable(theta_2_3, name="theta_2_3")(feature_1), repeats=[3], axis=-1)
    theta_3_1 = tf.repeat(Variable(theta_3_1, name="theta_3_1")(feature_1), repeats=[2], axis=-1)
    theta_3_2 = tf.repeat(Variable(theta_3_2, name="theta_3_2")(feature_1), repeats=[1], axis=-1)
    theta_3_3 = tf.repeat(Variable(theta_3_3, name="theta_3_3")(feature_1), repeats=[2], axis=-1)

    theta_1 = tf.square(tf.expand_dims(tf.concat([theta_1_1, theta_1_2], axis=-1, name="Theta_1"), axis=1))
    theta_2 = tf.square(tf.expand_dims(tf.concat([theta_2_1, theta_2_2, theta_2_3], axis=-1, name="Theta_2"), axis=1))
    theta_3 = tf.square(tf.expand_dims(tf.concat([theta_3_1, theta_3_2, theta_3_3], axis=-1, name="Theta_3"), axis=1))

    differences_1 = tf.math.divide_no_nan(tf.subtract(windows_f_1, feature_1), (div_1 + tf.math.pow(windows_f_1, mask_1))/2) * tf.constant([similarity_normalizers[3], similarity_normalizers[3], similarity_normalizers[0]])
    similarities_1 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_1), tf.broadcast_to(theta_1, tf.shape(differences_1))), axis=-1))

    differences_2 = tf.math.divide_no_nan(tf.subtract(windows_f_2, feature_2), (div_2 + tf.math.pow(windows_f_2, mask_2))/2) * tf.constant([similarity_normalizers[3], similarity_normalizers[3], similarity_normalizers[0], similarity_normalizers[2], similarity_normalizers[2], similarity_normalizers[2]])
    similarities_2 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_2), tf.broadcast_to(theta_2, tf.shape(differences_2))), axis=-1))

    differences_3 = tf.math.divide_no_nan(tf.subtract(windows_f_3, feature_3), (div_3 + tf.math.pow(windows_f_3, mask_3))/2) * tf.constant([similarity_normalizers[3], similarity_normalizers[3], similarity_normalizers[0], similarity_normalizers[1], similarity_normalizers[1]])
    similarities_3 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_3), tf.broadcast_to(theta_3, tf.shape(differences_3))), axis=-1))

    similarity_sum = layers.Add()([layers.Multiply()([Variable2(w_1, name="w_1")(similarities_1), similarities_1]),
                                   layers.Multiply()([Variable2(w_2, name="w_2")(similarities_2), similarities_2]),
                                   layers.Multiply()([Variable2(w_3, name="w_3")(similarities_3), similarities_3]),
                                   Variable2(bias, name="bias")(similarities_1)])
    similarity_sum = tf.concat([tf.reshape(similarity_sum, (1, height, width, k*k)), tf.ones((1, height, width, 1))*0.00005], axis=-1)
    windows_Q = tf.concat([tf.reshape(windows_Q, (1, height, width, k*k, number_of_surfaces)), tf.tile(val, [1, height, width, 1, 1])], axis=-2)
    messages = tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(windows_Q)), windows_Q), axis=-2)

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, 1, number_of_surfaces, 1])), axis=-1)
    add = layers.Add(activity_regularizer=regularizers.l2(0.0001))([unary_potentials, layers.Multiply()([Variable2(weight, name="weight")(compatibility_values), compatibility_values])])

    output = layers.Softmax()(-add)

    model = Model(inputs=[unary_potentials, Q, features_1, features_2, features_3, val], outputs=output)
    model.compile(loss=None, optimizer=optimizers.Adam(learning_rate=1e-3), metrics=[], run_eagerly=False)
    return model

# A CRF model for detecting complete surfaces given the smaller surface patches. This model was not present in the thesis because
# the Gaussian kernel model performed better than this one that just uses linear weights on the differences.
def conv_crf_LR(LR_weights, LR_bias, weight, auxiliary_weight, kernel_size, height, width):
    k = kernel_size*2+1
    Q = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, None), dtype=tf.float32)
    number_of_surfaces = tf.shape(Q)[-1]
    unary_potentials = layers.Input(shape=(height, width, None), dtype=tf.float32)
    val = layers.Input(batch_shape=(1, 1, 1, 1, None))
    matrix = tf.tile(tf.reshape(tf.ones((number_of_surfaces, number_of_surfaces)) - tf.eye(number_of_surfaces, number_of_surfaces),
                               (1, 1, 1, number_of_surfaces, number_of_surfaces)), [1, height, width, 1, 1])

    mask_np = np.ones((1, 1, 1, k, k, 1))
    mask_np[0, 0, 0, kernel_size, kernel_size, 0] = 0
    mask = tf.constant(mask_np, dtype=tf.float32)

    features_1 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 1), dtype=tf.float32)
    features_2 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 2), dtype=tf.float32)
    features_3 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 3), dtype=tf.float32)
    features_4 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 2), dtype=tf.float32)

    windows_Q = tf.reshape(tf.image.extract_patches(Q, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, number_of_surfaces))
    windows_Q = tf.multiply(windows_Q, tf.broadcast_to(mask, tf.shape(windows_Q)))
    windows_f_1 = tf.reshape(tf.image.extract_patches(features_1, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 1))
    windows_f_2 = tf.reshape(tf.image.extract_patches(features_2, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 2))
    windows_f_3 = tf.reshape(tf.image.extract_patches(features_3, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 3))
    windows_f_4 = tf.reshape(tf.image.extract_patches(features_4, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 2))

    f_1 = tf.gather(tf.gather(windows_f_1, [kernel_size], axis=3), [kernel_size], axis=4)
    f_2 = tf.gather(tf.gather(windows_f_2, [kernel_size], axis=3), [kernel_size], axis=4)
    f_3 = tf.gather(tf.gather(windows_f_3, [kernel_size], axis=3), [kernel_size], axis=4)
    f_4 = tf.gather(tf.gather(windows_f_4, [kernel_size], axis=3), [kernel_size], axis=4)

    feature_1 = tf.tile(f_1, [1, 1, 1, k, k, 1])
    feature_2 = tf.tile(f_2, [1, 1, 1, k, k, 1])
    feature_3 = tf.tile(f_3, [1, 1, 1, k, k, 1])
    feature_4 = tf.tile(f_4, [1, 1, 1, k, k, 1])

    differences_1 = tf.math.divide_no_nan(tf.subtract(windows_f_1, feature_1), feature_1)
    similarities_1 = tf.sqrt(tf.reduce_sum(tf.square(differences_1), axis=-1)) * similarity_normalizers[0]

    differences_2 = tf.subtract(windows_f_2, feature_2)
    similarities_2 = tf.sqrt(tf.reduce_sum(tf.square(differences_2), axis=-1)) * similarity_normalizers[1]

    differences_3 = tf.subtract(windows_f_3, feature_3)
    similarities_3 = tf.sqrt(tf.reduce_sum(tf.square(differences_3), axis=-1)) * similarity_normalizers[2]

    differences_4 = tf.subtract(windows_f_4, feature_4)
    similarities_4 = tf.sqrt(tf.reduce_sum(tf.square(differences_4), axis=-1)) * similarity_normalizers[3]

    similarities = tf.stack([similarities_1, similarities_2, similarities_3, similarities_4], axis=-1)
    similarities = tf.reshape(similarities, (1, height, width, k*k, 4))

    LR_mult_out = tf.reduce_sum(layers.Multiply()([Variable2(LR_weights, name="LR_weights")(similarities), similarities]), axis=-1)
    LR_out = layers.Add()([Variable2(LR_bias, name="LR_bias")(LR_mult_out), LR_mult_out])

    concat_out = tf.concat([LR_out, tf.ones((1, height, width, 1))*auxiliary_weight], axis=-1)
    windows_Q = tf.concat([tf.reshape(windows_Q, (1, height, width, k*k, number_of_surfaces)), tf.tile(val, [1, height, width, 1, 1])], axis=-2)
    messages = tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(concat_out, axis=-1), tf.shape(windows_Q)), windows_Q), axis=-2)

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, 1, number_of_surfaces, 1])), axis=-1)
    add = layers.Add()([unary_potentials, layers.Multiply()([Variable2(weight, name="weight")(compatibility_values), compatibility_values])])

    output = layers.Softmax()(-add)

    model = Model(inputs=[unary_potentials, Q, features_1, features_2, features_3, features_4, val], outputs=output)
    return model

# Create the inputs for the mean field update model.
def get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y):
    features_pad = [np.pad(f, [[kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]]) for f in features]
    Q = np.pad(Q, [[kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]])
    features_out = [list() for _ in features_pad]
    Q_in = []
    unary = []

    for x in range(div_x):
        for y in range(div_y):
            for i in range(len(features_pad)):
                features_out[i].append(features_pad[i][y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            Q_in.append(Q[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            unary.append(unary_potentials[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x])

    val = np.zeros((len(unary), 1, 1, 1, np.shape(Q)[-1]))
    val[:, 0, 0, 0, -1] = 1

    return [np.asarray(x) for x in [unary, Q_in, *features_out, val]]

def get_unary_potentials_and_initial_probabilities(data):
    patches, num_surfaces = data["patches"], data["num_surfaces"]
    height, width = np.shape(patches)
    potential = np.ones((1, 1, num_surfaces+1))
    potential[0,0,0] = 10
    unary_potentials = np.tile(potential, [height, width, 1])
    prob = np.ones((1, 1, num_surfaces+1)) / (num_surfaces)
    prob[0,0,0] = 0
    prob_copy = prob.copy()
    initial_probabilities = np.tile(prob, [height, width, 1])
    for y in range(height):
        for x in range(width):
            if patches[y][x] != 0:
                potential = np.ones(num_surfaces+1)
                potential[patches[y][x]] = 0
                potential[0] = 10
                unary_potentials[y][x] = potential
                prob = np.ones(num_surfaces+1)/(num_surfaces-1)*0.1
                prob[patches[y][x]] = 0.9
                prob[0] = 0
                initial_probabilities[y][x] = prob
    return unary_potentials, initial_probabilities, prob_copy

def extract_features(depth_image, lab_image, angles, grid):
    features_1_new = np.concatenate([grid, np.expand_dims(depth_image, axis=-1)], axis=-1)
    features_2_new = np.concatenate([features_1_new, lab_image], axis=-1)
    features_3_new = np.concatenate([features_1_new, angles], axis=-1)

    return (features_1_new, features_2_new, features_3_new)

# Create a single output from the batch-output of the mean field model.
def assemble_outputs(outputs, div_x, div_y, size_x, size_y, height, width):
    Q = np.zeros((height, width, *np.shape(outputs)[3:]))

    index = 0
    for x in range(div_x):
        for y in range(div_y):
            Q[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x, :] = outputs[index]
            index += 1
    return Q

def find_surfaces(models, image_data, mode, kernel_size, plot_result=False, save_index=-1):
    size_x, size_y = int(width / div_x), int(height / div_y)
    smoothing_model, surface_model, normals_and_log_depth, conv_crf_model = models
    grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = np.stack([grid[1], grid[0]], axis=-1)

    data = {}
    data["depth"], data["rgb"], data["annotation"] = image_data
    smoothed_depth = smoothing_model(data["depth"], grid)

    data["angles"], data["vectors"], data["points_3d"] = normals_and_log_depth(smoothed_depth)
    if mode == "LR":
        features = [np.expand_dims(data["depth"], axis=-1), data["angles"], data["rgb"], grid]
    else:
        features = extract_features(data["depth"], data["rgb"], data["angles"], grid)

    data["patches"], data["depth_edges"] = surface_model(smoothed_depth)
    data["num_surfaces"] = int(np.max(data["patches"]) + 1)
    unary_potentials, initial_Q, prob = get_unary_potentials_and_initial_probabilities(data)

    inputs = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y)
    out = conv_crf_model.predict(inputs, batch_size=1)
    Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width)
    Q[data["depth"] == 0] = prob

    inputs = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
    out = conv_crf_model.predict(inputs, batch_size=1)
    Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width)
    Q[data["depth"] == 0] = prob

    inputs = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
    out = conv_crf_model.predict(inputs, batch_size=1)
    Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width)
    Q[data["depth"] == 0] = 0
    data["surfaces"] = np.argmax(Q, axis=-1)
    data["surfaces"][data["surfaces"] == data["num_surfaces"]] = 0

    if plot_result:
        plot_surfaces(data["surfaces"])

    if save_index >= 0:
        os.makedirs(f"out/{save_index}", exist_ok=True)
        np.save(f"out/{save_index}/Q.npy", data["surfaces"])
        np.save(f"out/{save_index}/depth.npy", data["depth"])
        np.save(f"out/{save_index}/angles.npy", data["angles"])
        np.save(f"out/{save_index}/vectors.npy", data["vectors"])
        np.save(f"out/{save_index}/patches.npy", data["patches"])
        np.save(f"out/{save_index}/points.npy", data["points_3d"])
        np.save(f"out/{save_index}/edges.npy", data["depth_edges"])
    return data

# This method allows detecting the surfaces for the images from the train/test set that correspond to the set of given indices.
# By using "save_data=True", the output is saved and can be used in other methods to create training sets for the training of classifiers, or
# to create the segmentations for complete sets of indices.
def find_surfaces_for_indices(image_indices, mode="Gauss", kernel_size=standard_kernel_size, save_data=True, plot_result=False, return_results=False):
    models = get_GPU_models(mode, kernel_size)
    results = []
    for index in image_indices:
        print(index)
        image_data = load_image(index)
        data = find_surfaces(models, image_data, mode, kernel_size, plot_result, -1 if not save_data else index)
        if return_results:
            results.append(data)
    if return_results:
        return results

def find_surface_model(mode="Gauss", kernel_size=standard_kernel_size):
    models = get_GPU_models(mode, kernel_size)
    return lambda x: find_surfaces(models, x, mode, kernel_size, False, -1)

if __name__ == '__main__':
    find_surfaces_for_indices(list(range(0, 111)), mode="Gauss", plot_result=False, save_data=True)
    quit()
