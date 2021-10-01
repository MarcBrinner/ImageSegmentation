import gc

import numpy as np
import tornado.httputil

import plot_image
from image_processing_models_GPU import normals_and_log_depth_model_GPU, get_angle_arrays, Counts,\
    gaussian_filter_with_depth_factor_model_GPU, tf, layers, Variable, Variable2, regularizers, optimizers, Model, Components
from load_images import *
from standard_values import *
from plot_image import *
import tensorflow.keras.backend as K

physical_devices = tf.config.list_physical_devices('GPU')

train_indices = [109, 108, 107, 105, 104, 103, 102, 101, 100]
test_indices = [110, 106]

variable_names = ["w_1", "w_2", "w_3", "theta_1_1", "theta_1_2", "theta_2_1", "theta_2_2", "theta_2_3", "theta_3_1", "theta_3_2", "theta_3_3", "weight"]

def print_parameters(model):
    for name in variable_names:
        try:
            print(f"{name}: {model.get_layer(name).weights}")
        except:
            print(f"{name} not available.")

def save_parameters(model, index):
    array = np.zeros(len(variable_names), dtype="float64")
    for i in range(len(variable_names)):
        array[i] = model.get_layer(variable_names[i]).weights[0].numpy()
    print(array)
    np.save(f"parameters/{index}.npy", array)

def load_parameters(index):
    if index < 0:
        return list(np.reshape([0.8, 0.8, 1.5, 1 / 80, 5000.0, 1 / 100, 5000.0, 1 / 200, 1 / 100, 5000.0, 20.0, 0.01], (12, 1, 1)))
    return list(np.reshape(np.load(f"parameters/{index}.npy"), (12, 1, 1)))

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

def conv_crf(w_1, w_2, w_3, theta_1_1, theta_1_2, theta_2_1, theta_2_2, theta_2_3, theta_3_1, theta_3_2, theta_3_3, weight, kernel_size, height, width, intermediate=False):
    k = kernel_size*2+1
    Q = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, None), dtype=tf.float32)
    number_of_surfaces = tf.shape(Q)[-1]
    unary_potentials = layers.Input(shape=(height, width, None), dtype=tf.float32)

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

    theta_1 = tf.expand_dims(tf.concat([theta_1_1, theta_1_2], axis=-1, name="Theta_1"), axis=1)
    theta_2 = tf.expand_dims(tf.concat([theta_2_1, theta_2_2, theta_2_3], axis=-1, name="Theta_2"), axis=1)
    theta_3 = tf.expand_dims(tf.concat([theta_3_1, theta_3_2, theta_3_3], axis=-1, name="Theta_3"), axis=1)

    differences_1 = tf.math.divide_no_nan(tf.subtract(windows_f_1, feature_1), div_1)
    similarities_1 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_1), tf.broadcast_to(theta_1, tf.shape(differences_1))), axis=-1))

    differences_2 = tf.math.divide_no_nan(tf.subtract(windows_f_2, feature_2), div_2)
    similarities_2 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_2), tf.broadcast_to(theta_2, tf.shape(differences_2))), axis=-1))

    differences_3 = tf.math.divide_no_nan(tf.subtract(windows_f_3, feature_3), div_3)
    similarities_3 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_3), tf.broadcast_to(theta_3, tf.shape(differences_3))), axis=-1))

    similarity_sum = layers.Add()([layers.Multiply()([Variable2(w_1, name="w_1")(similarities_1), similarities_1]),
                                   layers.Multiply()([Variable2(w_2, name="w_2")(similarities_2), similarities_2]),
                                   layers.Multiply()([Variable2(w_3, name="w_3")(similarities_3), similarities_3])])
    messages = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(windows_Q)), windows_Q), axis=-2), axis=-2)

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, 1, number_of_surfaces, 1])), axis=-1)
    add = layers.Add(activity_regularizer=regularizers.l2(0.0001))([unary_potentials, layers.Multiply()([Variable2(weight, name="weight")(compatibility_values), compatibility_values])])

    output = layers.Softmax()(-add)

    model = Model(inputs=[unary_potentials, Q, features_1, features_2, features_3], outputs=similarity_sum if intermediate else output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-3), metrics=[], run_eagerly=True)
    return model

def custom_loss(y_actual, y_predicted):
    labels = tf.squeeze(tf.gather(y_actual, [0], axis=3), axis=3)
    weights = tf.squeeze(tf.gather(y_actual, [1], axis=3), axis=3)
    error = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(y_predicted - labels) * weights, axis=-1), axis=-1), axis=-1)
    accumulated_weights = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(weights, axis=-1), axis=-1), axis=-1)
    error = tf.sqrt(tf.math.divide_no_nan(error, accumulated_weights)) * 10
    print(" ", error)
    return error

def find_annotation_correspondence_general(data):
    patches, annotation = data["patches"], data["annotation"]
    mapping = {i: set() for i in range(int(np.max(annotation)+1))}
    height, width = np.shape(patches)
    for y in range(height):
        for x in range(width):
            index = patches[y][x]
            if index == 0:
                continue
            mapping[annotation[y][x]].add(index)
    return mapping

def find_annotation_correspondence_patch_specific(data):
    patches, annotation, num_surfaces = data["patches"], data["annotation"], data["num_surfaces"]
    correspondence = {i: set() for i in range(num_surfaces)}
    for y in range(height):
        for x in range(width):
            if (patch := patches[y][x]) != 0:
                correspondence[patch].add(annotation[y][x])
    return correspondence

def get_training_targets(data, div_x, div_y, size_x, size_y):
    patches, annotation, num_surfaces, depth_image = data["patches"], data["annotation"], data["num_surfaces"], data["depth"]

    correspondence_general = find_annotation_correspondence_general(data)
    correspondence_patch_specific = find_annotation_correspondence_patch_specific(data)
    height, width = np.shape(depth_image)
    Y = np.zeros((height, width, 2, num_surfaces))
    for y in range(height):
        for x in range(width):
            if depth_image[y][x] < 0.001:
                continue
            a = annotation[y][x]
            l = patches[y][x]

            if l != 0:
                label = np.zeros(num_surfaces)
                weight = np.ones(num_surfaces)

                if len(correspondence_patch_specific[l]) < 2:
                    label[l] = 1
                else:
                    possible_labels = set()
                    for ann in correspondence_patch_specific[l]:
                        for p in correspondence_general[ann]:
                            possible_labels.add(p)
                    for p in possible_labels:
                        weight[p] = 0
            elif a != 0:
                weight = np.ones(num_surfaces)
                label = np.zeros(num_surfaces)
                for m in correspondence_general[a]:
                    weight[m] = 0
            else:
                label = np.zeros(num_surfaces)
                weight = np.zeros(num_surfaces)
            if l != 0:
                label = np.zeros(num_surfaces)
                weight = np.zeros(num_surfaces)

            Y[y][x][0] = label
            Y[y][x][1] = weight

    Y_batch = []
    indices = []
    index = 0
    for x in range(div_x):
        for y in range(div_y):
            new_sample = Y[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x]
            weights_sum = np.sum(new_sample[:, :, 1, :])
            if weights_sum == 0:
                continue
            else:
                print("HERE")
                Y_batch.append(new_sample)
                indices.append(index)
            index += 1
    return np.asarray(Y_batch), indices

def get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y, indices=None):
    features_1, features_2, features_3 = [np.pad(f, [[kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]]) for f in features]
    Q = np.pad(Q, [[kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]])
    f_1 = []
    f_2 = []
    f_3 = []
    Q_in = []
    unary = []

    index = -1
    for x in range(div_x):
        for y in range(div_y):
            index += 1
            if indices is not None and index not in indices:
                continue
            f_1.append(features_1[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            f_2.append(features_2[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            f_3.append(features_3[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            Q_in.append(Q[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            unary.append(unary_potentials[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x])
    return [np.asarray(x) for x in [unary, Q_in, f_1, f_2, f_3]]

def get_unary_potentials_and_initial_probabilities(data):
    patches, num_surfaces = data["patches"], data["num_surfaces"]
    height, width = np.shape(patches)
    potential = np.ones((1, 1, num_surfaces))
    potential[0,0,0] = 10
    unary_potentials = np.tile(potential, [height, width, 1])
    prob = np.ones((1, 1, num_surfaces)) / (num_surfaces - 1)
    prob[0,0,0] = 0
    prob_copy = prob.copy()
    initial_probabilities = np.tile(prob, [height, width, 1])
    for y in range(height):
        for x in range(width):
            if patches[y][x] != 0:
                potential = np.ones(num_surfaces)
                potential[patches[y][x]] = 0
                potential[0] = 10
                unary_potentials[y][x] = potential
                prob = np.ones(num_surfaces)/(num_surfaces-2)*0.1
                prob[patches[y][x]] = 0.9
                prob[0] = 0
                initial_probabilities[y][x] = prob
    return unary_potentials, initial_probabilities, prob_copy

def extract_features(depth_image, lab_image, angles, grid):
    features_1_new = np.concatenate([grid, np.expand_dims(depth_image, axis=-1)], axis=-1)
    features_2_new = np.concatenate([features_1_new, lab_image], axis=-1)
    features_3_new = np.concatenate([features_1_new, angles], axis=-1)

    return (features_1_new, features_2_new, features_3_new)

def assemble_outputs(outputs, div_x, div_y, size_x, size_y, height, width, num_surfaces, intermediate=False):
    if intermediate:
        Q = np.zeros((height, width, *np.shape(outputs)[3:]))
    else:
        Q = np.zeros((height, width, num_surfaces))

    index = 0
    for x in range(div_x):
        for y in range(div_y):
            Q[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x, :] = outputs[index]
            index += 1
    return Q

def remove_noise(image):
    max_image = np.argmax(image, axis=-1)
    for y in range(1, height-1):
        for x in range(1, width-1):
            s = max_image[y][x]
            if s != 0:
                if max_image[y-1][x] != s and max_image[y+1][x] != s and max_image[y][x-1] != s and max_image[y][x-1] != s:
                    image[y][x] = 0
    return image

def train_model_on_images(image_indices, load_index=-1, save_index=4, epochs=1, kernel_size=10):
    div_x, div_y = 40, 40
    size_x, size_y = int(width / div_x), int(height / div_y)

    smoothing_model = gaussian_filter_with_depth_factor_model_GPU()
    normals_and_log_depth = normals_and_log_depth_model_GPU()
    surface_model = find_surfaces_model_GPU()
    conv_crf_model = conv_crf(*load_parameters(load_index), kernel_size, size_y, size_x)

    grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = np.stack([grid[1], grid[0]], axis=-1)
    for epoch in range(epochs):
        for image_index in image_indices:
            print(f"Training on image {image_index}")
            
            data = {}
            data["depth"], data["rgb"], data["annotation"] = load_image(image_index)
            smoothed_depth = smoothing_model(data["depth"], grid)

            log_depth, data["angles"], data["vectors"], data["points_3d"] = normals_and_log_depth(smoothed_depth)
            features = extract_features(smoothed_depth, data["rgb"], data["angles"], grid)

            data["patches"], data["depth_edges"] = surface_model(smoothed_depth)
            #plot_image.plot_surfaces(data["patches"])
            data["num_surfaces"] = int(np.max(data["patches"]) + 1)
            unary_potentials, initial_Q, prob = get_unary_potentials_and_initial_probabilities(data)

            print_parameters(conv_crf_model)

            Y, indices = get_training_targets(data, div_x, div_y, size_x, size_y)
            X = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y, indices)

            try:
                conv_crf_model.fit(X, Y, batch_size=1)
            except:
                print("Fail!")

            X = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y, range(div_y*div_x))
            out = conv_crf_model.predict(X, batch_size=1)
            Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, data["num_surfaces"])
            Q[data["depth"] == 0] = prob
            X = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y, indices)
            try:
                conv_crf_model.fit(X, Y, batch_size=1)
            except:
                print("Fail!")

            save_parameters(conv_crf_model, save_index)

def test_model_on_image(image_indices, load_index=-1, kernel_size=7):
    div_x, div_y = 4, 4
    size_x, size_y = int(width / div_x), int(height / div_y)

    smoothing_model = gaussian_filter_with_depth_factor_model_GPU()
    normals_and_log_depth = normals_and_log_depth_model_GPU()
    surface_model = find_surfaces_model_GPU()
    conv_crf_model = conv_crf(*load_parameters(load_index), kernel_size, size_y, size_x)
    conv_crf_model_intermediate = conv_crf(*load_parameters(load_index), kernel_size, size_y, size_x, True)

    print_parameters(conv_crf_model)

    results = []
    grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = np.stack([grid[1], grid[0]], axis=-1)
    for index in image_indices:
        print(index)

        data = {}
        data["depth"], data["rgb"], data["annotation"] = load_image(index)
        smoothed_depth = smoothing_model(data["depth"], grid)

        log_depth, data["angles"], data["vectors"], data["points_3d"] = normals_and_log_depth(smoothed_depth)
        features = extract_features(data["depth"], data["rgb"], data["angles"], grid)

        data["patches"], data["depth_edges"] = surface_model(smoothed_depth)
        plot_surfaces(data["patches"])
        data["num_surfaces"] = int(np.max(data["patches"]) + 1)
        unary_potentials, initial_Q, prob = get_unary_potentials_and_initial_probabilities(data)

        inputs = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y)
        out = conv_crf_model.predict(inputs, batch_size=1)
        Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, data["num_surfaces"])
        Q[data["depth"] == 0] = prob

        inputs = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
        out = conv_crf_model.predict(inputs, batch_size=1)
        Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, data["num_surfaces"])
        Q[data["depth"] == 0] = prob

        inputs = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
        out = conv_crf_model.predict(inputs, batch_size=1)
        Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, data["num_surfaces"])
        Q[data["depth"] == 0] = prob
        data["surfaces"] = np.argmax(Q, axis=-1)
        plot_surfaces(Q, True)

        inputs = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
        out = conv_crf_model_intermediate.predict(inputs, batch_size=1)
        Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, data["num_surfaces"], True)
        Q[data["depth"] == 0] = 0
        data["surfaces"] = np.argmax(Q, axis=-1)

        plot_surfaces(Q, True)
        results.append(Q)
        os.makedirs(f"out/{index}", exist_ok=True)
        np.save(f"out/{index}/Q.npy", Q)
        np.save(f"out/{index}/depth.npy", data["depth"])
        np.save(f"out/{index}/angles.npy", data["angles"])
        np.save(f"out/{index}/vectors.npy", data["vectors"])
        np.save(f"out/{index}/patches.npy", data["patches"])
        np.save(f"out/{index}/points.npy", data["points_3d"])
        np.save(f"out/{index}/edges.npy", data["depth_edges"])
    return results

if __name__ == '__main__':
    test_model_on_image(list(range(106, 111)), load_index=-1)
    quit()
