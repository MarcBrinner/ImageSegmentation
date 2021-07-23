import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import plot_image
import standard_values
from tensorflow.keras import layers, Model, initializers, optimizers, regularizers
from mean_field_update_tf import *
from calculate_normals import *
from numba import njit, prange
from load_images import *
from image_operations import convert_depth_image
from skimage import measure

train_indices = [109, 108, 107, 105, 104, 103, 102, 101, 100]
test_indices = [110, 106]

def find_edges_model_GPU(depth=4, factor_array=None, threshold=0.098046811455665408):
    if factor_array is None:
        factor_array = [standard_values.factor_y, standard_values.factor_x]

    depth_image_in = layers.Input(shape=(480, 640), batch_size=1)
    depth_image = tf.squeeze(depth_image_in, axis=0)

    shape = (tf.shape(depth_image)[0] - 2*depth, tf.shape(depth_image)[1] - 2*depth)

    curvature_scores = tf.zeros(shape)
    factors = tf.broadcast_to(tf.reshape(tf.constant(factor_array, dtype=tf.float32), (1, 1, 2)), (shape[0], shape[1], 2))
    central_points = tf.concat([tf.zeros((shape[0], shape[1], 2)), tf.expand_dims(depth_image[depth:-depth, depth:-depth], axis=-1)], axis=-1)

    for direction in [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]:
        d = tf.reshape(tf.constant(direction, dtype=tf.float32), (1, 1, 2))

        current_scores = tf.zeros_like(curvature_scores)
        prev_distances = tf.zeros_like(curvature_scores)
        prev_points = central_points
        for k in range(1, depth):
            new_points = tf.concat([factors*k*d, tf.expand_dims(depth_image[depth-k*direction[0]: -(depth+k*direction[0]), depth-k*direction[1]: -(depth+k*direction[1])], axis=-1)], axis=-1)
            dif_1 = tf.sqrt(tf.reduce_sum(tf.square(new_points - prev_points), axis=-1))
            dif_2 = tf.sqrt(tf.reduce_sum(tf.square(new_points - central_points), axis=-1))
            score = tf.math.divide_no_nan((dif_1 + prev_distances), dif_2) - 1
            current_scores = current_scores + score
            curvature_scores = curvature_scores + current_scores
            prev_distances = dif_2
            prev_points = new_points

    pixels = tf.cast(tf.less(curvature_scores, threshold), tf.int32)

    model = Model(inputs=[depth_image_in], outputs=pixels)
    return model

@njit()
def extract_data_points(depth_image, normal_image):
    points = []
    height, width = np.shape(depth_image)

    coordinate_normalizer = min(width, height)
    for y in range(0, height):
        for x in range(0, width):
            points.append(
                [x / coordinate_normalizer, y / coordinate_normalizer, normal_image[y][x][0], normal_image[y][x][1], normal_image[y][x][2], depth_image[y][x]])
    return points

def find_annotation_correspondence(surfaces, annotation):
    mapping = {i: set() for i in range(int(np.max(annotation)+1))}
    height, width = np.shape(surfaces)
    for y in range(height):
        for x in range(width):
            index = surfaces[y][x]
            if index == 0:
                continue
            mapping[annotation[y][x]].add(index)
    return mapping

def get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y):
    features_1, features_2, features_3 = [np.pad(f, [[kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]]) for f in features]
    Q = np.pad(Q, [[kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]])
    f_1 = []
    f_2 = []
    f_3 = []
    Q_in = []
    unary = []

    for x in range(div_x):
        for y in range(div_y):
            f_1.append(features_1[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            f_2.append(features_2[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            f_3.append(features_3[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            Q_in.append(Q[y*size_y:y*size_y + size_y+2*kernel_size, x*size_x:x*size_x + size_x + 2*kernel_size])
            unary.append(unary_potentials[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x])
    return [np.asarray(x) for x in [unary, Q_in, f_1, f_2, f_3]]

def get_training_targets(surfaces, annotation, number_of_surfaces, depth_image, div_x, div_y, size_x, size_y):
    correspondence = find_annotation_correspondence(surfaces, annotation)
    height, width = np.shape(depth_image)
    Y = np.zeros((height, width, number_of_surfaces))
    for y in range(height):
        for x in range(width):
            if depth_image[y][x] < 0.001:
                continue
            label = np.ones(number_of_surfaces)
            for l in correspondence[annotation[y][x]]:
                label[l] = 0
            Y[y][x] = label
    Y_batch = []
    for x in range(div_x):
        for y in range(div_y):
            Y_batch.append(Y[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x, :])
    return np.asarray(Y_batch)

def assemble_outputs(outputs, div_x, div_y, size_x, size_y, height, width, number_of_surfaces):
    Q = np.zeros((height, width, number_of_surfaces))
    index = 0
    for x in range(div_x):
        for y in range(div_y):
            Q[y*size_y:y*size_y + size_y, x*size_x:x*size_x + size_x, :] = outputs[index]
            index += 1
    return Q

def train_model_on_images(image_indices, load_index=-1, save_index=2, epochs=1, kernel_size=10):
    height, width = 480, 640
    div_x, div_y = 20, 20
    size_x, size_y = int(width / div_x), int(height / div_y)

    parameters = load_parameters(load_index)
    MFI_NN = conv_crf(*parameters, kernel_size, size_y, size_x)

    for epoch in range(epochs):
        for image_index in image_indices:
            print(f"Training on image {image_index}")

            depth_image, rgb_image, annotation = load_image(image_index)
            log_depth = convert_depth_image(depth_image)
            surfaces = find_smooth_surfaces_with_curvature_scores(depth_image)
            features = extract_features_conv(log_depth, rgb_image)
            number_of_surfaces = int(np.max(surfaces)+1)
            unary_potentials, initial_Q = get_unary_potentials_and_initial_probabilities_conv(surfaces, number_of_surfaces)

            print_parameters(MFI_NN)

            X = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y)
            Y = get_training_targets(surfaces, annotation, number_of_surfaces, log_depth, div_x, div_y, size_x, size_y)

            MFI_NN.fit(X, Y, batch_size=1)

            save_parameters(MFI_NN, save_index)

def test_model_on_image(image_index, load_index=-1, load_iteration=0, kernel_size = 10):
    depth_image, rgb_image, annotation = load_image(image_index)
    height, width = np.shape(depth_image)
    log_depth = convert_depth_image(depth_image)
    surfaces = find_smooth_surfaces_with_curvature_scores(depth_image)
    features = extract_features_conv(log_depth, rgb_image, log_depth)
    number_of_surfaces = int(np.max(surfaces) + 1)
    unary_potentials, initial_Q = get_unary_potentials_and_initial_probabilities_conv(surfaces, number_of_surfaces)
    if load_iteration >= 1:
        initial_Q = np.load(f"it_{load_iteration}.npy")

    parameters = load_parameters(load_index)

    div_x, div_y = 4, 4
    size_x, size_y = int(width / div_x), int(height / div_y)
    dataset = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y)

    MFI_NN = conv_crf(*parameters, kernel_size, size_y, size_x)
    out = MFI_NN.predict(dataset, batch_size=1)

    Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, number_of_surfaces)
    plot_surfaces(Q)

@njit()
def do_iteration_2(image, number, size):
    height, width = np.shape(image)
    new_image = np.zeros(np.shape(image))
    for i in range(size, height-size):
        for j in range(size, width-size):
            counter = 0
            for k in range(max(0, i - size), min(height, i + size + 1)):
                for l in range(max(0, j - size), min(width, j + size + 1)):
                    counter += image[k][l]
            if counter > number:
                new_image[i][j] = 1.0
    return new_image

@njit()
def smooth_surface_calculations(depth_image):
    height, width = np.shape(depth_image)
    plane_image = np.zeros((height, width))
    factor_x = math.tan(viewing_angle_x / 2) * 2 / width
    factor_y = math.tan(viewing_angle_y / 2) * 2 / height
    threshold = 0.007003343675404672
    size = 4
    positions = [[0, size], [0, 2*size], [size, 2*size], [2*size, 2*size], [2*size, size], [2*size, 0], [size, 0], [0, 0]]
    for y in prange(height):
        #print(y)
        for x in range(width):
            d = depth_image[y][x]
            if d < 0.0001:
                continue
            depth_factor_x = factor_x * d
            depth_factor_y = factor_y * d
            curvature_scores = calculate_curvature_scores(depth_image, size, y, x, np.asarray([depth_factor_x, depth_factor_y]), d, width, height)

            for i in range(len(positions)):
                if np.max(np.abs(np.asarray([curvature_scores[p[0]][p[1]] for p in [positions[i], positions[i-1], positions[i-2], positions[i-3]]]))) < 0 or \
                    np.sum(curvature_scores) < threshold*14:
                    plane_image[y][x] = 1
    return plane_image

@njit()
def remove_small_patches(index_image, surface_image, segment_count):
    height, width = np.shape(index_image)
    counter = np.zeros(segment_count+1, dtype="uint32")
    neighbors = []
    for i in range(segment_count+1):
        neighbors.append([0])
    zero_indices = set()
    for y in range(height):
        for x in range(width):
            index = index_image[y][x]
            counter[index] += 1
            if y < height - 1:
                other_index = index_image[y + 1][x]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)
            if x < width - 1:
                other_index = index_image[y][x+1]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)
            if surface_image[y][x] == 0:
                zero_indices.add(index)

    for i in range(len(neighbors)):
        new_list = list(set(neighbors[i]))
        new_list.remove(0)
        if i in new_list:
            new_list.remove(i)
        neighbors[i] = new_list

    relabeling = {0: 0}
    too_small = set()
    for i in range(1, segment_count+1):
        if counter[i] <= 10:
            too_small.add(i)
            neighbors[neighbors[i][0]].remove(i)

    for val in zero_indices:
        if len(neighbors[val]) == 1:
            relabeling[val] = neighbors[val][0]
        else:
            relabeling[val] = 0

    for val in too_small:
        if val not in relabeling:
            relabeling[val] = relabeling[neighbors[val][0]]

    free_indices = set(list(too_small) + list(zero_indices))
    new_segment_count = segment_count - len(free_indices) + 1
    free_indices = [x for x in free_indices if x < new_segment_count]
    for i in range(segment_count+1):
        if i not in relabeling:
            if i < new_segment_count:
                relabeling[i] = i
            else:
                relabeling[i] = free_indices.pop()
                for j in range(segment_count+1):
                    if j in relabeling and relabeling[j] == i:
                        relabeling[j] = relabeling[i]

    for y in range(height):
        for x in range(width):
            index_image[y][x] = relabeling[index_image[y][x]]

@njit()
def get_unary_potentials_and_initial_probabilities_conv(surface_image, number_of_labels):
    height, width = np.shape(surface_image)
    unary_potentials = np.zeros((height, width, number_of_labels))
    initial_probabilities = np.zeros((height, width, number_of_labels))
    for y in range(height):
        for x in range(width):
            if surface_image[y][x] != 0:
                potential = np.ones(number_of_labels)
                potential[surface_image[y][x]] = 0
                potential[0] = 10
                unary_potentials[y][x] = potential
                prob = np.ones(number_of_labels)/(number_of_labels-2)*0.1
                prob[surface_image[y][x]] = 0.9
                prob[0] = 0
                initial_probabilities[y][x] = prob
    return unary_potentials, initial_probabilities

@njit()
def extract_features_conv(depth_image, lab_image, log_depth):
    angle_image = calculate_normals_as_angles_final(depth_image, log_depth)
    height, width = np.shape(depth_image)
    features_1 = np.zeros((height, width, 3))
    features_2 = np.zeros((height, width, 6))
    features_3 = np.zeros((height, width, 5))
    for y in range(height):
        for x in range(width):
            features_1[y][x] = [y, x, depth_image[y][x]]
            features_2[y][x] = [y, x, depth_image[y][x], lab_image[y][x][0], lab_image[y][x][1], lab_image[y][x][2]]
            features_3[y][x] = [y, x, depth_image[y][x], angle_image[y][x][0], angle_image[y][x][1]]

    return features_1, features_2, features_3

def plot_surfaces(Q, max=True):
    if max:
        image = np.argmax(Q, axis=-1)
        image = np.reshape(image, (480, 640))
    else:
        image = np.reshape(Q, (480, 640))
    plt.imshow(image, cmap='nipy_spectral')
    plt.show()

def find_smooth_surfaces_with_curvature_scores(depth_image):
    depth_image = gaussian_filter_with_depth_factor(depth_image, 4)
    print("Filter applied.")

    surfaces = smooth_surface_calculations(depth_image)
    #surface_model = find_edges_model_GPU(4)
    #surfaces = surface_model.predict(np.asarray([depth_image]))
    #surfaces = np.pad(surfaces, [[4, 4], [4, 4]])
    print("Surface patches found.")
    surfaces = do_iteration_2(surfaces, 11, 2)
    surfaces = do_iteration_2(surfaces, 5, 1)
    print("Smoothing iterations done.")
    #indexed_surfaces = find_consecutive_patches(surfaces)
    indexed_surfaces, segment_count = measure.label(surfaces, background=-1, return_num=True)
    remove_small_patches(indexed_surfaces, surfaces, segment_count)
    print("Image cleaning done.")
    #colored_image = color_patches(indexed_surfaces)
    plot_surfaces(indexed_surfaces, False)
    return indexed_surfaces

if __name__ == '__main__':
    # image = load_image(110)
    # model = find_edges_model_GPU(4)
    # out = model.predict(np.asarray([image[0]]))
    # t = time.time()
    # out = model.predict(np.asarray([image[0]]))
    # print(time.time()-t)
    # plot_image.plot_array_PLT(out)
    # print()
    # quit()
    #train_model_on_images(train_indices)
    test_model_on_image(test_indices[0])
    quit()

