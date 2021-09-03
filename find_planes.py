import matplotlib.pyplot as plt
import time

import numpy as np

import calculate_normals
import find_edges
import plot_image
import standard_values
from image_processing_models_GPU import *
from calculate_normals import *
from numba import njit, prange
from load_images import *
from image_operations import convert_depth_image
from skimage import measure
from standard_values import *

train_indices = [109, 108, 107, 105, 104, 103, 102, 101, 100]
test_indices = [110, 106]

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

def remove_noise(image):
    max_image = np.argmax(image, axis=-1)
    for y in range(1, height-1):
        for x in range(1, width-1):
            s = max_image[y][x]
            if s != 0:
                if max_image[y-1][x] != s and max_image[y+1][x] != s and max_image[y][x-1] != s and max_image[y][x-1] != s:
                    image[y][x] = 0
    return image


def train_model_on_images(image_indices, load_index=2, save_index=3, epochs=1, kernel_size=10):
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

            depth_image, rgb_image, annotation = load_image(image_index)
            smoothed_depth = smoothing_model(depth_image, grid)

            log_depth, angles, vectors, points_in_space = normals_and_log_depth(smoothed_depth)
            features = extract_features(log_depth, rgb_image, angles, grid)

            surfaces = surface_model(smoothed_depth)

            number_of_surfaces = int(np.max(surfaces) + 1)
            unary_potentials, initial_Q = get_unary_potentials_and_initial_probabilities(surfaces, number_of_surfaces)

            print_parameters(conv_crf_model)

            X = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y)
            Y = get_training_targets(surfaces, annotation, number_of_surfaces, log_depth, div_x, div_y, size_x, size_y)

            conv_crf_model.fit(X, Y, batch_size=1)

            out = conv_crf_model.predict(X, batch_size=1)
            Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, number_of_surfaces)
            Q[depth_image == 0] = 0
            X = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
            conv_crf_model.fit(X, Y, batch_size=1)

            save_parameters(conv_crf_model, save_index)

def test_model_on_image(image_indices, load_index=-1, kernel_size=7):
    div_x, div_y = 4, 4
    size_x, size_y = int(width / div_x), int(height / div_y)

    smoothing_model = gaussian_filter_with_depth_factor_model_GPU()
    normals_and_log_depth = normals_and_log_depth_model_GPU()
    surface_model = find_surfaces_model_GPU()
    conv_crf_model = conv_crf_depth(*load_parameters(load_index), kernel_size, size_y, size_x)
    print_parameters(conv_crf_model)
    print(*load_parameters(load_index))

    results = []
    grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = np.stack([grid[1], grid[0]], axis=-1)
    for index in image_indices:
        print(index)
        depth_image, rgb_image, annotation = load_image(index)

        t = time.time()
        smoothed_depth = smoothing_model(depth_image, grid)
        log_depth, angles, vectors, points_in_space = normals_and_log_depth(smoothed_depth)
        features = extract_features(log_depth, rgb_image, angles, grid)

        surfaces, depth_edges = surface_model(smoothed_depth)
        plot_surfaces(surfaces, False)
        #plot_surfaces(surfaces, False)
        number_of_surfaces = int(np.max(surfaces) + 1)
        unary_potentials, initial_Q, prob = get_unary_potentials_and_initial_probabilities(surfaces, number_of_surfaces)

        data = get_inputs(features, unary_potentials, initial_Q, kernel_size, div_x, div_y, size_x, size_y)
        out = conv_crf_model.predict(data, batch_size=1)
        Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, number_of_surfaces)
        Q[depth_image == 0] = prob

        data = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
        out = conv_crf_model.predict(data, batch_size=1)
        Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, number_of_surfaces)
        Q[depth_image == 0] = prob

        data = get_inputs(features, unary_potentials, Q, kernel_size, div_x, div_y, size_x, size_y)
        out = conv_crf_model.predict(data, batch_size=1)
        Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width, number_of_surfaces)
        Q[depth_image == 0] = 0

        print(time.time()-t)
        plot_surfaces(Q)
        results.append(Q)
        os.makedirs(f"out/{index}", exist_ok=True)
        np.save(f"out/{index}/Q.npy", Q)
        np.save(f"out/{index}/depth.npy", depth_image)
        np.save(f"out/{index}/angles.npy", angles)
        np.save(f"out/{index}/vectors.npy", vectors)
        np.save(f"out/{index}/patches.npy", surfaces)
        np.save(f"out/{index}/points.npy", points_in_space)
        np.save(f"out/{index}/edges.npy", depth_edges)
        #return Q, depth_image, angles
    return results

def get_unary_potentials_and_initial_probabilities(surface_image, number_of_labels):
    height, width = np.shape(surface_image)
    potential = np.ones((1, 1, number_of_labels))
    potential[0] = 10
    unary_potentials = np.tile(potential, [height, width, 1])
    prob = np.ones((1, 1, number_of_labels)) / (number_of_labels - 1)
    prob[0] = 0
    initial_probabilities = np.tile(prob, [height, width, 1])
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
    return unary_potentials, initial_probabilities, prob

def extract_features(depth_image, lab_image, angles, grid):
    features_1_new = np.concatenate([grid, np.expand_dims(depth_image, axis=-1)], axis=-1)
    features_2_new = np.concatenate([features_1_new, lab_image], axis=-1)
    features_3_new = np.concatenate([features_1_new, angles], axis=-1)

    return (features_1_new, features_2_new, features_3_new)

def plot_surfaces(Q, max=True):
    if max:
        image = np.argmax(Q, axis=-1)
        image = np.reshape(image, (480, 640))
    else:
        image = np.reshape(Q, (480, 640))
    plt.imshow(image, cmap='nipy_spectral')
    plt.show()

if __name__ == '__main__':
    #train_model_on_images(train_indices)
    #test_model_on_image([0], load_index=2)
    #quit()
    test_model_on_image(list(range(107, 111)), load_index=-1)
    quit()
