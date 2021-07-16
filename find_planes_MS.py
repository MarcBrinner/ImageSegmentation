import seaborn as sns
import matplotlib.pyplot as plt
import time
import find_edges
import multiprocessing
from mean_field_update_tf import *
from calculate_normals import *
from plot_image import *
from collections import Counter
from numba import njit, prange
from load_images import *
from image_operations import convert_depth_image, rgb_to_Lab, calculate_curvature_scores_2
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from skimage import measure

train_indices = [109, 108, 107, 105, 104, 103, 102, 101, 100]
test_indices = [110, 106]

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

def find_planes_MS(depth_image, rgb_image):
    normal_image = calculate_normal_vectors_like_paper(depth_image)
    converted_depth_image = convert_depth_image(depth_image)
    data_points = np.asarray(extract_data_points(converted_depth_image, normal_image))
    ms = MeanShift(bandwidth=0.3).fit_predict(data_points)
    #ms = KMeans(n_init=1, n_clusters=15).fit_predict(data_points)
    counts = dict(Counter(ms))
    palette = np.asarray(sns.color_palette(None, len(counts)))
    height, width = np.shape(depth_image)
    print_image = np.zeros((height, width, 3))
    counter = 0
    for y in range(0, height):
        for x in range(0, width):
            print_image[y][x] = palette[ms[counter]]
            counter += 1
    plot_array(np.asarray(print_image*255, dtype="uint8"))

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

def get_initial_guess_parameters():
    return list(np.reshape([0.8, 0.8, 1.5, 1/80, 2500.0, 1/100, 2500.0, 1/200, 1/100, 2500.0, 20.0, 0.01], (12, 1, 1)))

def calculate_labels(surfaces, annotation, correspondence, number_of_surfaces, depth_image, number_to_fill):
    labels = []
    height, width = np.shape(surfaces)
    for y in range(height):
        for x in range(width):
            if depth_image[y][x] > 0.0001:
                label = np.ones(number_of_surfaces)
                for l in correspondence[annotation[y][x]]:
                    label[l] = 0
                labels.append(label)
    for i in range(number_to_fill):
        labels.append(np.zeros(number_of_surfaces))
    return np.asarray(labels, dtype="float64")


def train_model_on_images(image_indices, load_index=-1, save_index=2, batch_size=16, epochs=1):
    for epoch in range(epochs):
        for image_index in image_indices:
            print(f"Training on image {image_index}")
            depth_image, rgb_image, annotation = load_image(image_index)
            log_depth = convert_depth_image(depth_image)
            surfaces = find_smooth_surfaces_with_curvature_scores(depth_image)
            correspondence = find_annotation_correspondence(surfaces, annotation)
            features, number_to_fill, indices = extract_features(log_depth, rgb_image, batch_size)
            features = [np.asarray(f) for f in features]
            number_of_surfaces = int(np.max(surfaces)+1)
            unary_potentials, initial_Q = get_unary_potentials_and_initial_probabilities(surfaces, number_of_surfaces, depth_image, number_to_fill)
            initial_Q = np.asarray(initial_Q)
            unary_potentials = np.asarray(unary_potentials)
            if load_index >= 0:
                parameters = load_parameters(load_index)
            else:
                parameters = get_initial_guess_parameters()
                load_index = save_index

            print(parameters)

            matrix = np.ones((number_of_surfaces, number_of_surfaces)) - np.identity(number_of_surfaces)
            MFI_NN = mean_field_update_model_learned_2(np.shape(initial_Q)[0], number_of_surfaces, initial_Q, *features,
                                             matrix, *parameters, batch_size)

            print_parameters(MFI_NN)
            Y = calculate_labels(surfaces, annotation, correspondence, number_of_surfaces, log_depth, number_to_fill)

            # test_index = 371 * 640 + 524
            # out = MFI_NN.fit([features[0][test_index:test_index+batch_size], features[1][test_index:test_index+batch_size],
            #                      features[2][test_index:test_index+batch_size], features[3][test_index:test_index+batch_size],
            #                      features[4][test_index:test_index+batch_size], unary_potentials[test_index:test_index+batch_size]], Y[test_index: test_index+batch_size], batch_size=batch_size, epochs=20)
            # print(out)
            # quit()
            MFI_NN.fit([*features, unary_potentials], Y, batch_size=batch_size, epochs=1)

            save_parameters(MFI_NN, save_index)

def test_model_on_image(image_index, load_index=2, load_iteration=0, batch_size=16):
    depth_image, rgb_image, annotation = load_image(image_index)
    log_depth = convert_depth_image(depth_image)
    surfaces = find_smooth_surfaces_with_curvature_scores(depth_image)
    features, number_to_fill, indices = extract_features(log_depth, rgb_image, batch_size)
    features = [np.asarray(f) for f in features]
    number_of_surfaces = int(np.max(surfaces) + 1)
    unary_potentials, initial_Q = get_unary_potentials_and_initial_probabilities(surfaces, number_of_surfaces, depth_image, number_to_fill)
    initial_Q = np.asarray(initial_Q)
    unary_potentials = np.asarray(unary_potentials)
    if load_iteration >= 1:
        initial_Q = np.load(f"it_{load_iteration}.npy")

    parameters = load_parameters(load_index)

    matrix = np.ones((number_of_surfaces, number_of_surfaces)) - np.identity(number_of_surfaces)
    MFI_NN = mean_field_update_model_learned_2(np.shape(initial_Q)[0], number_of_surfaces, initial_Q, *features,
                                             matrix, *parameters, batch_size)
    print_parameters(MFI_NN)

    # test_index = 371 * 640 + 524
    # out = MFI_NN.predict([features[0][test_index:test_index+batch_size], features[1][test_index:test_index+batch_size],
    #                      features[2][test_index:test_index+batch_size], features[3][test_index:test_index+batch_size],
    #                      features[4][test_index:test_index+batch_size], unary_potentials[test_index:test_index+batch_size]], batch_size=batch_size)
    # print(out)

    Q = MFI_NN.predict([*features, unary_potentials], batch_size=batch_size)
    Q_complete = np.zeros((480*640, number_of_surfaces))
    for i in range(len(indices)):
        Q_complete[indices[i]] = Q[i]
    plot_surfaces(Q_complete)

    np.save(f"it_{load_iteration+1}.npy", Q_complete)

def test_model_on_image_2(image_index, load_index=2, load_iteration=0, batch_size=16):
    depth_image, rgb_image, annotation = load_image(image_index)
    log_depth = convert_depth_image(depth_image)
    surfaces = find_smooth_surfaces_with_curvature_scores(depth_image)
    features = extract_features_conv(log_depth, rgb_image)
    number_of_surfaces = int(np.max(surfaces) + 1)
    unary_potentials, initial_Q = get_unary_potentials_and_initial_probabilities_conv(surfaces, number_of_surfaces)
    if load_iteration >= 1:
        initial_Q = np.load(f"it_{load_iteration}.npy")

    parameters = get_initial_guess_parameters()#load_parameters(load_index)
    for f in features:
        print(np.shape(f))
    print(np.shape(unary_potentials))
    print(np.shape(initial_Q))
    matrix = np.ones((number_of_surfaces, number_of_surfaces)) - np.identity(number_of_surfaces)
    MFI_NN = mean_field_convolutional(number_of_surfaces, matrix, *parameters, 640, 480, 6)

    # test_index = 371 * 640 + 524
    # out = MFI_NN.predict([features[0][test_index:test_index+batch_size], features[1][test_index:test_index+batch_size],
    #                      features[2][test_index:test_index+batch_size], features[3][test_index:test_index+batch_size],
    #                      features[4][test_index:test_index+batch_size], unary_potentials[test_index:test_index+batch_size]], batch_size=batch_size)
    # print(out)

    Q = MFI_NN.predict([*[np.asarray([f]) for f in features], np.asarray([unary_potentials]), np.asarray([initial_Q])], batch_size=batch_size)
    plot_surfaces(Q)

    #np.save(f"it_{load_iteration+1}.npy", Q_complete)


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
                    np.sum(curvature_scores) < threshold*32:
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
def get_unary_potentials_and_initial_probabilities(surface_image, number_of_labels, depth_image, number_to_fill):
    height, width = np.shape(surface_image)
    unary_potentials = []
    initial_probabilities = []
    index = 0
    for y in range(height):
        for x in range(width):
            if depth_image[y][x] > 0.0001:
                if surface_image[y][x] == 0:
                    potential = np.ones(number_of_labels)/(number_of_labels-1)
                    potential[0] = 10
                    unary_potentials.append(potential)
                    prob = potential.copy()
                    prob[0] = 0
                    initial_probabilities.append(prob)
                else:
                    potential = np.ones(number_of_labels)
                    potential[surface_image[y][x]] = 0
                    potential[0] = 10
                    unary_potentials.append(potential)
                    prob = np.ones(number_of_labels)/(number_of_labels-2)*0.1
                    prob[surface_image[y][x]] = 0.9
                    prob[0] = 0
                    initial_probabilities.append(prob)
            index += 1
    for i in range(number_to_fill):
        unary_potentials.append(np.zeros(number_of_labels))
        initial_probabilities.append(np.zeros(number_of_labels))
    return unary_potentials, initial_probabilities

@njit()
def get_unary_potentials_and_initial_probabilities_conv(surface_image, number_of_labels):
    height, width = np.shape(surface_image)
    unary_potentials = np.zeros((height, width, number_of_labels))
    initial_probabilities = np.zeros((height, width, number_of_labels))
    for y in range(height):
        for x in range(width):
            if surface_image[y][x] == 0:
                potential = np.ones(number_of_labels)/(number_of_labels-1)
                potential[0] = 10
                unary_potentials[y][x] = potential
                prob = potential.copy()
                prob[0] = 0
                initial_probabilities[y][x] = prob
            else:
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
def extract_features(depth_image, lab_image, desired_batch_size):
    angle_image = calculate_normals_as_angles_final(depth_image)
    height, width = np.shape(depth_image)
    features_1 = []
    features_3 = []
    features_4 = []
    indices = []
    index = 0
    for y in range(height):
        for x in range(width):
            if depth_image[y][x] > 0.0001:
                indices.append(index)
                features_1.append(np.asarray([y, x, depth_image[y][x]]))
                features_3.append(np.asarray([y, x, depth_image[y][x], lab_image[y][x][0], lab_image[y][x][1], lab_image[y][x][2]]))
                features_4.append(np.asarray([y, x, depth_image[y][x], angle_image[y][x][0], angle_image[y][x][1]]))
            index += 1
    num_batches = len(indices)/desired_batch_size
    number_to_fill = int((int(np.ceil(num_batches)) - num_batches) * desired_batch_size)
    for i in range(number_to_fill):
        features_1.append(np.asarray([0.0, 0.0, 0.0]))
        features_3.append(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        features_4.append(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0]))
    return (features_1, features_3, features_4), number_to_fill, indices

@njit()
def extract_features_conv(depth_image, lab_image):
    angle_image = calculate_normals_as_angles_final(depth_image)
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

def mean_field_update_NN(Q, unary_potentials, features, theta_1, theta_2, theta_3, theta_4, theta_5, w_1, w_2, w_3, w_4, w_5, weight, number_of_labels, batch_size = 32):
    features_1, features_2, features_3, features_4, features_5 = features
    n, _ = np.shape(features_1)

    matrix = np.ones((number_of_labels, number_of_labels)) - np.identity(number_of_labels)
    MFI_NN = mean_field_update_model(n, number_of_labels, Q, features_1, features_2, features_3, features_4, features_5,
                                     matrix, w_1, w_2, w_3, w_4, w_5, theta_1, theta_2, theta_3, theta_4, theta_5, weight, batch_size)
    #test_index = 371*640+524
    #out = MFI_NN.predict([features_1[test_index:test_index+batch_size], features_2[test_index:test_index+batch_size],
    #                      features_3[test_index:test_index+batch_size], features_4[test_index:test_index+batch_size],
    #                      features_5[test_index:test_index+batch_size], unary_potentials[test_index:test_index+batch_size]], batch_size=batch_size)
    #print()
    Q_new = MFI_NN.predict([features_1, features_2, features_3, features_4, features_5, unary_potentials], batch_size=batch_size)
    return Q_new

def plot_surfaces(Q):
    image = np.argmax(Q, axis=-1)
    image = np.reshape(image, (480, 640))
    plt.imshow(image, cmap='nipy_spectral')
    plt.show()

def plot_surface_image(surfaces):
    plt.imshow(surfaces, cmap='nipy_spectral')
    plt.show()
    return

def find_smooth_surfaces_with_curvature_scores(depth_image):
    depth_image = gaussian_filter_with_depth_factor(depth_image, 4)
    print("Filter applied.")
    surfaces = smooth_surface_calculations(depth_image)
    print("Surface patches found.")
    surfaces = do_iteration_2(surfaces, 11, 2)
    surfaces = do_iteration_2(surfaces, 5, 1)
    print("Smoothing iterations done.")
    #indexed_surfaces = find_consecutive_patches(surfaces)
    indexed_surfaces, segment_count = measure.label(surfaces, background=-1, return_num=True)
    remove_small_patches(indexed_surfaces, surfaces, segment_count)
    print("Image cleaning done.")
    #colored_image = color_patches(indexed_surfaces)
    plot_surface_image(indexed_surfaces)
    return indexed_surfaces

def assign_all_pixels_to_surfaces(surface_image, depth_image, rgb_image, load_iteration=1):
    log_depth = convert_depth_image(depth_image)
    features = extract_features(log_depth, rgb_image)
    print("Features extracted.")
    number_of_labels = int(np.max(surface_image)+1)
    unary_potentials, Q = get_unary_potentials_and_initial_probabilities(surface_image, number_of_labels)
    print("Potentials calculated.")
    #Q = np.transpose(unary_potentials)
    #print(np.shape(Q))
    if load_iteration >= 1:
        Q = np.load(f"it_{load_iteration}.npy")
    t = time.time()
    new_Q = mean_field_update_NN(Q, unary_potentials, features, np.asarray([[1/25, 1/25]]), np.asarray([[1/80, 1/80, 100]]),
                                 np.asarray([[1/100, 1/100, 1/200, 1/200, 1/200]]), np.asarray([[1/200, 1/200, 11, 11]]),
                                 np.asarray([[1/300, 1/300, 50, 10, 10]]), 0.2, 0.2, 0.2, 0.2, 2, 0.01, number_of_labels)#0.01
    print(time.time() - t)
    np.save(f"it_{load_iteration+1}.npy", new_Q)
    print("Mean field update 1 done.")
    plot_surfaces(new_Q)

def find_surfaces(depth_image, rgb_image):
    surfaces = find_smooth_surfaces_with_curvature_scores(depth_image)
    assign_all_pixels_to_surfaces(surfaces, depth_image, rgb_image)
    return surfaces

def main():
    images = load_image(110)
    plot_surface_image(images[2])
    #plt.imshow(images[1])
    #plt.show()
    #quit()
    surfaces = find_surfaces(*images)
    #plot_array(surfaces, normalize=True)

def check_image():
    Q = np.load("it_1.npy")
    Q = np.reshape(Q, (480, 640, 47))
    maxes = np.argmax(Q, axis=-1)
    index = maxes == 0
    images = load_image(110)
    plot_array(images[1])
    plot_depth = convert_depth_image(images[0])
    plot_array(np.asarray(plot_depth*255, dtype="uint8"))
    angles = calculate_normals_as_angles_final(images[0])
    new_normals = angles_to_normals(angles, images[0])
    plot_array(np.asarray(new_normals * 127.5 + 127.5, dtype="uint8"))
    plot_surfaces(Q)
    quit()

if __name__ == '__main__':
    #train_model_on_images(train_indices)
    test_model_on_image_2(test_indices[0])
    #check_image()
    #test_model_on_image(110)
    quit()
    #print(np.sum(Q < 700) / np.sum(Q > 0))
    #print(np.min(Q))
    #quit()
    main()
