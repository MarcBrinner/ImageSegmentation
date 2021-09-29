import numpy as np
import assemble_objects
import detect_objects
import tensorflow as tf
import pickle
from image_operations import rgb_to_Lab
from load_images import load_image
from image_processing_models_GPU import Variable2, Print_Tensor
from tensorflow.keras import layers, optimizers, Model, losses
from standard_values import *
from find_planes import plot_surfaces
from sklearn.linear_model import LogisticRegression

#initial_guess_parameters = [1.1154784, 2, 2, 1, 1.1, 3, -1, 2, 1.0323303, 0.24122038]
initial_guess_parameters = [1, 2, 2, 1, 1.1, 3, -1, 2, 1, 0.02]

def load_image_and_surface_information(index):
    depth, rgb, annotation = load_image(index)
    lab = rgb_to_Lab(rgb)
    Q = np.load(f"out/{index}/Q.npy")
    depth_image = np.load(f"out/{index}/depth.npy")
    angles = np.load(f"out/{index}/angles.npy")
    patches = np.load(f"out/{index}/patches.npy")
    points_in_space = np.load(f"out/{index}/points.npy")
    depth_edges = np.load(f"out/{index}/edges.npy")
    Q = np.argmax(Q, axis=-1)
    return Q, depth_edges, rgb, lab, patches, angles, points_in_space, depth_image, annotation

def create_training_set():
    models = get_GPU_models()
    inputs = []
    labels = []
    for index in range(111):
        print(index)
        Q, depth_edges, rgb, lab, patches, angles, points_in_space, depth_image, annotation = load_image_and_surface_information(index)
        info = get_similarity_data_for_CRF(Q, depth_edges, rgb, lab, patches, models, angles, points_in_space, depth_image)
        number_of_surfaces = int(np.max(Q) + 1)
        num_boxes = np.shape(info[0])[0]
        Y = get_Y_value(annotation, Q, number_of_surfaces, num_boxes)[0]
        join_matrix = Y[0]
        not_join_matrix = Y[1]

        for i in range(number_of_surfaces-1):
            for j in range(number_of_surfaces-1):
                if (a := join_matrix[i][j]) == 1:
                    labels.append(1)
                elif (b := not_join_matrix[i][j]) == 1:
                    labels.append(0)
                if a > 0 or b > 0:
                    inputs.append(np.asarray([np.sum(info[0][:, i] * info[0][:, j]), *[info[k][i, j] for k in range(1, 12)]]))
        if index == 100:
            np.save("data/train_in.npy", np.asarray(inputs))
            np.save("data/train_labels.npy", np.asarray(labels))
            inputs = []
            labels = []
        elif index == 110:
            np.save("data/test_in.npy", np.asarray(inputs))
            np.save("data/test_labels.npy", np.asarray(labels))
            inputs = []
            labels = []

    return np.asarray(inputs), np.asarray(labels)

def train_unary_classifier():
    inputs_train = np.load("data/train_in.npy")
    labels_train = np.load("data/train_labels.npy")
    inputs_test = np.load("data/test_in.npy")
    labels_test = np.load("data/test_labels.npy")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(inputs_train, labels_train)
    print(clf.coef_)
    print(clf.intercept_)
    pickle.dump(clf, open("parameters/unary_potential_clf/clf.pkl", "wb"))
    print(np.sum(np.abs(clf.predict(inputs_train)-labels_train))/len(labels_train))
    print(np.sum(np.abs(clf.predict(inputs_test)-labels_test))/len(labels_test))

def assemble_objects_with_unary_classifier():
    clf = pickle.load(open("parameters/unary_potential_clf/clf.pkl", "rb"))
    models = get_GPU_models()
    for index in range(101, 111):
        Q, depth_edges, rgb, lab, patches, angles, points_in_space, depth_image, annotation = load_image_and_surface_information(index)
        info = get_similarity_data_for_CRF(Q, depth_edges, rgb, lab, patches, models, angles, points_in_space,
                                           depth_image)
        number_of_surfaces = int(np.max(Q) + 1)
        input_indices = []
        inputs = []

        for i in range(number_of_surfaces - 1):
            for j in range(number_of_surfaces - 1):
                inputs.append(np.asarray([np.sum(info[0][:, i] * info[0][:, j]), *[info[k][i, j] for k in range(1, 8)]]))
                input_indices.append((i, j))
        predictions = clf.predict(inputs)
        join_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
        for i in range(len(predictions)):
            if predictions[i] == 1:
                indices = input_indices[i]
                join_matrix[indices[0]+1, indices[1]+1] = 1
                join_matrix[indices[1]+1, indices[0]+1] = 1
        s, r = assemble_objects.join_surfaces_according_to_join_matrix(join_matrix, Q, number_of_surfaces)
        plot_surfaces(s)

def print_tensor(input):
    p = Print_Tensor()(input)
    return p

variable_names = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7", "w_8", "w_9", "weight"]

def print_parameters(model):
    for name in variable_names:
        try:
            print(f"{name}: {model.get_layer(name).weights}")
        except:
            print(f"{name}: Not available.")

def get_Y_value(annotation, surfaces, number_of_surfaces, num_boxes):
    num_annotations = np.max(annotation)
    annotation_counter = np.zeros((number_of_surfaces, num_annotations+1))
    annotation_counter[:, 0] = np.ones(number_of_surfaces)
    counts = np.zeros(number_of_surfaces)
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

    Y_1 = np.zeros((number_of_surfaces, number_of_surfaces))
    Y_2 = np.zeros((number_of_surfaces, number_of_surfaces))
    for s_1 in range(1, number_of_surfaces):
        for s_2 in range(s_1+1, number_of_surfaces):
            if (a := annotation_correspondence[s_1]) == (b := annotation_correspondence[s_2]) and a != 0:
                Y_1[s_1][s_2] = 1
            elif a > 0 or b > 0:
                Y_2[s_1][s_2] = 1

    Y = np.pad(np.stack([Y_1, Y_2, np.zeros_like(Y_1)], axis=0), ((0, 0), (0, num_boxes), (0, num_boxes)))
    for i in range(number_of_surfaces, number_of_surfaces + num_boxes):
        Y[2][i][i] = 1.0
    return np.asarray([Y[:, 1:, 1:]])

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(pairwise_potentials, Q_mult), matrix_1), axis=1)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-2)
    new_Q = tf.exp(unary_potentials - tf.multiply(compatibility, weight))
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_CRF(w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, weight, num_iterations=10):
    Q_in = layers.Input(shape=(None, None))
    #unary_in = layers.Input(shape=(None, None))
    num_labels = tf.shape(Q_in)[1]

    convexity_in = layers.Input(shape=(None, None))
    color_in = layers.Input(shape=(None, None))
    texture_in = layers.Input(shape=(None, None))
    coplanarity_in = layers.Input(shape=(None, None))
    boxes_in = layers.Input(shape=(None, None))
    neighborhood_in = layers.Input(shape=(None, None))

    boxes = (tf.sigmoid(layers.Add()([Variable2(w_7, name="w_7")(boxes_in), layers.Multiply()([Variable2(w_8, name="w_8")(boxes_in), boxes_in])])) - 0.5) * 2

    pairwise_potentials = layers.Add()([layers.Multiply()([-Variable2(w_1, name="w_1")(convexity_in), convexity_in]),
                                        #layers.Multiply()([Variable2(w_2, name="w_2")(color_in), color_in]),
                                        #layers.Multiply()([Variable2(w_3, name="w_3")(texture_in), texture_in]),
                                        #layers.Multiply()([-Variable2(w_4, name="w_4")(coplanarity_in), coplanarity_in]),
                                        #layers.Multiply()([-Variable2(w_6, name="w_6")(neighborhood_in), neighborhood_in]),
                                        Variable2(w_9, name="w_9")(neighborhood_in)])
    num_boxes = num_labels - tf.shape(pairwise_potentials)[1]
    pairwise_potentials = tf.pad(pairwise_potentials, [[0, 0], [0, num_boxes], [0, num_boxes]]) #+ layers.Multiply()([-Variable2(w_5, name="w_5")(boxes), boxes])

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)
    weight = Variable2(weight, name="weight")(Q_in)
    Q = Q_in
    for _ in range(num_iterations):
        Q = mean_field_iteration(Q_in, pairwise_potentials, Q, num_labels, matrix_1, matrix_2, weight)
    output = Q

    model = Model(inputs=[Q_in, convexity_in, color_in, texture_in, coplanarity_in, boxes_in, neighborhood_in], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-3), metrics=[], run_eagerly=True)
    return model

def custom_loss(y_actual, y_predicted):
    join_matrix = tf.squeeze(tf.gather(y_actual, [0], axis=1), axis=1)
    not_join_matrix = tf.squeeze(tf.gather(y_actual, [1], axis=1), axis=1)
    entropy_mask = tf.reduce_sum(tf.squeeze(tf.gather(y_actual, [2], axis=1), axis=1), axis=-1)

    num_labels = tf.shape(y_predicted)[2]

    prediction_expanded_1 = tf.tile(tf.expand_dims(y_predicted, axis=1), [1, num_labels, 1, 1])
    prediction_expanded_2 = tf.tile(tf.expand_dims(y_predicted, axis=2), [1, 1, num_labels, 1])
    mult_out = tf.multiply(prediction_expanded_1, prediction_expanded_2)
    sum_out = tf.reduce_sum(mult_out, axis=-1)

    positive_error = tf.reduce_sum(tf.reduce_sum(tf.multiply(join_matrix, sum_out), axis=-1),axis=-1)
    negative_error = tf.reduce_sum(tf.reduce_sum(tf.multiply(not_join_matrix, sum_out), axis=-1),axis=-1)
    entropy = -tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(y_predicted, tf.math.log(y_predicted)), axis=-1), entropy_mask), axis=-1)

    num_joins = tf.reduce_sum(tf.reduce_sum(join_matrix, axis=-1), axis=-1)
    num_not_joins = tf.reduce_sum(tf.reduce_sum(not_join_matrix, axis=-1), axis=-1)
    num_entropies = tf.reduce_sum(entropy_mask, axis=-1)

    error = -print_tensor(tf.math.divide_no_nan(positive_error, num_joins))\
            + print_tensor(tf.math.divide_no_nan(negative_error, num_not_joins))#\
            #+ 0.3 * entropy/num_entropies
    return error

def get_initial_probabilities(num_labels, bbox_overlap):
    num_boxes = np.shape(bbox_overlap)[0]
    potentials_surfaces = np.eye(num_labels, num_labels + num_boxes) * 1
    potentials_surfaces = potentials_surfaces + np.random.uniform(size=np.shape(potentials_surfaces))
    potentials_surfaces = potentials_surfaces / np.sum(potentials_surfaces, axis=-1, keepdims=True)

    potentials_boxes = bbox_overlap.copy()
    potentials_boxes = potentials_boxes + np.random.uniform(size=np.shape(potentials_boxes)) * 0.1
    potentials_boxes = potentials_boxes / np.sum(potentials_boxes, axis=-1, keepdims=True)
    potentials_boxes[np.isnan(potentials_boxes)] = 0
    potentials_boxes = np.pad(potentials_boxes, ((0, 0), (0, num_boxes)))

    potentials = np.concatenate([potentials_surfaces, potentials_boxes], axis=0)
    return potentials

def get_neighborhood_matrix(neighbors_list, number_of_surfaces):
    neighbor_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(len(neighbors_list)):
        for neighbor in neighbors_list[i]:
            neighbor_matrix[i][neighbor] = 1

    return neighbor_matrix

def calc_box_and_surface_overlap(bboxes, surfaces, number_of_surfaces):
    n_bboxes = np.shape(bboxes)[0]
    overlap_counter = np.zeros((n_bboxes, number_of_surfaces))

    for i in range(n_bboxes):
        box = np.round(bboxes[i]).astype("int32")
        for y in range(box[1], box[3]+1):
            for x in range(box[0], box[2]+1):
                overlap_counter[i][surfaces[y][x]] += 1

    counts = np.zeros(number_of_surfaces)
    for y in range(height):
        for x in range(width):
            counts[surfaces[y][x]] += 1

    for i in range(number_of_surfaces):
        if counts[i] == 0: counts[i] = 1

    overlap_counter = overlap_counter / np.expand_dims(counts, axis=0)
    return overlap_counter

def create_bbox_similarity_matrix(bbox_overlap_matrix):
    num_boxes, num_labels = np.shape(bbox_overlap_matrix)
    similarity_matrix = np.zeros((num_boxes + num_labels, num_boxes + num_labels))
    similarity_matrix[num_labels:, :num_labels] = bbox_overlap_matrix
    return similarity_matrix

def determine_occlusion_and_connection_infos(closest_points, surfaces, norm_image, number_of_surfaces):
    occlusion_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
    close_matrix = np.zeros((number_of_surfaces, number_of_surfaces))
    closest_points_array = np.zeros((number_of_surfaces, number_of_surfaces, 2), dtype="int32")

    index_1 = 0
    for i in range(number_of_surfaces):
        index_2 = 0
        for j in range(number_of_surfaces):
            closest_points_array[i][j] = closest_points[index_1][index_2]
            index_2 += 1
        index_1 += 1

    for i in range(number_of_surfaces):
        for j in range(i + 1, number_of_surfaces):
            p_1 = closest_points_array[i][j]
            p_2 = closest_points_array[j][i]

            dif = p_2 - p_1
            length = np.max(np.abs(dif))
            direction = dif / length

            positions = np.swapaxes(np.expand_dims(p_1, axis=-1) + np.ndarray.astype(np.round(np.expand_dims(direction, axis=-1) *
                                                                          (lambda x: np.stack([x, x]))(np.arange(1, length))), dtype="int32"),axis1=0, axis2=1)
            pos_1 = p_1.copy()
            pos_2 = p_2.copy()
            index_1 = 0
            index_2 = len(positions)
            index = 0
            other_index = False
            other_surface_counter = 0
            zero_counter = 0
            for p in positions:
                s = surfaces[p[1]][p[0]]
                if s == i:
                    index_1 = index
                    pos_1 = p.copy()
                elif s == j:
                    index_2 = index
                    pos_2 = p.copy()
                    break
                elif s != 0:
                    other_surface_counter += 1
                elif s == 0:
                    zero_counter += 1
                index += 1
            if zero_counter >= 45 or other_surface_counter >= 3:
                other_index = True
            if not other_index and abs(index_2 - index_1) < 8:
                close_matrix[i, j] = 1
                close_matrix[j, i] = 1
                continue

            d_1 = norm_image[pos_1[1]][pos_1[0]]
            d_2 = norm_image[pos_2[1]][pos_2[0]]
            d_dif = d_2 - d_1

            occluded = True

            if index_1 != index_2:
                index_dif = 1 / (index_2 - index_1) * d_dif

                for k in range(index_2 - index_1 - 1):
                    p = positions[index_1 + k + 1]
                    s = surfaces[p[1]][p[0]]
                    if norm_image[p[1]][p[0]] > k * index_dif + d_1 + 1 and s != 0:
                        occluded = False
                        break

            if occluded:
                occlusion_matrix[i, j] = 1
                occlusion_matrix[j, i] = 1
    return occlusion_matrix, close_matrix, closest_points_array

def get_position_and_occlusion_infos(nearest_points_func, positions, surface_patch_points, surfaces, norm_image, number_of_surfaces):
    input = assemble_objects.determine_occlusion_line_points(np.ones((number_of_surfaces, number_of_surfaces)), positions, surface_patch_points)
    nearest_points_for_occlusion = nearest_points_func(*input)
    join_matrix, close_matrix, closest_points = determine_occlusion_and_connection_infos(nearest_points_for_occlusion, surfaces, norm_image, number_of_surfaces)
    return join_matrix, close_matrix, closest_points

def calculate_depth_extend_distances(depth_extend, number_of_surfaces):
    distances = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(number_of_surfaces):
        for j in range(i+1, number_of_surfaces):
            d = assemble_objects.depth_extend_distance(i, j, depth_extend)
            distances[i, j] = d
            distances[j, i] = d
    return distances

def get_similarity_data_for_CRF(surfaces, depth_edges, rgb_image, lab_image, patches, models, normal_angles, points_in_space, depth_image):
    number_of_surfaces = int(np.max(surfaces) + 1)
    assemble_objects.set_number_of_surfaces(number_of_surfaces)
    color_similarity_model, texture_similarity_model, angle_similarity_model, texture_model, nearest_points_func, object_detector, _ = models

    average_positions, histogram_color, histogram_angles, histogram_texture, centroids, \
    average_normals, centroid_indices, surfaces, planes, surface_patch_points, neighbors, border_centers, norm_image, depth_extend \
        = assemble_objects.extract_information(rgb_image, texture_model, surfaces, patches, normal_angles, lab_image, depth_image,
                              points_in_space, depth_edges)

    similarities_color = color_similarity_model(histogram_color)
    similarities_texture = assemble_objects.texture_similarity_calc(histogram_texture)
    similarities_angle = angle_similarity_model(histogram_angles)/20

    planes = assemble_objects.find_even_planes(np.swapaxes(histogram_angles, 2, 0))
    coplanarity_matrix = assemble_objects.determine_coplanarity(np.ones((number_of_surfaces, number_of_surfaces)), centroids,
                                                         assemble_objects.angles_to_normals(average_normals).astype("float32"), planes, number_of_surfaces)

    convexity_matrix, concave, _ = assemble_objects.determine_convexly_connected_surfaces(nearest_points_func, surface_patch_points, neighbors, border_centers,
                                                                normal_angles, surfaces, points_in_space, coplanarity_matrix, norm_image, np.ones((number_of_surfaces, number_of_surfaces)))

    neighborhood_matrix = get_neighborhood_matrix(neighbors, number_of_surfaces)
    bboxes = object_detector(rgb_image)
    bbox_overlap_matrix = calc_box_and_surface_overlap(bboxes, surfaces, number_of_surfaces)
    #detect_objects.show_bounding_boxes(rgb_image, bboxes)
    #plot_surfaces(surfaces)
    #centroid_distances = np.sqrt(np.sum(np.square(np.tile(np.expand_dims(centroids, axis=0), [number_of_surfaces, 1, 1]) - np.tile(np.expand_dims(centroids, axis=1), [1, number_of_surfaces, 1])), axis=-1))/1000
    occlusion_matrix, close_matrix, closest_points = get_position_and_occlusion_infos(nearest_points_func, average_positions, surface_patch_points, surfaces, norm_image, number_of_surfaces)
    close_matrix[neighborhood_matrix == 1] = 0
    distances = np.sqrt(np.sum(np.square(closest_points - np.swapaxes(closest_points, axis1=0, axis2=1)), axis=-1))/500
    depth_extend_distances = calculate_depth_extend_distances(depth_extend, number_of_surfaces)/1000

    return bbox_overlap_matrix[:, 1:], similarities_texture[1:, 1:], similarities_color[1:, 1:], convexity_matrix[1:, 1:], coplanarity_matrix[1:, 1:],\
           neighborhood_matrix[1:, 1:], concave[1:, 1:], distances[1:, 1:], occlusion_matrix[1:, 1:], close_matrix[1:, 1:], depth_extend_distances[1:, 1:],\
           similarities_angle[1:, 1:]

def plot_prediction(prediction, surfaces):
    max_vals = np.argmax(prediction[0], axis=-1)
    for i in range(len(max_vals)):
        if max_vals[i] != i:
            surfaces[surfaces == i+1] = max_vals[i]+1
    assemble_objects.plot_surfaces(surfaces)

def assemble_objects_CRF(surfaces, depth_edges, rgb_image, lab_image, patches, models, normal_angles, points_in_space, depth_image, annotation, train=False, plot=False):
    bbox_overlap_matrix, similarities_texture, similarities_color, convexity_matrix, coplanarity_matrix, neighborhood_matrix, concave, centroid_distances =\
        get_similarity_data_for_CRF(surfaces, depth_edges, rgb_image, lab_image, patches, models, normal_angles, points_in_space, depth_image)
    bbox_similarities = create_bbox_similarity_matrix(bbox_overlap_matrix)

    CRF_model = models[-1]

    num_labels = int(np.max(surfaces))
    num_boxes = np.shape(bbox_overlap_matrix)[0]
    Q_in = get_initial_probabilities(num_labels, bbox_overlap_matrix)
    unary_in = Q_in.copy()
    unary_in[num_labels:] = -1000

    input = [np.asarray([e]) for e in [Q_in, convexity_matrix, similarities_color, similarities_texture,
                                                              coplanarity_matrix, bbox_similarities, neighborhood_matrix]]

    if not train:
        prediction = CRF_model.predict(input)
        plot_prediction(prediction, surfaces)
    else:
        Y = get_Y_value(annotation, surfaces, num_labels+1, num_boxes)
        #s, l = assemble_objects.join_surfaces_according_to_join_matrix(join_matrix, surfaces, num_labels+1)
        #assemble_objects.plot_surfaces(s)
        CRF_model.fit(input, Y, epochs=700)
        print_parameters(CRF_model)
        prediction = CRF_model.predict(input)
        plot_prediction(prediction, surfaces)

def get_GPU_models():
    return assemble_objects.chi_squared_distances_model((10, 10), (4, 4)), \
           assemble_objects.chi_squared_distances_model_1D(), \
           assemble_objects.chi_squared_distances_model((4, 4), (1, 1)),\
           assemble_objects.extract_texture_function(), \
           assemble_objects.calculate_nearest_points,\
           detect_objects.get_object_detector(),\
           mean_field_CRF(*initial_guess_parameters)

def main():
    models = get_GPU_models()
    #for index in list(range(0, 111)):
    train = True
    index = 4
    while True:
        print(index)
        depth, rgb, annotation = load_image(index)
        lab = rgb_to_Lab(rgb)
        Q = np.load(f"out/{index}/Q.npy")
        depth_image = np.load(f"out/{index}/depth.npy")
        angles = np.load(f"out/{index}/angles.npy")
        patches = np.load(f"out/{index}/patches.npy")
        points_in_space = np.load(f"out/{index}/points.npy")
        depth_edges = np.load(f"out/{index}/edges.npy")
        Q = np.argmax(Q, axis=-1)
        assemble_objects_CRF(Q, depth_edges, rgb, lab, patches, models, angles, points_in_space, depth_image, annotation, train)
        train = False
        index += 1
    quit()

if __name__ == '__main__':
    #create_training_set()
    train_unary_classifier()
    #quit()
    #assemble_objects_with_unary_classifier()
    quit()
    train_unary_classifier()
    quit()
    main()