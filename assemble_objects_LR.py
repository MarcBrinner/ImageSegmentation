import numpy as np
import pickle
from find_planes import plot_surfaces
from sklearn.linear_model import LogisticRegression
from assemble_objects_CRF import get_GPU_models
from load_images import load_image_and_surface_information

def create_training_set():
    models = get_GPU_models()
    inputs = []
    labels = []
    for index in range(111):
        print(index)
        data = load_image_and_surface_information(index)
        info = get_similarity_data_for_CRF(data)
        num_boxes = np.shape(info[0])[0]
        Y = get_Y_value(annotation, Q, num_surfaces, num_boxes)[0]
        join_matrix = Y[0]
        not_join_matrix = Y[1]

        for i in range(num_surfaces-1):
            for j in range(num_surfaces-1):
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

def quadratic_feature_expansion(data):
    num_samples, num_features = np.shape(data)
    exp_1 = np.tile(np.expand_dims(data, axis=1), [1, num_features, 1])
    exp_2 = np.tile(np.expand_dims(data, axis=2), [1, 1, num_features])
    mult = np.multiply(exp_1, exp_2)
    reshaped = np.reshape(mult, (num_samples, num_features**2))
    return np.concatenate([data, reshaped], axis=1)

def train_unary_classifier():
    inputs_train = np.load("data/train_in.npy")
    labels_train = np.load("data/train_labels.npy")
    inputs_test = np.load("data/test_in.npy")
    labels_test = np.load("data/test_labels.npy")
    clf = LogisticRegression(max_iter=1000, penalty="l2")
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
        num_surfaces = int(np.max(Q) + 1)
        input_indices = []
        inputs = []

        for i in range(num_surfaces - 1):
            for j in range(num_surfaces - 1):
                inputs.append(np.asarray([np.sum(info[0][:, i] * info[0][:, j]), *[info[k][i, j] for k in range(1, 8)]]))
                input_indices.append((i, j))
        predictions = clf.predict(inputs)
        join_matrix = np.zeros((num_surfaces, num_surfaces))
        for i in range(len(predictions)):
            if predictions[i] == 1:
                indices = input_indices[i]
                join_matrix[indices[0]+1, indices[1]+1] = 1
                join_matrix[indices[1]+1, indices[0]+1] = 1
        s, r = assemble_objects.join_surfaces_according_to_join_matrix(join_matrix, Q, num_surfaces)
        plot_surfaces(s)