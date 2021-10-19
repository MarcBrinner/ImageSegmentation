import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model, losses, optimizers, regularizers
import standard_values
from find_planes import plot_surfaces
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from assemble_objects_CRF import get_GPU_models, calculate_pairwise_similarity_features_for_surfaces, get_Y_value,\
    create_similarity_feature_matrix, plot_prediction, Variable2, custom_loss, print_parameters, print_tensor
from load_images import load_image_and_surface_information
from assemble_objects_rules import join_surfaces_according_to_join_matrix

model_types = ["LR", "Tree", "Forest", "Neural", "Ensemble"]
num_features = 14

def create_training_set_for_specific_clf(clf_type="Forest"):
    features = pickle.load(open("data/train_object_assemble_CRF_dataset/features.pkl", "rb"))
    clf = load_clf(clf_type)

    results = []
    for feature_matrix in features:
        predictions = clf.predict_proba(np.reshape(feature_matrix, (-1, np.shape(feature_matrix)[-1])))[:, 1]
        predictions = np.reshape(predictions, (np.shape(feature_matrix)[0], - 1))
        results.append(predictions)
    pickle.dump(results, open(f"data/train_object_assemble_CRF_dataset/potentials_{clf_type}.pkl", "wb"))

def mean_field_iteration(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(pairwise_potentials, axis=-1), Q_mult), matrix_1), axis=2)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-1)
    compatibility = compatibility - tf.reduce_min(compatibility, axis=-1, keepdims=True)
    compatibility = tf.math.divide_no_nan(compatibility, tf.reduce_max(compatibility)) * 15
    new_Q = tf.exp(tf.multiply(unary_potentials, 0.5) - compatibility)
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_iteration_2(unary_potentials, pairwise_potentials, Q, num_labels, matrix_1, matrix_2):
    Q_mult = tf.tile(tf.expand_dims(Q, axis=1), [1, num_labels, 1, 1])
    messages = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(pairwise_potentials, axis=-1), Q_mult), matrix_1), axis=2)
    compatibility = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, num_labels, 1]), matrix_2), axis=-1)
    compatibility = compatibility - tf.reduce_min(compatibility, axis=-1, keepdims=True)
    compatibility = tf.math.divide_no_nan(compatibility, tf.reduce_max(compatibility)) * 10
    return compatibility
    new_Q = tf.exp(tf.multiply(unary_potentials, 0.5) - tf.multiply(compatibility, weight))
    new_Q = tf.math.divide_no_nan(new_Q, tf.reduce_sum(new_Q, axis=-1, keepdims=True))
    return new_Q

def mean_field_CRF(num_iterations=60):
    Q_in = layers.Input(shape=(None, None), name="Q")
    pairwise_potentials_in = layers.Input(shape=(None, None), name="potentials")
    num_labels = tf.shape(Q_in)[-1]
    pairwise_potentials = layers.LeakyReLU(alpha=0.5)(pairwise_potentials_in)

    #pairwise_potentials = pairwise_potentials_in * Variable2(pairwise_scale, name="pw_scale")(pairwise_potentials_in)
    #pairwise_potentials = pairwise_potentials - Variable2(pairwise_subtract, name="pw_sub")(pairwise_potentials)

    matrix_1 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=-1)
    matrix_2 = tf.ones((1, num_labels, num_labels, num_labels)) - tf.expand_dims(tf.expand_dims(tf.eye(num_labels, num_labels), axis=0), axis=0)

    Q = Q_in
    Q_acc = tf.expand_dims(Q, axis=1)
    for _ in range(num_iterations):
        Q = 0.9 * Q + 0.1 * mean_field_iteration(Q_in, pairwise_potentials, Q, num_labels, matrix_1, matrix_2)
        Q_acc = tf.concat([Q_acc, tf.expand_dims(Q, axis=1)], axis=1)

    model = Model(inputs=[Q_in, pairwise_potentials_in], outputs=Q_acc)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=3e-2), metrics=[], run_eagerly=True)
    model.run_eagerly = True
    return model

def create_training_set():
    models = get_GPU_models()
    inputs = []
    labels = []
    for set_type in ["train", "test"]:
        for index in (standard_values.train_indices if set_type == "train" else standard_values.test_indices):
            print(index)
            data = load_image_and_surface_information(index)
            calculate_pairwise_similarity_features_for_surfaces(data, models)

            Y = get_Y_value(data)[0]
            join_matrix = Y[0][:-data["num_bboxes"], :-data["num_bboxes"]]
            not_join_matrix = Y[1][:-data["num_bboxes"], :-data["num_bboxes"]]

            feature_matrix = create_similarity_feature_matrix(data)
            index_matrix = (join_matrix + not_join_matrix) == 1
            num_indices = np.sum(index_matrix)

            labels = labels + list(np.ndarray.flatten(join_matrix[index_matrix]))
            inputs = inputs + list(np.reshape(feature_matrix[index_matrix], (num_indices, np.shape(feature_matrix)[-1])))

        np.save(f"data/assemble_objects_with_clf_datasets/{set_type}_inputs.npy", np.asarray(inputs))
        np.save(f"data/assemble_objects_with_clf_datasets/{set_type}_labels.npy", np.asarray(labels))
        inputs = []
        labels = []

def get_nn_clf(num_features=num_features):
    input = layers.Input(shape=(num_features,))
    d_1 = layers.Dropout(0.1)(layers.Dense(100, kernel_regularizer=regularizers.l2(0.00001), activation="relu")(input))
    d_2 = layers.Dropout(0.1)(layers.Dense(100, kernel_regularizer=regularizers.l2(0.00001), activation="relu")(d_1))
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0001))(d_2)
    model = Model(inputs=input, outputs=tf.squeeze(out, axis=-1))
    model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam(learning_rate=5e-4))
    return model

def train_unary_classifier(type="LR"):
    if type not in model_types:
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
    if model_type not in model_types:
        print("Invalid model type.")
        quit()
    clf = load_clf(model_type)
    inputs_train = np.load("data/assemble_objects_with_clf_datasets/train_inputs.npy")
    labels_train = np.load("data/assemble_objects_with_clf_datasets/train_labels.npy")
    inputs_test = np.load("data/assemble_objects_with_clf_datasets/test_inputs.npy")
    labels_test = np.load("data/assemble_objects_with_clf_datasets/test_labels.npy")
    print(np.sum(np.abs(np.round(clf.predict(inputs_train))-labels_train))/len(labels_train))
    print(np.sum(np.abs(np.round(clf.predict(inputs_test))-labels_test))/len(labels_test))

def load_clf(type_str):
    if type_str == "Neural":
        clf_neural = get_nn_clf()
        clf_neural.load_weights("parameters/pairwise_surface_clf/clf_Neural.ckpt")
        clf = type('', (), {})()
        clf.predict = lambda x: np.round(clf_neural.predict(x))
        clf.predict_proba = lambda x: (lambda y: np.stack([1-y, y], axis=-1))(clf_neural.predict(x))
    elif type_str == "Ensemble":
        clfs = [load_clf(t) for t in model_types if t != "Ensemble"]
        clf = type('', (), {})()
        clf.predict_proba = lambda x: np.sum(np.asarray([clf.predict_proba(x) for clf in clfs]), axis=0) / len(clfs)
        clf.predict = lambda x: np.round(np.sum(np.asarray([clf.predict_proba(x)[:, 1] for clf in clfs]), axis=0) / len(clfs))
    else:
        clf = pickle.load(open(f"parameters/pairwise_surface_clf/clf_{type_str}.pkl", "rb"))
    return clf

def assemble_objects_with_unary_classifier(type="LR"):
    if type not in model_types:
        print("Invalid model type.")
        quit()
    clf = load_clf(type)
    models = get_GPU_models()
    for index in standard_values.test_indices[3:]:
        data = load_image_and_surface_information(index)
        calculate_pairwise_similarity_features_for_surfaces(data, models)
        feature_matrix = create_similarity_feature_matrix(data)

        predictions = clf.predict(np.reshape(feature_matrix, (-1, np.shape(feature_matrix)[-1])))
        join_matrix = np.zeros((data["num_surfaces"], data["num_surfaces"]))
        join_matrix[1:, 1:] = np.reshape(predictions, (data["num_surfaces"]-1, data["num_surfaces"]-1))

        new_surfaces, _ = join_surfaces_according_to_join_matrix(join_matrix, data["surfaces"], data["num_surfaces"])
        plot_surfaces(new_surfaces)

def get_initial_probabilities(pairwise_potentials):
    initial_probabilities = pairwise_potentials.copy()
    num_labels = np.shape(pairwise_potentials)[0]
    initial_probabilities[np.eye(num_labels) == 1] = 1
    initial_probabilities = initial_probabilities / np.sum(initial_probabilities, axis=-1, keepdims=True)
    return initial_probabilities

def assemble_objects_crf(clf_type):
    if clf_type not in model_types:
        print("Invalid model type.")
        quit()
    clf = load_clf(clf_type)
    models = get_GPU_models()
    crf = mean_field_CRF(60)
    for index in standard_values.test_indices[3:]:
        data = load_image_and_surface_information(index)
        plot_surfaces(data["surfaces"])
        calculate_pairwise_similarity_features_for_surfaces(data, models)
        feature_matrix = create_similarity_feature_matrix(data)

        predictions = np.reshape(clf.predict_proba(np.reshape(feature_matrix, (-1, np.shape(feature_matrix)[-1])))[:, 1], (data["num_surfaces"]-1, data["num_surfaces"]-1))
        #predictions[predictions < 0.5] = 0
        initial_probabilities = get_initial_probabilities(predictions)

        crf_out = crf.predict([np.asarray([x]) for x in [initial_probabilities, predictions-0.5]])

        plot_prediction(crf_out[0, -1], data["surfaces"])
        #plot_surfaces(data["surfaces"])

def train_CRF(clf_type="Forest"):
    potentials = pickle.load(open(f"data/train_object_assemble_CRF_dataset/potentials_{clf_type}.pkl", "rb"))
    Y = pickle.load(open("data/train_object_assemble_CRF_dataset/Y.pkl", "rb"))

    def gen():
        for i in range(len(standard_values.train_indices)):
            num_labels = np.shape(potentials[i])[-1]
            yield {"Q": get_initial_probabilities(potentials[i]), "potentials": potentials[i]}, tf.squeeze(Y[i], axis=0)[:, :num_labels, :num_labels]

    dataset = tf.data.Dataset.from_generator(gen, output_types=({"Q": tf.float32, "potentials": tf.float32}, tf.float32))
    # p = assemble_objects_pairwise_clf.load_clf("LR")
    # CRF = mean_field_CRF_test(np.reshape(p.coef_, (1, 12)), np.asarray([[p.intercept_]]), np.asarray([[0.1]]), 1)
    # print(CRF.evaluate(dataset.batch(1), verbose=1))
    #
    # CRF = mean_field_CRF_test(np.ones((1, 12)), np.asarray([[0]]), np.asarray([[0.1]]), 1)
    # print(CRF.evaluate(dataset.batch(1), verbose=1))
    # print_parameters(CRF, ["feature_weight", "LR_bias", "weight_1", "weight_2"])
    CRF = mean_field_CRF()
    CRF.fit(dataset.batch(1), verbose=1, epochs=20)
    print_parameters(CRF, ["pw_scale", "pw_sub", "positive_compatibility"])

    quit()
    for x, y in dataset:
        h = CRF.predict([np.asarray([k]) for k in x.values()], batch_size=1)
        print()
        quit()
    quit()

def assemble_objects_hierarchical(model_type, join_threshold = 0.5, join_prevention_threshold = 0.15):
    if model_type not in model_types:
        print("Invalid model type.")
        quit()
    clf = load_clf(model_type)
    models = get_GPU_models()
    for index in standard_values.test_indices[3:]:
        data = load_image_and_surface_information(index)
        #plot_surfaces(data["surfaces"])
        calculate_pairwise_similarity_features_for_surfaces(data, models)
        feature_matrix = create_similarity_feature_matrix(data)

        predictions = np.pad(np.reshape(clf.predict_proba(np.reshape(feature_matrix, (-1, np.shape(feature_matrix)[-1])))[:, 1], (-1, data["num_surfaces"]-1)), [[1, 0], [1, 0]])
        join_pairs = {}
        for i in range(1, data["num_surfaces"]):
            for j in range(i+1, data["num_surfaces"]):
                join_probability = predictions[i, j]
                if join_probability > join_threshold:
                    join_pairs[(i, j)] = join_probability

        not_join_matrix = np.zeros_like(predictions)
        not_join_matrix[predictions < join_prevention_threshold] = 1

        join_matrix = np.eye(data["num_surfaces"])
        index_list = np.arange(data["num_surfaces"])
        for (i, j), value in sorted(join_pairs.items(), key=lambda x: x[1], reverse=True):
            if join_matrix[i, j] == 1:
                continue

            indices_1 = list(index_list[join_matrix[i] == 1])
            indices_2 = list(index_list[join_matrix[j] == 1])

            join = True
            for index_1 in indices_1:
                for index_2 in indices_2:
                    if not_join_matrix[index_1, index_2] == 1:
                        join = False
                        break
                if not join:
                    break
            if join:
                for index_1 in indices_1:
                    for index_2 in indices_2:
                        join_matrix[index_1, index_2] = 1
        s, _ = join_surfaces_according_to_join_matrix(join_matrix, data["surfaces"], data["num_surfaces"])
        plot_surfaces(s)

if __name__ == '__main__':
    #create_training_set_for_specific_clf()
    #quit()
    #train_CRF()
    #quit()
    #evaluate_clf("Forest")
    #quit()
    #train_unary_classifier("Neural")
    #quit()
    #assemble_objects_with_unary_classifier("Forest")
    assemble_objects_crf("Forest")