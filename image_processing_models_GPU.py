import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from standard_values import *
from tensorflow.keras import layers, Model, initializers, optimizers, regularizers

variable_names = ["w_1", "w_2", "w_3", "theta_1_1", "theta_1_2", "theta_2_1", "theta_2_2", "theta_2_3", "theta_3_1", "theta_3_2", "theta_3_3", "weight"]

def print_parameters(model):
    for name in variable_names:
        print(f"{name}: {model.get_layer(name).weights}")

def save_parameters(model, index):
    array = np.zeros(len(variable_names), dtype="float64")
    for i in range(len(variable_names)):
        array[i] = model.get_layer(variable_names[i]).weights[0].numpy()
    print(array)
    np.save(f"parameters/{index}.npy", array)

def load_parameters(index):
    if index < 0:
        return list(np.reshape([0.8, 0.8, 1.5, 1 / 80, 25000.0, 1 / 100, 25000.0, 1 / 200, 1 / 100, 25000.0, 20.0, 0.01], (12, 1, 1)))
    return list(np.reshape(np.load(f"parameters/{index}.npy"), (12, 1, 1)))

class Variable(layers.Layer):
    def __init__(self, initial_value, **kwargs):
        self.initial_value = initial_value
        self.output_shape2 = np.shape(initial_value)
        super(Variable, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='weights',
                                      shape=self.output_shape2, trainable=True,
                                      initializer=initializers.constant(self.initial_value), dtype=tf.float32)
        super(Variable, self).build(input_shape)

    def call(self, input_data):
        return tf.broadcast_to(self.kernel, (tf.shape(input_data)[0], self.output_shape2[0], self.output_shape2[1]))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_shape2[0]

class Variable2(layers.Layer):
    def __init__(self, initial_value, **kwargs):
        self.initial_value = initial_value
        self.output_shape2 = np.shape(initial_value)
        super(Variable2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='weights',
                                      shape=self.output_shape2, trainable=True, initializer=initializers.constant(self.initial_value), dtype=tf.float32)
        super(Variable2, self).build(input_shape)

    def call(self, input_data):
        return tf.broadcast_to(self.kernel, tf.shape(input_data))

    def compute_output_shape(self, input_shape):
        return input_shape

class Print_Tensor(layers.Layer):
    def __init__(self, **kwargs):
        super(Print_Tensor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Print_Tensor, self).build(input_shape)

    def call(self, input_data):
        return input_data

    def compute_output_shape(self, input_shape):
        return input_shape

def mean_field_update_model_learned(number_of_pixels, number_of_surfaces, Q, features_1, features_2, features_3, matrix,
                            w_1, w_2, w_3, theta_1_1, theta_1_2, theta_2_1, theta_2_2, theta_2_3, theta_3_1, theta_3_2, theta_3_3, weight, batch_size):
    Q = tf.tile(tf.expand_dims(tf.constant(Q, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    matrix =  tf.tile(tf.expand_dims(tf.constant(matrix, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    features_1 = tf.tile(tf.expand_dims(tf.constant(features_1, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_2 = tf.tile(tf.expand_dims(tf.constant(features_2, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_3 = tf.tile(tf.expand_dims(tf.constant(features_3, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    feature_1 = layers.Input(shape=(3,), batch_size=batch_size, dtype=tf.float32)
    feature_2 = layers.Input(shape=(6,), batch_size=batch_size, dtype=tf.float32)
    feature_3 = layers.Input(shape=(5,), batch_size=batch_size, dtype=tf.float32)

    theta_1_1 = tf.repeat(Variable(theta_1_1, name="theta_1_1")(feature_1), repeats=[2], axis=-1)
    theta_1_2 = tf.repeat(Variable(theta_1_2, name="theta_1_2")(feature_1), repeats=[1], axis=-1)
    theta_2_1 = tf.repeat(Variable(theta_2_1, name="theta_2_1")(feature_1), repeats=[2], axis=-1)
    theta_2_2 = tf.repeat(Variable(theta_2_2, name="theta_2_2")(feature_1), repeats=[1], axis=-1)
    theta_2_3 = tf.repeat(Variable(theta_2_3, name="theta_2_3")(feature_1), repeats=[3], axis=-1)
    theta_3_1 = tf.repeat(Variable(theta_3_1, name="theta_3_1")(feature_1), repeats=[2], axis=-1)
    theta_3_2 = tf.repeat(Variable(theta_3_2, name="theta_3_2")(feature_1), repeats=[1], axis=-1)
    theta_3_3 = tf.repeat(Variable(theta_3_3, name="theta_3_3")(feature_1), repeats=[2], axis=-1)

    theta_1 = tf.concat([theta_1_1, theta_1_2], axis=-1, name="Theta_1")
    theta_2 = tf.concat([theta_2_1, theta_2_2, theta_2_3], axis=-1, name="Theta_2")
    theta_3 = tf.concat([theta_3_1, theta_3_2, theta_3_3], axis=-1, name="Theta_3")

    unary_potentials = layers.Input(shape=(number_of_surfaces,), batch_size=batch_size, dtype=tf.float32)

    feature_1_extended = tf.tile(tf.expand_dims(feature_1, axis=1), np.asarray([1, number_of_pixels, 1]))
    differences_1 = tf.subtract(feature_1_extended, features_1)
    similarities_1 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_1), tf.broadcast_to(theta_1, tf.shape(differences_1))), axis=-1))

    feature_2_extended = tf.tile(tf.expand_dims(feature_2, axis=1), np.asarray([1, number_of_pixels, 1]))
    differences_2 = tf.subtract(feature_2_extended, features_2)
    similarities_2 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_2), tf.broadcast_to(theta_2, tf.shape(differences_2))), axis=-1))

    feature_3_extended = tf.tile(tf.expand_dims(feature_3, axis=1), np.asarray([1, number_of_pixels, 1]))
    differences_3 = tf.subtract(feature_3_extended, features_3)
    similarities_3 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_3), tf.broadcast_to(theta_3, tf.shape(differences_3))), axis=-1))

    similarity_sum = layers.Add()([layers.Multiply()([Variable2(w_1, name="w_1")(similarities_1), similarities_1]),
                                 layers.Multiply()([Variable2(w_2, name="w_2")(similarities_2), similarities_2]),
                                 layers.Multiply()([Variable2(w_3, name="w_3")(similarities_3), similarities_3])])

    messages = tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(Q)), Q), axis=1)
    messages = layers.Multiply()([Variable2(weight, name="weight")(messages), messages])

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=1), [1, number_of_surfaces, 1])), axis=-1)
    #p = Print_Tensor(dtype=tf.float32)(compatibility_values)
    add = layers.Add(activity_regularizer=regularizers.l2(0.0001))([unary_potentials, compatibility_values])

    #o = Print_Tensor()(layers.Softmax()(-add))
    output = layers.Softmax()(-add)

    model = Model(inputs=[feature_1, feature_3, feature_4, unary_potentials], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[],
                  run_eagerly=False)
    return model


def update_step(Q, similarities, matrix, weight, unary_potentials):
    messages = tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarities, axis=-1), tf.shape(Q)), Q), axis=1)
    messages = layers.Multiply()([Variable2(weight, name="weight")(messages), messages])

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=1), [1, number_of_surfaces, 1])), axis=-1)
    add = layers.Add(activity_regularizer=regularizers.l2(0.001))([unary_potentials, compatibility_values])

    Q_new = layers.Softmax()(-add)
    return Q_new

def mean_field_update_assemble_surfaces(number_of_surfaces, sigma, weight, matrix):

    similarities = layers.Input(shape=(number_of_surfaces, number_of_surfaces))
    unary_potentials = layers.Input(shape=(number_of_surfaces, number_of_surfaces))
    Q = layers.Input(shape=(number_of_surfaces, number_of_surfaces))

    sigma = Variable(sigma)(similarities)
    weight = Variable(weight)(similarities)

    similarities = tf.multiply(tf.square(similarities), tf.broadcast_to(sigma, tf.shape(similarities)))

    for _ in range(5):
        Q = update_step(Q, similarities, matrix, weight, unary_potentials)

    output = Q
    model = Model(inputs=[similarities, unary_potentials, Q], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[],
                  run_eagerly=False)
    return model

def conv_crf(w_1, w_2, w_3, theta_1_1, theta_1_2, theta_2_1, theta_2_2, theta_2_3, theta_3_1, theta_3_2, theta_3_3, weight, kernel_size, height, width):
    k = kernel_size*2+1
    Q = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, None), dtype=tf.float32)
    number_of_surfaces = tf.shape(Q)[-1]
    unary_potentials = layers.Input(shape=(height, width, None), dtype=tf.float32)

    matrix = tf.tile(tf.reshape(tf.ones((number_of_surfaces, number_of_surfaces)) - tf.eye(number_of_surfaces, number_of_surfaces),
                               (1, 1, 1, number_of_surfaces, number_of_surfaces)), [1, height, width, 1, 1])

    mask_np = np.ones((1, 1, 1, k, k, 1))
    mask_np[0][0][0][kernel_size][kernel_size][0] = 0
    mask = tf.constant(mask_np, dtype=tf.float32)

    features_1 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 3), dtype=tf.float32)
    features_2 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 6), dtype=tf.float32)
    features_3 = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, 5), dtype=tf.float32)

    windows_Q = tf.reshape(tf.image.extract_patches(Q, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, number_of_surfaces))
    windows_Q = tf.multiply(windows_Q, tf.broadcast_to(mask, tf.shape(windows_Q)))
    windows_f_1 = tf.reshape(tf.image.extract_patches(features_1, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 3))
    windows_f_2 = tf.reshape(tf.image.extract_patches(features_2, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 6))
    windows_f_3 = tf.reshape(tf.image.extract_patches(features_3, [1, k, k, 1], padding='VALID', strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (1, height, width, k, k, 5))

    feature_1 = tf.tile(tf.gather(tf.gather(windows_f_1, [kernel_size], axis=3), [kernel_size], axis=4), [1, 1, 1, k, k, 1])
    feature_2 = tf.tile(tf.gather(tf.gather(windows_f_2, [kernel_size], axis=3), [kernel_size], axis=4), [1, 1, 1, k, k, 1])
    feature_3 = tf.tile(tf.gather(tf.gather(windows_f_3, [kernel_size], axis=3), [kernel_size], axis=4), [1, 1, 1, k, k, 1])

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

    differences_1 = tf.subtract(windows_f_1, feature_1)
    similarities_1 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_1), tf.broadcast_to(theta_1, tf.shape(differences_1))), axis=-1))

    differences_2 = tf.subtract(windows_f_2, feature_2)
    similarities_2 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_2), tf.broadcast_to(theta_2, tf.shape(differences_2))), axis=-1))

    differences_3 = tf.subtract(windows_f_3, feature_3)
    similarities_3 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_3), tf.broadcast_to(theta_3, tf.shape(differences_3))), axis=-1))

    similarity_sum = layers.Add()([layers.Multiply()([Variable2(w_1, name="w_1")(similarities_1), similarities_1]),
                                   layers.Multiply()([Variable2(w_2, name="w_2")(similarities_2), similarities_2]),
                                   layers.Multiply()([Variable2(w_3, name="w_3")(similarities_3), similarities_3])])

    messages = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(windows_Q)), windows_Q), axis=-2), axis=-2)
    messages = layers.Multiply()([Variable2(weight, name="weight")(messages), messages])

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, 1, number_of_surfaces, 1])), axis=-1)
    add = layers.Add(activity_regularizer=regularizers.l2(0.0001))([unary_potentials, compatibility_values])

    output = layers.Softmax()(-add)

    model = Model(inputs=[unary_potentials, Q, features_1, features_2, features_3], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[], run_eagerly=False)
    return model

def conv_crf_depth(w_1, w_2, w_3, theta_1_1, theta_1_2, theta_2_1, theta_2_2, theta_2_3, theta_3_1, theta_3_2, theta_3_3, weight, kernel_size, height, width):
    k = kernel_size*2+1
    Q = layers.Input(shape=(height+2*kernel_size, width+2*kernel_size, None), dtype=tf.float32)
    number_of_surfaces = tf.shape(Q)[-1]
    unary_potentials = layers.Input(shape=(height, width, None), dtype=tf.float32)

    matrix = tf.tile(tf.reshape(tf.ones((number_of_surfaces, number_of_surfaces)) - tf.eye(number_of_surfaces, number_of_surfaces),
                               (1, 1, 1, number_of_surfaces, number_of_surfaces)), [1, height, width, 1, 1])

    mask_np = np.ones((1, 1, 1, k, k, 1))
    mask_np[0][0][0][kernel_size][kernel_size][0] = 0
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

    windows_f_1 = tf.math.divide_no_nan(windows_f_1, div_1)
    windows_f_2 = tf.math.divide_no_nan(windows_f_2, div_2)
    windows_f_3 = tf.math.divide_no_nan(windows_f_3, div_3)

    feature_1 = tf.math.divide_no_nan(feature_1, div_1)
    feature_2 = tf.math.divide_no_nan(feature_2, div_2)
    feature_3 = tf.math.divide_no_nan(feature_3, div_3)

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

    differences_1 = tf.subtract(windows_f_1, feature_1)
    similarities_1 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_1), tf.broadcast_to(theta_1, tf.shape(differences_1))), axis=-1))

    differences_2 = tf.subtract(windows_f_2, feature_2)
    similarities_2 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_2), tf.broadcast_to(theta_2, tf.shape(differences_2))), axis=-1))

    differences_3 = tf.subtract(windows_f_3, feature_3)
    similarities_3 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_3), tf.broadcast_to(theta_3, tf.shape(differences_3))), axis=-1))

    similarity_sum = layers.Add()([layers.Multiply()([Variable2(w_1, name="w_1")(similarities_1), similarities_1]),
                                   layers.Multiply()([Variable2(w_2, name="w_2")(similarities_2), similarities_2]),
                                   layers.Multiply()([Variable2(w_3, name="w_3")(similarities_3), similarities_3])])

    messages = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(windows_Q)), windows_Q), axis=-2), axis=-2)
    messages = layers.Multiply()([Variable2(weight, name="weight")(messages), messages])

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=-2), [1, 1, 1, number_of_surfaces, 1])), axis=-1)
    add = layers.Add(activity_regularizer=regularizers.l2(0.0001))([unary_potentials, compatibility_values])

    output = layers.Softmax()(-add)

    model = Model(inputs=[unary_potentials, Q, features_1, features_2, features_3], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[], run_eagerly=False)
    return model

def custom_loss(y_actual, y_predicted):
    only_wrong_ones = tf.multiply(y_actual, y_predicted)
    error = tf.reduce_sum(only_wrong_ones, axis=-1)
    #print(y_predicted)
    #print(y_predicted)
    print(tf.reduce_min(error))
    return 100*error

def angle_calculator(vec):
    mult_1 = tf.constant(np.asarray([0.0, 1.0, 1.0]), dtype=tf.float32)
    mult_2 = tf.constant(np.asarray([1.0, 0.0, 1.0]), dtype=tf.float32)

    new_vec_1 = tf.multiply(vec, mult_1)
    new_vec_2 = tf.multiply(vec, mult_2)

    angle_1 = tf.acos(tf.clip_by_value(tf.math.divide_no_nan(-new_vec_1[2], tf.linalg.norm(new_vec_1)), -1, 1)) * tf.sign(new_vec_1[1])
    angle_2 = tf.acos(tf.clip_by_value(tf.math.divide_no_nan(-new_vec_2[2], tf.linalg.norm(new_vec_2)), -1, 1)) * -tf.sign(new_vec_2[0])
    return tf.stack([angle_1, angle_2], axis=0)

class Components(layers.Layer):
    def call(self, input):
        return tfa.image.connected_components(input)

class Counts(layers.Layer):
    def call(self, input):
        return tf.math.bincount(input)

class Calc_Angles(layers.Layer):
    def call(self, input):
        return tf.vectorized_map(lambda x: tf.vectorized_map(angle_calculator, x), input)

class Depth_Cutoff(layers.Layer):
    def call(self, input, middle):
        return tf.vectorized_map(lambda x: tf.vectorized_map(lambda y: tf.where(tf.greater(tf.abs(tf.subtract(y, y[middle])), 0.3), tf.zeros_like(y), tf.ones_like(y)), x), input)

class Gauss_Weights(layers.Layer):
    def call(self, input, middle, sigma, mask):
        return tf.vectorized_map(lambda x: tf.vectorized_map(lambda y: tf.multiply(tf.exp(-tf.divide(tf.square(tf.subtract(y, y[middle])), sigma)), mask), x), input)

class Depth_Factors(layers.Layer):
    def call(self, input, factors, middle):
        return tf.vectorized_map(lambda x:
                                        tf.vectorized_map(
                                             lambda y: (lambda z: tf.math.divide_no_nan(z*factors, tf.reduce_sum(z, axis=-1, keepdims=True)))(y),
                                             x),
                                 input)


class Remove_Small_Patches(layers.Layer):
    def call(self, image, counts, threshold):
        return tf.vectorized_map(lambda x: tf.vectorized_map(lambda y: tf.where(tf.greater(tf.gather(counts, [y]), threshold), 0, 1), x), image)

def uniform_filter_with_depth_cutoff(image, depth_image, window, middle, dimension, vals_per_window):
    patches = tf.reshape(tf.image.extract_patches(image, window, padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (height, width, vals_per_window, dimension))
    depth_patches = tf.squeeze(tf.image.extract_patches(depth_image, window, padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), axis=0)
    mult_values = Depth_Cutoff()(depth_patches, middle)
    sums = tf.reduce_sum(mult_values, axis=-1, keepdims=True)
    val_sums = tf.reduce_sum(tf.multiply(tf.expand_dims(mult_values, axis=-1), patches), axis=-2)
    return tf.math.divide_no_nan(val_sums, sums)

def gaussian_filter_with_depth_factor_model_GPU(size=3, height=height, width=width, sigma_1=0, sigma_2=0):

    s = 2*size+1
    window = [1, s, s, 1]
    middle = int((s**2-1)/2)
    values = s**2

    depth_image = layers.Input(shape=(height, width), dtype=tf.float32)
    indices = layers.Input(shape=(height, width, 2), dtype=tf.float32)

    depth_patches = tf.reshape(tf.image.extract_patches(tf.expand_dims(depth_image, axis=-1),
                                                        window, padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (height, width, values, 1))
    centers = tf.gather(depth_patches, [middle], axis=-2)
    index_patches = tf.reshape(tf.image.extract_patches(indices, window, padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (height, width, values, 2))
    indicator_patches = tf.where(tf.less(depth_patches, 0.001), 0.0, 1.0)

    index_differences = tf.abs(tf.subtract(index_patches, tf.gather(index_patches, [middle], axis=-2)))
    depth_patches_div = tf.math.divide_no_nan(depth_patches, centers)

    gauss_values = tf.exp(-tf.multiply(tf.square(tf.subtract(depth_patches_div, 1)), sigma_1) - tf.multiply(tf.reduce_sum(tf.square(index_differences), axis=-1, keepdims=True), sigma_2)) * indicator_patches

    sums = tf.reduce_sum(gauss_values, axis=-2)
    val_sums = tf.reduce_sum(tf.multiply(gauss_values, depth_patches), axis=-2)
    out = tf.squeeze(tf.math.divide_no_nan(val_sums, sums), axis=-1) * tf.squeeze(tf.where(tf.less(depth_image, 0.001), 0.0, 1.0), axis=0)

    model = Model(inputs=[depth_image, indices], outputs=[out])
    return lambda x, y:  model.predict([np.asarray([x]), np.asarray([y])])


def convert_to_log_depth(depth_image):
    log_image = tf.math.log(depth_image)
    min = tf.reduce_min(tf.where(tf.math.is_inf(log_image), np.inf, log_image))
    max = tf.reduce_max(log_image)

    log_image = log_image - min
    log_image = tf.math.divide_no_nan(log_image, max-min)
    log_image = (tf.ones_like(log_image) - log_image) * 0.85 + 0.15

    log_image = tf.where(tf.math.is_inf(log_image), 0.0, log_image)
    return log_image

def get_angle_arrays(height=height, width=width):
    x_val = tf.constant(viewing_angle_x / 2, dtype="float32")
    y_val = tf.constant(viewing_angle_y / 2, dtype="float32")

    x_list = tf.scalar_mul(x_val, tf.subtract(tf.divide(tf.range(0, width, 1, dtype="float32"), width / 2 - 0.5), 1))
    y_list = tf.scalar_mul(y_val,
                           tf.subtract(tf.divide(tf.range(height - 1, -1, -1, dtype="float32"), height / 2 - 0.5), 1))

    x_array = tf.expand_dims(tf.tile(tf.expand_dims(x_list, axis=0), [height, 1]), axis=-1)
    y_array = tf.expand_dims(tf.tile(tf.expand_dims(y_list, axis=1), [1, width]), axis=-1)
    angle_values_x = tf.math.tan(x_array)
    angle_values_y = tf.math.tan(y_array)
    return angle_values_x, angle_values_y

def normals_and_log_depth_model_GPU(pool_size=3, height=height, width=width):
    p = pool_size*2+1

    angle_values_x, angle_values_y = get_angle_arrays()

    depth_image = layers.Input(batch_shape=(height, width), dtype=tf.float32)
    depth_image_expanded = tf.expand_dims(depth_image, axis=-1)

    log_image = tf.expand_dims(convert_to_log_depth(depth_image_expanded), axis=0)

    #smoothed_depth = uniform_filter_with_depth_cutoff(depth_image_expanded, log_image, [1, p, p, 1], int((p**2-1)/2), 1, p**2)
    #smoothed_log = uniform_filter_with_depth_cutoff(log_image, log_image, [1, p, p, 1], int((p**2-1)/2), 1, p**2)

    x_array = angle_values_x*depth_image_expanded
    y_array = angle_values_y*depth_image_expanded
    points_in_space = layers.Concatenate()([x_array, y_array, depth_image_expanded])

    pad_1 = tf.pad(points_in_space, [[2, 0], [0, 0], [0, 0]])
    pad_2 = tf.pad(points_in_space, [[0, 2], [0, 0], [0, 0]])
    pad_3 = tf.pad(points_in_space, [[0, 0], [2, 0], [0, 0]])
    pad_4 = tf.pad(points_in_space, [[0, 0], [0, 2], [0, 0]])

    sub_1 = tf.subtract(pad_1, pad_2)[1:-1, :, :]
    sub_2 = tf.subtract(pad_4, pad_3)[:, 1:-1, :]

    vectors = tf.linalg.cross(sub_1, sub_2)
    condition = -tf.cast(tf.greater(tf.gather(vectors, [2], axis=-1), 0), tf.float32)
    vectors = 2*tf.multiply(tf.broadcast_to(condition, tf.shape(vectors)), vectors) + vectors

    norm = tf.norm(vectors, axis=-1, keepdims=True)
    normals = tf.math.divide_no_nan(vectors, tf.broadcast_to(norm, tf.shape(vectors)))

    angles = Calc_Angles()(normals)
    smoothed_angles = uniform_filter_with_depth_cutoff(tf.expand_dims(angles, axis=0), log_image, [1, p, p, 1], int((p**2-1)/2), 2, p**2)

    model = Model(inputs=depth_image, outputs=[tf.squeeze(tf.squeeze(log_image, axis=-1), axis=0), smoothed_angles, normals, points_in_space])
    return lambda image: model.predict(image, batch_size=height)

def find_surfaces_model_GPU(depth=4, threshold=0.007003343675404672 * 11, height=height, width=width, alpha=3, s1=2, s2=1, n1=11, n2=5, component_threshold=20):
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

def extract_texture_function():
    resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(480, 640, 3))
    #pool = layers.AvgPool2D((3, 3), strides=(1, 1), padding="same")(resnet.layers[38].output)
    #norm = tf.math.divide_no_nan(resnet.layers[38].output, tf.norm(resnet.layers[70].output, axis=-1, keepdims=True))
    model = Model(inputs=resnet.inputs, outputs=tf.squeeze(resnet.layers[17].output, axis=0))
    return lambda x: model(np.expand_dims(x, axis=0))

if __name__ == '__main__':
    extract_texture_function()
