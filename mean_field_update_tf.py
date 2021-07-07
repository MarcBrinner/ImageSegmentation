import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, initializers, optimizers, regularizers

variable_names = ["w_1", "w_2", "w_3", "w_4", "w_5", "theta_1_1", "theta_2_1", "theta_2_2", "theta_3_1", "theta_3_2", "theta_4_1", "theta_4_2", "theta_5_1", "theta_5_2", "theta_5_3", "weight"]

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
    return list(np.load(f"parameters/{index}.npy"))

class Variable(layers.Layer):
    def __init__(self, initial_value, **kwargs):
        self.initial_value = initial_value
        self.output_shape2 = np.shape(initial_value)
        super(Variable, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='weights',
                                      shape=self.output_shape2, trainable=True, initializer=initializers.constant(self.initial_value), dtype=tf.float32)
        super(Variable, self).build(input_shape)

    def call(self, input_data):
        return self.kernel

    def compute_output_shape(self, input_shape):
        return (self.output_shape2[0], self.output_shape2[1])

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
        return self.kernel

    def compute_output_shape(self, input_shape):
        return (self.output_shape2[0], self.output_shape2[1])

class Print_Tensor(layers.Layer):
    def __init__(self, **kwargs):
        super(Print_Tensor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Print_Tensor, self).build(input_shape)

    def call(self, input_data):
        return input_data

    def compute_output_shape(self, input_shape):
        return input_shape

def mean_field_update_model(number_of_pixels, number_of_surfaces, Q, features_1, features_2, features_3, features_4, features_5, matrix,
                            w_1, w_2, w_3, w_4, w_5, theta_1, theta_2, theta_3, theta_4, theta_5, weight, batch_size):
    Q = tf.tile(tf.expand_dims(tf.constant(Q, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    matrix =  tf.tile(tf.expand_dims(tf.constant(matrix, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    features_1 = tf.tile(tf.expand_dims(tf.constant(features_1, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_2 = tf.tile(tf.expand_dims(tf.constant(features_2, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_3 = tf.tile(tf.expand_dims(tf.constant(features_3, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_4 = tf.tile(tf.expand_dims(tf.constant(features_4, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_5 = tf.tile(tf.expand_dims(tf.constant(features_5, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    w_1 = tf.constant(w_1, dtype=tf.float32)
    w_2 = tf.constant(w_2, dtype=tf.float32)
    w_3 = tf.constant(w_3, dtype=tf.float32)
    w_4 = tf.constant(w_4, dtype=tf.float32)
    w_5 = tf.constant(w_5, dtype=tf.float32)

    theta_1 = tf.tile(tf.expand_dims(tf.constant(theta_1, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    theta_2 = tf.tile(tf.expand_dims(tf.constant(theta_2, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    theta_3 = tf.tile(tf.expand_dims(tf.constant(theta_3, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    theta_4 = tf.tile(tf.expand_dims(tf.constant(theta_4, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    theta_5 = tf.tile(tf.expand_dims(tf.constant(theta_5, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    feature_1 = layers.Input(shape=(2,), batch_size=batch_size, dtype=tf.float32)
    feature_2 = layers.Input(shape=(3,), batch_size=batch_size, dtype=tf.float32)
    feature_3 = layers.Input(shape=(5,), batch_size=batch_size, dtype=tf.float32)
    feature_4 = layers.Input(shape=(4,), batch_size=batch_size, dtype=tf.float32)
    feature_5 = layers.Input(shape=(5,), batch_size=batch_size, dtype=tf.float32)

    weight = tf.constant(weight, dtype=tf.float32)

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

    feature_4_extended = tf.tile(tf.expand_dims(feature_4, axis=1), np.asarray([1, number_of_pixels, 1]))
    differences_4 = tf.subtract(feature_4_extended, features_4)
    similarities_4 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_4), tf.broadcast_to(theta_4, tf.shape(differences_4))),axis=-1))

    feature_5_extended = tf.tile(tf.expand_dims(feature_5, axis=1), np.asarray([1, number_of_pixels, 1]))
    differences_5 = tf.subtract(feature_5_extended, features_5)
    similarities_5 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_5), tf.broadcast_to(theta_5, tf.shape(differences_5))),axis=-1))

    similarity_sum = layers.Add()([tf.scalar_mul(w_1, similarities_1),
                                 tf.scalar_mul(w_2, similarities_2),
                                 tf.scalar_mul(w_3, similarities_3),
                                 tf.scalar_mul(w_4, similarities_4),
                                 tf.scalar_mul(w_5, similarities_5)])

    messages = tf.scalar_mul(weight, tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(Q)), Q), axis=1))

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=1), [1, number_of_surfaces, 1])), axis=-1)
    potentials = tf.exp(-unary_potentials - compatibility_values)

    output = tf.truediv(potentials, tf.broadcast_to(tf.reduce_sum(potentials, axis=-1, keepdims=True), tf.shape(potentials)))

    model = Model(inputs=[feature_1, feature_2, feature_3, feature_4, feature_5, unary_potentials], outputs=output)
    model.summary()
    return model

def mean_field_update_model_learned(number_of_pixels, number_of_surfaces, Q, features_1, features_2, features_3, features_4, features_5, matrix,
                            w_1, w_2, w_3, w_4, w_5, theta_1_1, theta_2_1, theta_2_2, theta_3_1, theta_3_2, theta_4_1, theta_4_2, theta_5_1, theta_5_2, theta_5_3, weight, batch_size):
    Q = tf.tile(tf.expand_dims(tf.constant(Q, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    matrix =  tf.tile(tf.expand_dims(tf.constant(matrix, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    features_1 = tf.tile(tf.expand_dims(tf.constant(features_1, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_2 = tf.tile(tf.expand_dims(tf.constant(features_2, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_3 = tf.tile(tf.expand_dims(tf.constant(features_3, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_4 = tf.tile(tf.expand_dims(tf.constant(features_4, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_5 = tf.tile(tf.expand_dims(tf.constant(features_5, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    feature_1 = layers.Input(shape=(2,), batch_size=batch_size, dtype=tf.float32)
    feature_2 = layers.Input(shape=(3,), batch_size=batch_size, dtype=tf.float32)
    feature_3 = layers.Input(shape=(5,), batch_size=batch_size, dtype=tf.float32)
    feature_4 = layers.Input(shape=(4,), batch_size=batch_size, dtype=tf.float32)
    feature_5 = layers.Input(shape=(5,), batch_size=batch_size, dtype=tf.float32)

    w_1 = Variable(w_1, name="w_1")(feature_1)
    w_2 = Variable(w_2, name="w_2")(feature_1)
    w_3 = Variable(w_3, name="w_3")(feature_1)
    w_4 = Variable(w_4, name="w_4")(feature_1)
    w_5 = Variable(w_5, name="w_5")(feature_1)

    theta_1 = tf.expand_dims(tf.repeat(Variable(theta_1_1, name="theta_1_1")(feature_1), repeats=[2], axis=-1), axis=0, name="Theta_1")
    theta_2_1 = tf.expand_dims(tf.repeat(Variable(theta_2_1, name="theta_2_1")(feature_1), repeats=[2], axis=-1), axis=0)
    theta_2_2 = tf.expand_dims(tf.repeat(Variable(theta_2_2, name="theta_2_2")(feature_1), repeats=[1], axis=-1), axis=0)
    theta_3_1 = tf.expand_dims(tf.repeat(Variable(theta_3_1, name="theta_3_1")(feature_1), repeats=[2], axis=-1), axis=0)
    theta_3_2 = tf.expand_dims(tf.repeat(Variable(theta_3_2, name="theta_3_2")(feature_1), repeats=[3], axis=-1), axis=0)
    theta_4_1 = tf.expand_dims(tf.repeat(Variable(theta_4_1, name="theta_4_1")(feature_1), repeats=[2], axis=-1), axis=0)
    theta_4_2 = tf.expand_dims(tf.repeat(Variable(theta_4_2, name="theta_4_2")(feature_1), repeats=[2], axis=-1), axis=0)
    theta_5_1 = tf.expand_dims(tf.repeat(Variable(theta_5_1, name="theta_5_1")(feature_1), repeats=[2], axis=-1), axis=0)
    theta_5_2 = tf.expand_dims(tf.repeat(Variable(theta_5_2, name="theta_5_2")(feature_1), repeats=[1], axis=-1), axis=0)
    theta_5_3 = tf.expand_dims(tf.repeat(Variable(theta_5_3, name="theta_5_3")(feature_1), repeats=[2], axis=-1), axis=0)

    theta_2 = tf.concat([theta_2_1, theta_2_2], axis=-1, name="Theta_2")
    theta_3 = tf.concat([theta_3_1, theta_3_2], axis=-1, name="Theta_3")
    theta_4 = tf.concat([theta_4_1, theta_4_2], axis=-1, name="Theta_4")
    theta_5 = tf.concat([theta_5_1, theta_5_2, theta_5_3], axis=-1, name="Theta_5")


    theta_1 = tf.tile(tf.expand_dims(theta_1, axis=0), [batch_size, 1, 1])
    theta_2 = tf.tile(tf.expand_dims(theta_2, axis=0), [batch_size, 1, 1])
    theta_3 = tf.tile(tf.expand_dims(theta_3, axis=0), [batch_size, 1, 1])
    theta_4 = tf.tile(tf.expand_dims(theta_4, axis=0), [batch_size, 1, 1])
    theta_5 = tf.tile(tf.expand_dims(theta_5, axis=0), [batch_size, 1, 1])

    weight = Variable(weight, name="weight")(feature_1)

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

    feature_4_extended = tf.tile(tf.expand_dims(feature_4, axis=1), np.asarray([1, number_of_pixels, 1]))
    differences_4 = tf.subtract(feature_4_extended, features_4)
    similarities_4 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_4), tf.broadcast_to(theta_4, tf.shape(differences_4))),axis=-1))

    feature_5_extended = tf.tile(tf.expand_dims(feature_5, axis=1), np.asarray([1, number_of_pixels, 1]))
    differences_5 = tf.subtract(feature_5_extended, features_5)
    similarities_5 = tf.exp(-tf.reduce_sum(tf.multiply(tf.square(differences_5), tf.broadcast_to(theta_5, tf.shape(differences_5))),axis=-1))

    similarity_sum = layers.Add()([tf.scalar_mul(w_1, similarities_1),
                                 tf.scalar_mul(w_2, similarities_2),
                                 tf.scalar_mul(w_3, similarities_3),
                                 tf.scalar_mul(w_4, similarities_4),
                                 tf.scalar_mul(w_5, similarities_5)])

    messages = tf.scalar_mul(weight, tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(Q)), Q), axis=1))

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=1), [1, number_of_surfaces, 1])), axis=-1)
    #p = Print_Tensor(dtype=tf.float32)(compatibility_values)
    add = layers.Add(activity_regularizer=regularizers.l2(0.00001))([unary_potentials, compatibility_values])

    #o = Print_Tensor()(layers.Softmax()(-add))
    output = layers.Softmax()(-add)

    model = Model(inputs=[feature_1, feature_2, feature_3, feature_4, feature_5, unary_potentials], outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[],
                  run_eagerly=False)
    model.summary()
    return model

def custom_loss(y_actual, y_predicted):
    only_wrong_ones = tf.multiply(y_actual, y_predicted)
    error = tf.reduce_sum(only_wrong_ones, axis=-1)
    #print(y_predicted)
    #print(y_predicted)
    print(tf.reduce_min(error))
    return 100*error

if __name__ == '__main__':
    batch_size = 64
    number_of_surfaces = 33
    n = 480*640
    model = mean_field_update_model(n, number_of_surfaces, np.zeros((n, number_of_surfaces)), np.zeros((n, 2)), np.zeros((n, 3)),
                                    np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((number_of_surfaces, number_of_surfaces)), 1, 1, 1, 1,
                                    np.ones((1, 2)), np.ones((1, 3)), np.ones((1, 5)), np.ones((1, 5)), 1/100000, batch_size)
    output = model.predict([np.zeros((batch_size, 2)), np.zeros((batch_size, 3)), np.zeros((batch_size, 5)), np.zeros((batch_size, 5)), np.zeros((batch_size, number_of_surfaces))])
    print(output)