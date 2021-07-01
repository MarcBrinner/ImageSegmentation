import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

def mean_field_update_model(number_of_pixels, number_of_surfaces, Q, features_1, features_2, features_3, features_4, matrix,
                            w_1, w_2, w_3, w_4, theta_1, theta_2, theta_3, theta_4, weight, batch_size):
    Q = tf.tile(tf.expand_dims(tf.constant(Q, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    matrix =  tf.tile(tf.expand_dims(tf.constant(matrix, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    features_1 = tf.tile(tf.expand_dims(tf.constant(features_1, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_2 = tf.tile(tf.expand_dims(tf.constant(features_2, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_3 = tf.tile(tf.expand_dims(tf.constant(features_3, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    features_4 = tf.tile(tf.expand_dims(tf.constant(features_4, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    w_1 = tf.constant(w_1, dtype=tf.float32)
    w_2 = tf.constant(w_2, dtype=tf.float32)
    w_3 = tf.constant(w_3, dtype=tf.float32)
    w_4 = tf.constant(w_4, dtype=tf.float32)

    theta_1 = tf.tile(tf.expand_dims(tf.constant(theta_1, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    theta_2 = tf.tile(tf.expand_dims(tf.constant(theta_2, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    theta_3 = tf.tile(tf.expand_dims(tf.constant(theta_3, dtype=tf.float32), axis=0), [batch_size, 1, 1])
    theta_4 = tf.tile(tf.expand_dims(tf.constant(theta_4, dtype=tf.float32), axis=0), [batch_size, 1, 1])

    feature_1 = layers.Input(shape=(2,), batch_size=batch_size, dtype=tf.float32)
    feature_2 = layers.Input(shape=(3,), batch_size=batch_size, dtype=tf.float32)
    feature_3 = layers.Input(shape=(5,), batch_size=batch_size, dtype=tf.float32)
    feature_4 = layers.Input(shape=(5,), batch_size=batch_size, dtype=tf.float32)

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

    similarity_sum = layers.Add()([tf.scalar_mul(w_1, similarities_1),
                                 tf.scalar_mul(w_2, similarities_2),
                                 tf.scalar_mul(w_3, similarities_3),
                                 tf.scalar_mul(w_4, similarities_4)])

    messages = tf.scalar_mul(weight, tf.reduce_sum(tf.multiply(tf.broadcast_to(tf.expand_dims(similarity_sum, axis=-1), tf.shape(Q)), Q), axis=1))

    compatibility_values = tf.reduce_sum(tf.multiply(matrix, tf.tile(tf.expand_dims(messages, axis=1), [1, number_of_surfaces, 1])), axis=1)
    potentials = tf.exp(-unary_potentials - compatibility_values)

    output = tf.truediv(potentials, tf.broadcast_to(tf.reduce_sum(potentials, axis=-1, keepdims=True), tf.shape(potentials)))

    model = Model(inputs=[feature_1, feature_2, feature_3, feature_4, unary_potentials], outputs=output)
    model.summary()
    return model

if __name__ == '__main__':
    batch_size = 64
    number_of_surfaces = 33
    n = 480*640
    model = mean_field_update_model(n, number_of_surfaces, np.zeros((n, number_of_surfaces)), np.zeros((n, 2)), np.zeros((n, 3)),
                                    np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((number_of_surfaces, number_of_surfaces)), 1, 1, 1, 1,
                                    np.ones((1, 2)), np.ones((1, 3)), np.ones((1, 5)), np.ones((1, 5)), 1/100000, batch_size)
    output = model.predict([np.zeros((batch_size, 2)), np.zeros((batch_size, 3)), np.zeros((batch_size, 5)), np.zeros((batch_size, 5)), np.zeros((batch_size, number_of_surfaces))])
    print(output)