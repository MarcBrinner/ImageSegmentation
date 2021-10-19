import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from standard_values import *
from tensorflow.keras import layers, Model, initializers, optimizers, regularizers, losses

def print_tensor(input):
    p = Print_Tensor()(input)
    return p

class Variable(layers.Layer):
    def __init__(self, initial_value, **kwargs):
        self.initial_value = initial_value
        self.output_shape2 = np.shape(initial_value)
        super(Variable, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='weights',
                                      shape=self.output_shape2, trainable=False,
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
        #print(tf.reduce_max(tf.abs(input_data)))
        #print(tf.abs(input_data))
        print(tf.reduce_max(input_data))
        return input_data

    def compute_output_shape(self, input_shape):
        return input_shape

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

def extract_texture_function():
    resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(480, 640, 3))
    model = Model(inputs=resnet.inputs, outputs=tf.squeeze(resnet.layers[17].output, axis=0))
    return lambda x: model(np.expand_dims(x, axis=0))

def chi_squared_distances_model_2D(pool_size=(5, 5), strides=(2, 2)):
    input = layers.Input(shape=(None, None, None))
    number_of_surfaces = tf.shape(input)[-1]
    pool = layers.AveragePooling2D(pool_size=pool_size, strides=strides)(input)
    pool = tf.transpose(pool, (0, 3, 1, 2))
    expansion_1 = tf.repeat(tf.expand_dims(pool, axis=1), repeats=[number_of_surfaces], axis=1)
    expansion_2 = tf.repeat(tf.expand_dims(pool, axis=2), repeats=[number_of_surfaces], axis=2)
    squared_difference = tf.square(tf.subtract(expansion_1, expansion_2))
    addition = tf.add(expansion_1, expansion_2)
    distance = tf.reduce_sum(tf.reduce_sum(tf.math.divide_no_nan(squared_difference, addition), axis=-1), axis=-1)[0]
    model = Model(inputs=[input], outputs=distance)
    return lambda histogram: model.predict(np.asarray([histogram]))

def chi_squared_distances_model_1D():
    input = layers.Input(shape=(None, None))
    number_of_surfaces = tf.shape(input)[-1]
    transpose = tf.transpose(input, (0, 2, 1))
    expansion_1 = tf.repeat(tf.expand_dims(transpose, axis=1), repeats=[number_of_surfaces], axis=1)
    expansion_2 = tf.repeat(tf.expand_dims(transpose, axis=2), repeats=[number_of_surfaces], axis=2)
    squared_difference = tf.square(tf.subtract(expansion_1, expansion_2))
    addition = tf.add(expansion_1, expansion_2)
    distance = tf.reduce_sum(tf.math.divide_no_nan(squared_difference, addition), axis=-1)
    model = Model(inputs=[input], outputs=distance[0])
    return lambda histogram: model.predict(np.asarray([histogram]))
