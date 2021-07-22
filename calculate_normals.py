import math

import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers, Model
from image_filters import *
from image_operations import calculate_curvature_scores, convert_depth_image
from scipy.spatial.transform import Rotation
from load_images import load_image
from plot_image import plot_normals
from standard_values import *

@njit()
def calculate_normals_plane_fitting(image, neighborhood_size):
    return_array = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
    sigma = np.array((1.4, 0.002, 0.002))
    height, width = np.shape(image)
    factor_x = math.tan(viewing_angle_x / 2) * 2 / width
    factor_y = math.tan(viewing_angle_y / 2) * 2 / height
    for i in range(height):
        print(i)
        values_y = list(range(max(i - neighborhood_size, 0), min(i + neighborhood_size + 1, height - 1)))
        values_x = list(range(0, neighborhood_size + 1))
        for j in range(width):
            d = image[i][j]
            if d < 0.0001:
                del values_x[0]
                if j + neighborhood_size + 1 < width:
                    values_x.append(j + neighborhood_size + 1)
                continue
            depth_factor_x = factor_x * d
            depth_factor_y = factor_y * d
            diffs = []
            inputs = []
            outputs = []
            for k in values_y:
                for l in values_x:
                    if k != i or l != j:
                        #percentage = abs(k-i)/(abs(k-i) + abs(l-j))
                        #diffs.append([(d - image[k][l]) / (percentage * depth_factor_y + (1-percentage)*depth_factor_x), k-i, l-j])
                        diffs.append([(d - image[k][l]) / math.sqrt((abs(k-i) * depth_factor_y)**2 + (abs(l-j)*depth_factor_x)**2), k-i, l-j])
                        inputs.append([(l-j)*depth_factor_x, (i-k)*depth_factor_y, 1.0])
                        outputs.append(float(image[k][l]))
            diffs.append([0.0, 0.0, 0.0])
            inputs.append([0.0, 0.0, 1.0])
            outputs.append(d)

            diffs = np.asarray(diffs)
            inputs = np.asarray(inputs)
            outputs = np.asarray(outputs)
            weights = np.expand_dims(np.exp(-np.sum(diffs*sigma*diffs, axis=-1)), axis=-1)
            weighted_inputs = np.square(weights)*inputs
            vec = np.linalg.lstsq(np.dot(np.transpose(inputs), weighted_inputs), np.dot(np.transpose(weighted_inputs), outputs))[0]

            n = np.linalg.norm(vec)
            if n > 0:
                vec[2] = -1
                vec = vec/np.linalg.norm(vec)

            return_array[i][j] = vec
            del values_x[0]
            if j+neighborhood_size+1 < width:
                values_x.append(j + neighborhood_size + 1)
    return return_array

@njit()
def calculate_normals_plane_fitting_with_curvature(image):
    neighborhood_value = 5
    return_array = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
    sigma = np.array((1, 0, 0, 4))
    height, width = np.shape(image)
    factor_x = math.tan(viewing_angle_x / 2) * 2 / width
    factor_y = math.tan(viewing_angle_y / 2) * 2 / height
    for i in range(height):
        print(i)
        values_y = list(range(max(i - neighborhood_value, 0), min(i + neighborhood_value + 1, height - 1)))
        values_x = list(range(0, neighborhood_value+1))
        for j in range(width):
            d = image[i][j]
            if d < 0.0001:
                del values_x[0]
                if j + neighborhood_value + 1 < width:
                    values_x.append(j + neighborhood_value + 1)
                continue
            depth_factor_x = factor_x * d
            depth_factor_y = factor_y * d
            curvature_scores = calculate_curvature_scores(image, neighborhood_value, i, j, np.asarray([depth_factor_x, depth_factor_y]), d, width, height)

            diffs = []
            inputs = []
            outputs = []
            for k in values_y:
                for l in values_x:
                    if k != i or l != j:
                        curvature_score = curvature_scores[k-i+neighborhood_value][l-j+neighborhood_value]
                        percentage = abs(k-i)/(abs(k-i) + abs(l-j))
                        diffs.append([(d - image[k][l]) / math.sqrt((abs(k-i) * depth_factor_y)**2 + (abs(l-j)*depth_factor_x)**2), k-i, l-j, curvature_score])
                        #diffs.append([(d - image[k][l]) / (percentage * depth_factor_y + (1-percentage)*depth_factor_x), k-i, l-j, curvature_score])
                        inputs.append([(l-j)*depth_factor_x, (i-k)*depth_factor_y, 1.0])
                        outputs.append(float(image[k][l]))
            diffs.append([0.0, 0.0, 0.0, 0.0])
            inputs.append([0.0, 0.0, 1.0])
            outputs.append(d)

            diffs = np.asarray(diffs)
            inputs = np.asarray(inputs)
            outputs = np.asarray(outputs)
            weights = np.expand_dims(np.exp(-np.sum(diffs*sigma*diffs, axis=-1)), axis=-1)
            weighted_inputs = np.square(weights)*inputs
            vec = np.linalg.lstsq(np.dot(np.transpose(inputs), weighted_inputs), np.dot(np.transpose(weighted_inputs), outputs))[0]

            n = np.linalg.norm(vec)
            if n > 0:
                vec[2] = -1
                vec = vec/np.linalg.norm(vec)

            return_array[i][j] = vec
            del values_x[0]
            if j+neighborhood_value+1 < width:
                values_x.append(j+neighborhood_value+1)
    return return_array


@njit()
def calculate_normals_cross_product(image):
    height, width = np.shape(image)
    factor_x = math.tan(viewing_angle_x/2) * 2 / width
    factor_y = math.tan(viewing_angle_y/2) * 2 / height
    return_array = np.zeros((height, width, 3))
    for i in range(1, height-1):
        for j in range(1, width-1):
            if image[i][j] == 0:
                continue
            depth_dif = image[i-1][j] - image[i+1][j]
            vec_1 = [0, 2 * factor_y * image[i][j], depth_dif]

            depth_dif = image[i][j+1] - image[i][j-1]
            vec_2 = [2 * factor_x * image[i][j], 0, depth_dif]

            cp = np.cross(vec_1, vec_2)
            if np.any(np.isnan(cp)):
                cp = np.asarray([0.0, 0.0, 0.0])
            else:
                if cp[2] > 0:
                    cp = -cp
                cp = cp/np.linalg.norm(cp)
            return_array[i][j] = cp
    return return_array

@njit()
def calc_angle(vec_1, vec_2):
    return math.acos(np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2)))

@njit()
def normals_to_angles(normal_image):
    height, width, x = np.shape(normal_image)
    angles = np.zeros((height, width, 2))
    for y in range(height):
        for x in range(width):
            vec = normal_image[y][x]
            if np.linalg.norm(vec) < 0.001:
                angles[y][x] = [math.pi, math.pi]
                continue

            vec_proj = vec.copy()
            vec_proj[0] = 0
            angles[y][x][0] = math.acos(-vec_proj[2]/(np.linalg.norm(vec_proj))) * np.sign(vec_proj[1])

            vec_proj = vec.copy()
            vec_proj[1] = 0
            angles[y][x][1] = math.acos(-vec_proj[2]/(np.linalg.norm(vec_proj))) * -np.sign(vec_proj[0])
    return angles

def angles_to_normals(angles, depth_image):
    height, width, _ = np.shape(angles)
    normals = np.zeros((height, width, 3))
    axis_1 = np.asarray([1.0, 0.0, 0.0])
    axis_2 = np.asarray([0.0, 1.0, 0.0])
    middle_vector = np.asarray([0, 0, -1])
    for y in range(height):
        for x in range(width):
            if depth_image[y][x] < 0.0001:
                continue
            vec = middle_vector.copy()
            rot = Rotation.from_rotvec(angles[y][x][0] * axis_1)
            vec = rot.apply(vec)
            rot = Rotation.from_rotvec(angles[y][x][1] * axis_2)
            vec = rot.apply(vec)
            normals[y][x] = vec
    return normals

@njit()
def calculate_normals_as_angles_final(depth_image, log_depth):
    smoothed = uniform_filter_with_log_depth_cutoff_depth(depth_image, log_depth, 2)
    normals = calculate_normals_cross_product(smoothed)
    angles = normals_to_angles(normals)
    angles = uniform_filter_with_log_depth_cutoff_angles(angles, log_depth, 2)
    return angles

def angle_calculator(vec):
    mult_1 = tf.constant(np.asarray([0.0, 1.0, 1.0]), dtype=tf.float32)
    mult_2 = tf.constant(np.asarray([1.0, 0.0, 1.0]), dtype=tf.float32)

    new_vec_1 = tf.multiply(vec, mult_1)
    new_vec_2 = tf.multiply(vec, mult_2)

    angle_1 = tf.acos(tf.clip_by_value(tf.math.divide_no_nan(-new_vec_1[2], tf.linalg.norm(new_vec_1)), -1, 1)) * tf.sign(new_vec_1[1])
    angle_2 = tf.acos(tf.clip_by_value(tf.math.divide_no_nan(-new_vec_2[2], tf.linalg.norm(new_vec_2)), -1, 1)) * -tf.sign(new_vec_2[0])
    return tf.stack([angle_1, angle_2], axis=0)


class Calc_Angles(layers.Layer):
    def call(self, input):
        return tf.vectorized_map(lambda x: tf.vectorized_map(angle_calculator, x), input)

class Depth_Cutoff(layers.Layer):
    def call(self, input, middle):
        return tf.vectorized_map(lambda x: tf.vectorized_map(lambda y: tf.where(tf.greater(tf.abs(tf.subtract(y, y[middle])), 0.03), tf.zeros_like(y), tf.ones_like(y)), x), input)

def uniform_filter_with_depth_cutoff(image, depth_image, window, middle, dimension, vals_per_window):
    patches = tf.reshape(tf.image.extract_patches(image, window, padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), (height, width, vals_per_window, dimension))
    depth_patches = tf.squeeze(tf.image.extract_patches(depth_image, window, padding="SAME", strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]), axis=0)
    mult_values = Depth_Cutoff()(depth_patches, middle)
    sums = tf.reduce_sum(mult_values, axis=-1, keepdims=True)
    val_sums = tf.reduce_sum(tf.multiply(tf.expand_dims(mult_values, axis=-1), patches), axis=-2)
    return tf.math.divide_no_nan(val_sums, sums)

def convert_to_log_depth(depth_image):
    log_image = tf.math.log(depth_image)
    min = tf.reduce_min(tf.where(tf.math.is_inf(log_image), np.inf, log_image))
    max = tf.reduce_max(log_image)

    log_image = log_image - min
    log_image = tf.math.divide_no_nan(log_image, max-min)
    log_image = (tf.ones_like(log_image) - log_image) * 0.85 + 0.15

    log_image = tf.where(tf.math.is_inf(log_image), 0.0, log_image)
    return log_image

def calculate_normals_GPU_model(pool_size=2, height=height, width=width):
    p = pool_size*2+1

    depth_image = layers.Input(batch_shape=(1, height, width), dtype=tf.float32)
    depth_image_expanded = tf.expand_dims(depth_image, axis=-1)

    log_image = tf.expand_dims(convert_to_log_depth(depth_image), axis=-1)

    smoothed_depth = uniform_filter_with_depth_cutoff(depth_image_expanded, log_image, [1, 5, 5, 1], 12, 1, 25)
    smoothed_log = uniform_filter_with_depth_cutoff(log_image, log_image, [1, 5, 5, 1], 12, 1, 25)

    factor_x = layers.Input(shape=(1,), dtype=tf.float32)
    factor_y = layers.Input(shape=(1,), dtype=tf.float32)

    zeros = tf.zeros(tf.shape(smoothed_depth))
    x_values = tf.multiply(tf.broadcast_to(factor_x, tf.shape(smoothed_depth)), smoothed_depth)
    y_values = tf.multiply(tf.broadcast_to(factor_y, tf.shape(smoothed_depth)), smoothed_depth)

    pad_1 = tf.pad(smoothed_depth, [[2, 0], [0, 0], [0, 0]])
    pad_2 = tf.pad(smoothed_depth, [[0, 2], [0, 0], [0, 0]])
    pad_3 = tf.pad(smoothed_depth, [[0, 0], [2, 0], [0, 0]])
    pad_4 = tf.pad(smoothed_depth, [[0, 0], [0, 2], [0, 0]])

    sub_1 = tf.subtract(pad_1, pad_2)[1:-1, :, :]
    sub_2 = tf.subtract(pad_4, pad_3)[:, 1:-1, :]

    concat_1 = layers.Concatenate(axis=-1)([zeros, y_values, sub_1])
    concat_2 = layers.Concatenate(axis=-1)([x_values, zeros, sub_2])

    vectors = tf.linalg.cross(concat_1, concat_2)
    condition = -tf.cast(tf.greater(tf.gather(vectors, [2], axis=-1), 0), tf.float32)
    vectors = 2*tf.multiply(tf.broadcast_to(condition, tf.shape(vectors)), vectors) + vectors

    norm = tf.norm(vectors, axis=-1, keepdims=True)
    normals = tf.math.divide_no_nan(vectors, tf.broadcast_to(norm, tf.shape(vectors)))

    angles = Calc_Angles()(normals)
    smoothed_angles = uniform_filter_with_depth_cutoff(tf.expand_dims(angles, axis=0), log_image, [1, 5, 5, 1], 12, 2, 25)

    model = Model(inputs=[depth_image, factor_x, factor_y], outputs=[smoothed_log, smoothed_angles])
    return model

def create_normal_calculation_function_GPU():
    model = calculate_normals_GPU_model()
    return lambda image: model.predict([np.asarray([image]), np.asarray([2*factor_x]), np.asarray([2*factor_y])], batch_size=1)