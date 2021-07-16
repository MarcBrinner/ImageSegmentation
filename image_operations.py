import numpy as np
import math
from PIL import ImageCms, Image
from numba import njit

neighborhood_value = 7
viewing_angle_x = 62.0 / 180 * math.pi
viewing_angle_y = 48.6 / 180 * math.pi

def rgb_to_Lab(image):
    image = Image.fromarray(image)

    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")

    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    lab_im = np.asarray(ImageCms.applyTransform(image, rgb2lab_transform))

    return lab_im

def convert_depth_image(image):
    new_image = np.log10(image, where=image != 0)

    min = np.min(new_image, where=image != 0, initial=np.inf)
    max = np.max(new_image, where=image != 0, initial=-np.inf)

    new_image = new_image - min
    new_image = new_image / (max-min)
    new_image = (1 - new_image) * 0.85 + 0.15
    new_image[image == 0] = 0

    return new_image

@njit()
def calculate_curvature_scores(image, log_depth, neighborhood_value, i, j, factor_array, d, width, height):
    curvature_scores = np.zeros((neighborhood_value * 2 + 1, neighborhood_value * 2 + 1))
    central_point = np.asarray([0, 0, d])
    directions = np.asarray([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    for direction in directions:
        distance_prev_point = 0
        prev_point = None
        prev_score = 0
        log_depth_prev_point = 0
        for k in range(1, neighborhood_value + 1):
            x_y_difference = direction * k
            current_x_y = x_y_difference * factor_array
            current_indices = (i+x_y_difference[0], j+x_y_difference[1])
            if current_indices[0] >= height or current_indices[0] < 0 or current_indices[1] >= width or current_indices[1] < 0:
                continue
            current_point = np.asarray([current_x_y[0], current_x_y[1], image[current_indices[0]][current_indices[1]]])
            distance_current_point = np.sqrt(np.sum(np.square(current_point - central_point)))
            log_depth_current_point = log_depth[current_indices[0]][current_indices[1]]
            if k == 1:
                distance_prev_point = distance_current_point
                prev_point = current_point
                log_depth_prev_point = log_depth_current_point
                continue
            if abs(log_depth_prev_point-log_depth_current_point) > 0.01:
                break
            distance_current_points = np.sqrt(np.sum(np.square(current_point - prev_point)))
            score = (distance_current_points + distance_prev_point) / distance_current_point - 1 + prev_score
            curvature_scores[neighborhood_value + x_y_difference[0]][neighborhood_value + x_y_difference[1]] = score
            prev_score = score
            distance_prev_point = distance_current_point
            prev_point = current_point
            log_depth_prev_point = log_depth_current_point

    # for direction in directions:
    #     curvature_scores[neighborhood_value + direction[0]][neighborhood_value + direction[1]] =\
    #         0.5 * curvature_scores[neighborhood_value + 2*direction[0]][neighborhood_value + 2*direction[1]]
    #
    # for level in range(2, neighborhood_value+1):
    #     for k in range(8):
    #         direction_1 = level * directions[k-1] + neighborhood_value
    #         direction_2 = level * directions[k] + neighborhood_value
    #         if direction_1[0] != direction_2[0]:
    #             different_index = 0
    #             values = [np.asarray([min(direction_1[0], direction_2[0]) + l, direction_1[1]]) for l in range(1, level)]
    #         else:
    #             different_index = 1
    #             values = [np.asarray([direction_1[0], min(direction_1[1], direction_2[1]) + l]) for l in range(1, level)]
    #         for value in values:
    #             percentage = abs(direction_1[different_index] - value[different_index])/level
    #             interpolated_value = percentage * curvature_scores[direction_2[0]][direction_2[1]] +\
    #                                  (1-percentage) * curvature_scores[direction_1[0]][direction_1[1]]
    #             curvature_scores[value[0]][value[1]] = interpolated_value
    return curvature_scores

@njit()
def calculate_curvature_scores_2(depth_image, log_depth, neighborhood_value, i, j, width, height):
    curvature_scores = np.zeros(4)
    directions = np.asarray([[1, 0], [1, 1], [-1, 1], [-1, 0]])
    for d in range(4):
        direction = directions[d]
        prev_log_depth_plus = log_depth[i][j]
        prev_log_depth_minus = prev_log_depth_plus
        max_val_plus = -1
        max_val_minus = -1
        for k in range(1, neighborhood_value+1):
            if max_val_plus < 0:
                if i+direction[0]*k < 0 or i+direction[0]*k >= height or j+direction[1]*k < 0 or j+direction[1]*k >= width:
                    max_val_plus = k-1
                else:
                    log_depth_plus = log_depth[i+direction[0]*k][j+direction[1]*k]
                    if abs(log_depth_plus-prev_log_depth_plus) > 0.01:
                        max_val_plus = k-1
            if max_val_minus < 0:
                if i-direction[0]*k < 0 or i-direction[0]*k >= height or j-direction[1]*k < 0 or j-direction[1]*k >= width:
                    max_val_minus = k - 1
                else:
                    log_depth_minus = log_depth[i-direction[0]*k][j-direction[1]*k]
                    if abs(log_depth_minus - prev_log_depth_minus) > 0.01:
                        max_val_minus = k - 1
        if max_val_minus == -1:
            max_val_minus = neighborhood_value
        if max_val_plus == -1:
            max_val_plus = neighborhood_value

        start_depth = depth_image[i-max_val_minus*direction[0]][j-max_val_minus*direction[1]]
        end_depth = depth_image[i+max_val_plus*direction[0]][j+max_val_plus*direction[1]]

        if max_val_plus + max_val_minus <= 1:
            continue
        #a = (end_depth - start_depth)/(max_val_plus + max_val_minus)
        inputs = np.zeros((max_val_minus + max_val_plus + 1, 2))
        outputs = np.zeros(max_val_minus + max_val_plus + 1)
        normalizer = depth_image[i][j]/10
        for k in range(-max_val_minus, max_val_plus+1):
            inputs[k] = np.asarray([k, 1])
            outputs[k] = depth_image[i+k*direction[0]][j+k*direction[1]] / normalizer
        vec = np.linalg.lstsq(np.dot(np.transpose(inputs), inputs), np.dot(np.transpose(inputs), outputs))[0]

        sum = np.sum(np.square(np.dot(inputs, vec) - outputs))
        curvature_scores[d] = sum
    return curvature_scores


if __name__ == '__main__':
    pass