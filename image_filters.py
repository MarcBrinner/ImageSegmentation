import numpy as np
import scipy.ndimage.filters
import math
from image_operations import viewing_angle_y, viewing_angle_x
from numba import njit

def median_filter(image, size):
    return scipy.ndimage.filters.median_filter(image, size)

@njit()
def gaussian_filter_with_norm_check(image, size, sigma):
    height, width, dim = np.shape(image)
    new_image = np.zeros(np.shape(image))
    sigma = 1/sigma**2
    for i in range(height):
        for j in range(width):
            if np.linalg.norm(image[i][j]) < 0.0001:
                continue
            weights = 0
            sum = np.zeros(dim)
            for k in range(max(0, i - size), min(height - 1, i + size + 1)):
                for l in range(max(0, j - size), min(width - 1, j + size + 1)):
                    if np.linalg.norm(image[k][l]) > 0.0001:
                        weight = np.exp(-sigma*((i-k)**2 + (j-l)**2))
                        weights += weight
                        sum += weight*image[k][l]
            if weights > 0:
                sum = sum / np.linalg.norm(sum)
                new_image[i][j] = sum
    return new_image

@njit()
def gaussian_filter_with_depth_check(image, size, sigma, depth_image, none_value):
    height, width, dim = np.shape(image)
    new_image = np.zeros(np.shape(image))
    sigma = 1 / sigma ** 2
    for i in range(height):
        for j in range(width):
            if depth_image[i][j] < 0.0001:
                new_image[i][j] = none_value.copy()
                continue
            weights = 0
            sum = np.zeros(dim)
            for k in range(max(0, i - size), min(height - 1, i + size + 1)):
                for l in range(max(0, j - size), min(width - 1, j + size + 1)):
                    if depth_image[i][j] > 0.0001:
                        weight = np.exp(-sigma * ((i - k) ** 2 + (j - l) ** 2))
                        weights += weight
                        sum += weight * image[k][l]
            if weights > 0:
                sum = sum / np.linalg.norm(sum)
                new_image[i][j] = sum
            else:
                new_image[i][j] = none_value.copy()
    return new_image
@njit()
def uniform_filter_without_zero(image, size):
    shape = np.shape(image)
    new_image = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            counter = 0
            sum = 0
            for k in range(max(0, i-size), min(shape[0]-1, i+size+1)):
                for l in range(max(0, j-size), min(shape[1]-1, j+size+1)):
                    if image[k][l] > 0.001:
                        counter += 1
                        sum += image[k][l]
            if counter > 0:
                new_image[i][j] = sum/counter
    return new_image

@njit()
def gaussian_filter_with_depth_factor(depth_image, size, sigma=None):
    height, width = np.shape(depth_image)
    factor_x = math.tan(viewing_angle_x/2) * 2 / width
    factor_y = math.tan(viewing_angle_y/2) * 2 / height
    if sigma == None:
        sigma = np.asarray([0, 0.0001])
    new_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            d = depth_image[i][j]
            if d < 0.00001:
                continue
            weights = []
            sum = 0
            distance_factor_x = d * factor_x
            distance_factor_y = d * factor_y
            for k in range(max(0, i - size), min(height - 1, i + size + 1)):
                for l in range(max(0, j - size), min(width - 1, j + size + 1)):
                    if depth_image[k][l] > 0.00001 and (k != i or j != l):
                        percentage = abs(k-i)/(abs(k-i) + abs(l-j))
                        depth_normalizer = percentage * distance_factor_y + (1-percentage) * distance_factor_x
                        difference = math.sqrt((k-i)**2 + (l-j)**2) * sigma[0] +\
                                     ((depth_image[k][l] - d)/depth_normalizer)**2 * sigma[1]
                        weight = np.exp(-difference)
                        weights.append(weight)
                        sum += weight*depth_image[k][l]
            if len(weights) > 0:
                weight = 1 * np.max(np.asarray(weights))
                weights.append(weight)
                sum += d * weight
                weights = np.sum(np.asarray(weights))
                new_image[i][j] = sum/weights
    return new_image