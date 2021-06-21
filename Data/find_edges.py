import math
import numpy as np
from numba import njit
from image_operations import viewing_angle_y, viewing_angle_x

@njit()
def find_edges_from_depth_image(image):
    edge_image = np.zeros(np.shape(image), dtype="uint8")
    height, width = np.shape(image)
    factor_x = math.tan(viewing_angle_x / 2) * 2 / width
    factor_y = math.tan(viewing_angle_y / 2) * 2 / height
    for i in range(1, height-1):
        for j in range(1, width-1):
            d = image[i][j]
            if d < 0.0001:
                continue
            depth_factor_x = factor_x * d
            depth_factor_y = factor_y * d
            alpha = 6
            if np.max(np.abs(np.asarray([image[i-1][j], image[i+1][j]]) - d)) > depth_factor_y*alpha or\
                np.max(np.abs(np.asarray([image[i][j+1], image[i][j-1]]) - d)) > depth_factor_x * alpha:
                edge_image[i][j] = 255
    return edge_image

@njit
def make_2d(arraylist):
    n = len(arraylist)
    k = arraylist[0].shape[0]
    a2d = np.zeros((n, k))
    for i in range(n):
        a2d[i] = arraylist[i]
    return(a2d)

@njit()
def do_iteration(image, number):
    height, width = np.shape(image)
    new_image = np.zeros(np.shape(image))
    for i in range(1, height-1):
        for j in range(1, width-1):
            #print(np.sum(np.asarray([image[i-1][j], image[i+1][j], image[i][j+1], image[i][j-1]])))
            if np.sum(np.asarray([image[i-1][j], image[i+1][j], image[i][j+1], image[i][j-1], image[i][j], image[i-1][j-1], image[i-1][j+1], image[i+1][j-1], image[i+1][j+1],])) >= number:
                new_image[i][j] = 1.0
    return new_image

@njit()
def find_edges_from_normal_image(image, depth_image):
    height, width, _ = np.shape(image)
    edge_image = np.zeros((height, width), dtype="uint8")
    for i in range(1, height-1):
        for j in range(1, width-1):
            if depth_image[i][j] < 0.0001:
                continue
            alpha = math.pi / 25
            vec = image[i][j]
            if np.max(np.arccos(np.dot(make_2d([image[i-1][j], image[i+1][j], image[i][j+1], image[i][j-1]]), vec)) > alpha):
                edge_image[i][j] = 1
    return edge_image

@njit()
def find_edges_from_normal_image_2(image, depth_image):
    height, width, _ = np.shape(image)
    edge_image = np.zeros((height, width), dtype="uint8")
    for i in range(2, height-2, 2):
        print(i)
        for j in range(2, width-2, 2):
            if depth_image[i][j] < 0.0001:
                continue
            alpha = math.pi / 15
            vec = image[i][j]
            counter = 0
            for direction in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                if np.arccos(np.dot(image[i+2*direction[0]][j+2*direction[1]], vec)) > alpha:
                    edge_image[i+direction[0]][j+direction[1]] = 1
                    counter += 1
            if counter >= 1:
                edge_image[i][j] = 1
    return edge_image

@njit()
def find_edges_from_normal_image_3(image, depth_image, alpha=0.85):
    height, width, _ = np.shape(image)
    edge_image = np.zeros((height, width), dtype="float32")
    for i in range(1, height-1):
        for j in range(1, width-1):
            if depth_image[i][j] < 0.0001:
                continue
            vec = image[i][j]
            sum = 0
            for direction in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                sum += np.abs(np.dot(image[i+direction[0]][j+direction[1]], vec))
            sum = sum/8
            if sum < alpha:
                edge_image[i][j] = 1.0
    return edge_image