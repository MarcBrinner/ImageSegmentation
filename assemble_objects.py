import numpy as np
from scipy.spatial.transform import Rotation
from numba import njit

@njit()
def determine_neighbors(index_image, segment_count):
    height, width = np.shape(index_image)
    neighbors = []
    for i in range(segment_count+1):
        neighbors.append([0])
    for y in range(height):
        for x in range(width):
            index = index_image[y][x]
            if y < height - 1:
                other_index = index_image[y + 1][x]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)
            if x < width - 1:
                other_index = index_image[y][x + 1]
                neighbors[index].append(other_index)
                neighbors[other_index].append(index)

    for i in range(len(neighbors)):
        new_list = list(set(neighbors[i]))
        if i in new_list:
            new_list.remove(i)
        neighbors[i] = new_list

    return neighbors

def angles_to_normals(angles):
    n, _ = np.shape(angles)
    normals = np.zeros((n, 3))
    axis_1 = np.asarray([1.0, 0.0, 0.0])
    axis_2 = np.asarray([0.0, 1.0, 0.0])
    middle_vector = np.asarray([0, 0, -1])
    for i in range(n):
        vec = middle_vector.copy()
        rot = Rotation.from_rotvec(angles[i][0] * axis_1)
        vec = rot.apply(vec)
        rot = Rotation.from_rotvec(angles[i][1] * axis_2)
        vec = rot.apply(vec)
        normals[i] = vec
    return normals

@njit()
def calc_average_normal_and_position(angle_image, index_image, number_of_surfaces):
    height, width = np.shape(index_image)
    counter = np.zeros(number_of_surfaces)
    angles = np.zeros((number_of_surfaces, 2))
    positions = np.zeros((number_of_surfaces, 2))
    for y in range(height):
        for x in range(width):
            surface = index_image[y][x]
            counter[surface] += 1
            angles[surface] += angle_image[y][x]
            positions[surface] += np.asarray([x, y])
    angles = angles/counter
    positions = positions/counter
    return angles, positions

def determine_centroid(depth_image, positions):
    centroids = np.zeros((len(positions), 3))
    for i, pos in enumerate(positions):
        discrete_y = int(pos[1])
        discrete_x = int(pos[0])
        centroids[i] = np.asarray([discrete_x, discrete_y, depth_image[discrete_y][discrete_x]])
    return centroids

def determine_convexity(normals, centroids, neighbors, number_of_surfaces):
    convexity = np.zeros((number_of_surfaces, number_of_surfaces))
    for i in range(number_of_surfaces):
        for j in range(i+1, number_of_surfaces):
            if j not in neighbors[i]:
                convexity[i][j] = 0
                continue
            diff = centroids[i] - centroids[j]
            v1 = np.dot(diff, normals[i])
            v2 = np.dot(diff, normals[j])
            if v1 - v2 > 0:
                convexity[i][j] = 1
                convexity[j][i] = 1

