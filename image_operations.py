import numpy as np
import load_images
import math
from plot_image import *
from image_filters import *
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

def convert_depth_image(image, plot=False):
    new_image = np.log10(image, where=image != 0)

    min = np.min(new_image, where=image != 0, initial=np.inf)
    max = np.max(new_image, where=image != 0, initial=-np.inf)

    new_image = new_image - min
    new_image = new_image / (max-min)
    new_image = (1 - new_image) * 0.85 + 0.15
    new_image[image == 0] = 0

    if plot:
        plot_array(new_image*255)
    return new_image

@njit()
def calculate_curvature_scores(image, neighborhood_value, i, j, factor_array, d, width, height):
    curvature_scores = np.zeros((neighborhood_value * 2 + 1, neighborhood_value * 2 + 1))
    central_point = np.asarray([0, 0, d])
    directions = np.asarray([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    for direction in directions:
        distance_prev_point = 0
        prev_point = None
        prev_score = 0
        for k in range(1, neighborhood_value + 1):
            x_y_difference = direction * k
            current_x_y = x_y_difference * factor_array
            if i + x_y_difference[0] >= height or i + x_y_difference[0] < 0 or j + x_y_difference[1] >= width or j + x_y_difference[1] < 0:
                continue
            current_point = np.asarray([current_x_y[0], current_x_y[1], image[i + x_y_difference[0]][j + x_y_difference[1]]])
            distance_current_point = np.sqrt(np.sum(np.square(current_point - central_point)))
            if k == 1:
                distance_prev_point = distance_current_point
                prev_point = current_point
                continue
            distance_current_points = np.sqrt(np.sum(np.square(current_point - prev_point)))
            score = (distance_current_points + distance_prev_point) / distance_current_point - 1 + prev_score
            curvature_scores[neighborhood_value + x_y_difference[0]][neighborhood_value + x_y_difference[1]] = score
            prev_score = score
            distance_prev_point = distance_current_point
            prev_point = current_point

    for direction in directions:
        curvature_scores[neighborhood_value + direction[0]][neighborhood_value + direction[1]] =\
            0.5 * curvature_scores[neighborhood_value + 2*direction[0]][neighborhood_value + 2*direction[1]]

    for level in range(2, neighborhood_value+1):
        for k in range(8):
            direction_1 = level * directions[k-1] + neighborhood_value
            direction_2 = level * directions[k] + neighborhood_value
            if direction_1[0] != direction_2[0]:
                different_index = 0
                values = [np.asarray([min(direction_1[0], direction_2[0]) + l, direction_1[1]]) for l in range(1, level)]
            else:
                different_index = 1
                values = [np.asarray([direction_1[0], min(direction_1[1], direction_2[1]) + l]) for l in range(1, level)]
            for value in values:
                percentage = abs(direction_1[different_index] - value[different_index])/level
                interpolated_value = percentage * curvature_scores[direction_2[0]][direction_2[1]] +\
                                     (1-percentage) * curvature_scores[direction_1[0]][direction_1[1]]
                curvature_scores[value[0]][value[1]] = interpolated_value
    return curvature_scores


def find_planes(depth_image, normals):
    assignments = np.zeros(np.shape(depth_image), dtype="uint8")
    height, width = np.shape(image)
    for i in range(height):
        for j in range(width):
            if depth_image[j][i] < 0.0001 or assignments[j][i] != 0:
                continue

def determine_normal_vectors_final(depth_image, rgb_image):
    lab = rgb_to_Lab(rgb_image)
    smoothed = gaussian_filter_with_context(depth_image, lab, 3)
    normals = normals_CP(smoothed)
    return normals

def determine_normal_vectors_like_paper(depth_image):
    median_filtered_image = median_filter(depth_image, 3)
    normals = normals_CP(median_filtered_image)
    plot_image = np.asarray(normals * 127.5 + 127.5, dtype="uint8")
    load_images.plot_array(plot_image)
    normals = gaussian_filter(normals, 2, 0.5)
    plot_image = np.asarray(normals * 127.5 + 127.5, dtype="uint8")
    load_images.plot_array(plot_image)
    edges = find_edges_from_normal_image_3(normals, depth_image, alpha=0.99)
    for i in range(5):
        edges = do_iteration(edges, 5)
    plot_image = np.asarray(edges * 255, dtype="uint8")
    load_images.plot_array(plot_image)
    quit()


if __name__ == '__main__':
    image, rgb = load_images.load_image(110)
    determine_normal_vectors_like_paper(image)

    lab = rgb_to_Lab(rgb)
    smoothed = gaussian_filter_with_context(image, lab, 3)

    #image = uniform_filter_without_zero(image, 1)
    #image = normals_CP(image)
    #normals2 = calculate_normals_2(image)
    #normals = calculate_normals_2(image)
    normals = normals_CP(smoothed)
    edges = find_edges_from_normal_image_3(normals, image)
    load_images.plot_array(np.asarray(edges*255, dtype="uint8"))
    quit()
    edges = do_iteration(edges, 5)
    edges = do_iteration(edges, 5)
    edges = do_iteration(edges, 6)
    edges = do_iteration(edges, 6)
    load_images.plot_array(np.asarray(edges*255, dtype="uint8"))
    #quit()
    #normals_im = np.asarray(normals*127.5 + 127.5, dtype="uint8")
    #normals_im_2 = np.asarray(normals*127.5 + 127.5, dtype="uint8")

    #PLT.imshow(rgb)
    #PLT.show()
    #load_images.plot_array(normals_im)
    #load_images.plot_array(normals_im_2)

    quit()
    image = convert_depth_image(image)
    image = calculate_normals(image)
    image = np.asarray(image*127.5 + 127.5, dtype="uint8")
    load_images.plot_array(image)