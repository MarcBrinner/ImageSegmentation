import numpy as np
import load_images
import time
from PIL import ImageCms, Image
import math
from scipy.ndimage.filters import uniform_filter
from numba import njit

neighborhood_value = 10

def rgb_to_Lab(image):
    image = Image.fromarray(image)

    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")

    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    lab_im = np.asarray(ImageCms.applyTransform(image, rgb2lab_transform))
    return lab_im

@njit()
def uniform_filter_without_zero(image, size):
    shape = np.shape(image)
    new_image = np.zeros(shape)
    for i in range(shape[0]):
        print(i)
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
def context_aware_smoothing(depth_image, lab_image, size):
    height, width = np.shape(depth_image)
    factor_x = math.tan(31) * 2 / width
    factor_y = math.tan(31) * 2 / height
    sigma_weights = np.asarray([0.2, 1, 0.01])
    new_image = np.zeros((height, width))
    for i in range(height):
        print(i)
        for j in range(width):
            weights = 0
            sum = 0
            d = depth_image[i][j]
            distance_factor_x = d * factor_x
            distance_factor_y = d * factor_y
            for k in range(max(0, i - size), min(height - 1, i + size + 1)):
                for l in range(max(0, j - size), min(width - 1, j + size + 1)):
                    if depth_image[k][l] != 0:
                        difference = (k-i)**2 * sigma_weights[0] + (l-j)**2 * sigma_weights[0] +\
                                     (np.log10(d)-np.log10(image[k][l]))**2 * sigma_weights[1] +\
                                     np.sum(np.square(np.subtract(lab_image[i][j], lab_image[k][l])))*sigma_weights[2]
                        weight = np.exp(-difference)
                        weights += weight
                        sum += weight*depth_image[k][l]
            if weights > 0:
                new_image[i][j] = sum/weights
    return new_image

def convert_depth_image(image):
    new_image = uniform_filter_without_zero(image, size=1)
    #new_image = median_filter(new_image, size=5)
    #new_image = image.copy()
    new_image = np.log10(new_image, where=new_image != 0)
    min = np.inf
    max = -np.inf
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if new_image[i][j] != 0:
                if new_image[i][j] < min:
                    min = new_image[i][j]
                elif new_image[i][j] > max:
                    max = new_image[i][j]

    new_image = new_image - min
    new_image = new_image / (max-min)
    new_image = (1 - new_image) * 0.85 + 0.15
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if image[i][j] < 1:
                new_image[i][j] = 0

    load_images.plot_array(new_image*255)
    return new_image

@njit()
def calculate_normals(image):
    return_array = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
    square_length = neighborhood_value*2+1
    sigma = np.array((0.005, 0.005, 0))
    height, width = np.shape(image)
    for i in range(height):
        values_y = list(range(max(i - neighborhood_value, 0), min(i + neighborhood_value + 1, height - 1)))
        values_x = list(range(0, neighborhood_value))
        outputs = np.zeros(square_length**2)
        inputs = np.zeros((square_length**2, 3), dtype="float64")
        points = np.zeros((square_length**2, 3), dtype="float64")
        active = np.zeros((square_length**2, 1))
        index_x = 0
        for x in values_x:
            index_y = 0
            for y in values_y:
                inputs[index_x*square_length+index_y] = np.asarray([x, y, 1.0])
                points[index_x*square_length+index_y] = np.asarray([x, y, image[y][x]])
                outputs[index_x*square_length+index_y] = image[y][x]
                active[index_x*square_length+index_y][0] = 1.0
                index_y += 1
            index_x += 1
        fill_index = 3
        for j in range(width):
            new_x = j+neighborhood_value

            if new_x < width:
                index = 0
                for y in values_y:
                    inputs[fill_index*square_length + index] = np.asarray([new_x, y, 1])
                    outputs[fill_index*square_length + index] = image[y][new_x]
                    points[fill_index*square_length + index] = np.asarray([new_x, y, image[y][new_x]])
                    active[fill_index*square_length + index][0] = 1.0
                    index = index + 1
            else:
                for index in range(len(values_y)):
                    active[fill_index*square_length + index] = 0.0
            diff = points - np.asarray([j, i, image[i][j]])

            weights = np.expand_dims(np.exp(-np.sum(diff*sigma*diff, axis=-1)), axis=-1)
            #weights = np.ones((square_length**2, 1))
            mean = np.sum(inputs, axis=0)/np.sum(active)
            mean[2] = 0
            centered_inputs = inputs - mean
            variance = np.sum(np.square(inputs), axis=0)/np.sum(active)
            centered_inputs = centered_inputs/variance
            weighted_inputs = np.reshape(np.square(weights*active).repeat(3).reshape(-1, 3), (square_length**2, 3))*centered_inputs
            vec = np.linalg.lstsq(np.dot(np.transpose(centered_inputs), weighted_inputs), np.dot(np.transpose(weighted_inputs), outputs))[0]
            fill_index = fill_index + 1
            if fill_index >= square_length:
                fill_index = 0
            n = np.linalg.norm(vec)
            if n > 0:
                vec = -vec
                vec[2] = 1
                vec = vec/np.linalg.norm(vec)
            #if vec[0] < 0:
            #    vec = -vec
            return_array[i][j] = vec
    return return_array

@njit()
def normals_CP(image):
    height, width = np.shape(image)
    factor_x = math.tan(31) * 2 / width
    factor_y = math.tan(31) * 2 / height
    return_array = np.zeros((height, width, 3))
    for i in range(1, height-1):
        for j in range(1, width-1):
            depth_dif = image[i-1][j] - image[i+1][j]
            vec_1 = [0, 2 * factor_y * image[i][j], depth_dif]

            depth_dif = image[i][j+1] - image[i][j-1]
            vec_2 = [2 * factor_x * image[i][j], 0, depth_dif]

            cp = np.cross(vec_1, vec_2)
            if cp[2] < 0:
                cp = -cp
            cp = cp/np.linalg.norm(cp)
            return_array[i][j] = cp
    return return_array



if __name__ == '__main__':
    image, rgb = load_images.load_image(110)
    lab = rgb_to_Lab(rgb)
    image = context_aware_smoothing(image, lab, 5)

    #image = uniform_filter_without_zero(image, 1)
    image = normals_CP(image)
    image = np.asarray(image*127.5 + 127.5, dtype="uint8")
    load_images.plot_array(image)
    quit()
    image = convert_depth_image(image)
    image = calculate_normals(image)
    image = np.asarray(image*127.5 + 127.5, dtype="uint8")
    load_images.plot_array(image)