import numpy as np
import load_images
import time
from scipy.ndimage.filters import median_filter
from numba import njit

neighborhood_value = 5

def convert_depth_image(image):
    new_image = median_filter(image, size=5)
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
    sigma = np.array((0.15, 0.15, 40.0))
    height, width = np.shape(image)
    for i in range(height):
        #i = 100
        values_y = list(range(max(i - 3, 0), min(i + 4, height - 1)))
        values_x = list(range(0, 3))
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
                if vec[2] < 0:
                    vec = -vec
                vec[2] = 1
                vec = vec/np.linalg.norm(vec)
            #if vec[0] < 0:
            #    vec = -vec
            return_array[i][j] = vec
    return return_array

if __name__ == '__main__':
    image, _ = load_images.load_image(110)
    image = convert_depth_image(image)
    image = calculate_normals(image)
    image = np.asarray(image*127.5 + 127.5, dtype="uint8")
    load_images.plot_array(image)