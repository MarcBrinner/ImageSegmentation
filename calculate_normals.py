from image_filters import *
from image_operations import calculate_curvature_scores
from scipy.spatial.transform import Rotation
from plot_image import plot_array

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
            if np.linalg.norm(vec) < 0.001: continue

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
def calculate_normals_as_angles_final(depth_image, plot_normals=False):
    smoothed = uniform_filter_without_zero(depth_image, 2)
    normals = calculate_normals_cross_product(smoothed)
    angles = normals_to_angles(normals)
    angles = gaussian_filter_with_depth_check(angles, 5, 2, depth_image, np.asarray([math.pi, math.pi]))
    return angles