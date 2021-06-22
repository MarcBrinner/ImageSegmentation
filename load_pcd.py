import numpy as np
import open3d as o3d
import os
import load_images
from scipy.spatial.transform import Rotation

def load_image_as_point_cloud(index):
    files = os.listdir("Data/pcd")
    files = load_images.sorted_alphanumeric(files)
    file = files[index]
    pcd = o3d.io.read_point_cloud(f"Data/pcd/{file}")
    rot = Rotation.from_quat([1, 0, 0, 0]).as_matrix()
    pcd.rotate(rot)
    return pcd

def show_3D_plot(pcd):
    o3d.visualization.draw_geometries([pcd])

def extract_depth_image(pcd):
    w = 640
    h = 480

    min_d = min([x[2] for x in pcd.points])
    max_d = max([x[2] for x in pcd.points])-min_d

    min_x = min([x[0] for x in pcd.points])
    max_x = max([x[0] for x in pcd.points]) - min_x

    min_y = min([x[1] for x in pcd.points])
    max_y = max([x[1] for x in pcd.points]) - min_y

    image = np.zeros((h, w, 3), dtype="uint8")
    for point in pcd.points:
        x = int(np.round((point[0]-min_x)/max_x*(w-1)))
        y = h-1-int(np.round((point[1]-min_y)/max_y*(h-1)))
        d = ((point[2]-min_d)/max_d)*255
        image[y][x] = int(d)
    return image

def extract_depth_image_new(pcd):
    w = 640
    h = 480

    min_d = min([x[2] for x in pcd.points])
    max_d = max([x[2] for x in pcd.points])-min_d

    min_x = min([x[0] for x in pcd.points])
    max_x = max([x[0] for x in pcd.points]) - min_x

    min_y = min([x[1] for x in pcd.points])
    max_y = max([x[1] for x in pcd.points]) - min_y

    rot = Rotation.from_quat([1, 0, 0, 0]).as_matrix()

    image = np.zeros((h, w, 3), dtype="uint8")
    for point in pcd.points:
        x = int(np.round((point[0]-min_x)/max_x*(w-1)))
        y = int(np.round((point[1]-min_y)/max_y*(h-1)))
        d = ((point[2]-min_d)/max_d)*255
        vec = np.asarray([x, y, d])
        new_vec = np.matmul(rot, vec)
        image[int(y)][int(x)] = int(d)
    return image

def extract_RGB_image(pcd):
    w = 640
    h = 480

    min_x = min([x[0] for x in pcd.points])
    max_x = max([x[0] for x in pcd.points]) - min_x

    min_y = min([x[1] for x in pcd.points])
    max_y = max([x[1] for x in pcd.points]) - min_y

    image = np.zeros((h, w, 3), dtype="uint8")
    for point, color in zip(pcd.points, pcd.colors):
        x = int(np.round((point[0]-min_x)/max_x*(w-1)))
        y = h-1-int(np.round((point[1]-min_y)/max_y*(h-1)))
        color = np.asarray([int(color[0]*255), int(color[1]*255), int(color[2]*255)])
        image[y][x] = color
    return image


if __name__ == '__main__':
    pass
