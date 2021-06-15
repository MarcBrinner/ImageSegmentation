import numpy as np
import open3d as o3d
import os
from PIL import Image

def load_image_as_point_cloud(index):
    files = os.listdir("Data/pcd")
    file = files[index]
    pcd = o3d.io.read_point_cloud(f"Data/pcd/{file}")
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
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

def plot_array(array):
    image = Image.fromarray(array)
    image.show()

if __name__ == '__main__':
    i = Image.open("Data/disparity/test64.png")
    i2 = Image.open("Data/RGB/test64.png")
    i2.show()
    a = np.asarray(i)
    a = np.asarray(255 - a / np.max(a) * 255, dtype="uint8")
    i = Image.fromarray(a)
    i.show()
    quit()
    pcd = load_image_as_point_cloud(0)
    image = extract_depth_image(pcd)
    #print(image)
    #print(np.shape(image))
    plot_array(image)
    image = extract_RGB_image(pcd)
    plot_array(image)
    show_3D_plot(pcd)
