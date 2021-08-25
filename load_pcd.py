import numpy as np
import open3d as o3d
import os
import load_images
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def load_image_as_point_cloud(index):
    files = os.listdir("Data/pcd")
    files = load_images.sorted_alphanumeric(files)
    file = files[index]
    pcd = o3d.io.read_point_cloud(f"Data/pcd/{file}")
    rot = Rotation.from_quat([1, 0, 0, 0]).as_matrix()
    points = np.asarray(pcd.points)
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

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def create_pcd_from_points(points, colors, normals):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    p = np.reshape(points, (640*480, 3))
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.reshape(points, (480*640, 3)))
    # pcd.colors = o3d.utility.Vector3dVector(np.reshape(colors, (480*640, 3)))
    # pcd.normals = o3d.utility.Vector3dVector(np.reshape(normals, (480*640, 3)))
    # o3d.visualization.draw_geometries([pcd])
    p = np.asarray([p[32*i] for i in range(int(640*480/32))])
    ax.scatter(p[:, 0], p[:, 1], p[:, 2]) # plot the point (2,3,4) on the figure
    set_axes_equal(ax)
    plt.show()


if __name__ == '__main__':
    load_image_as_point_cloud(0)
