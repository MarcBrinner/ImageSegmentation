import hdbscan
from find_planes import *
from sklearn.cluster import DBSCAN

def create_datasets(normals, surfaces, number_of_surfaces):
    points = {}
    for i in range(number_of_surfaces):
        points[i] = []

    height, width = np.shape(surfaces)
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            index = surfaces[y][x]
            if index == 0:
                continue
            normal = normals[y][x]
            point = [x/width, y/height, normal[0], normal[1]]
            points[index].append(point)
    return [np.asarray(p) for p in points.values()]

def cluster_points(points):
    results = []
    index = 0
    clustering = hdbscan.HDBSCAN(min_cluster_size=30, cluster_selection_epsilon=0.02, core_dist_n_jobs=10)
    for p in points:
        print(index)
        index += 1
        if len(p) < 10:
            results.append([])
            continue
        clustering.fit(np.asarray(p))
        results.append(clustering.labels_)
    return results

def create_new_surface_image(points, clustering_results, height, width):
    new_surface_image = np.zeros((height, width))
    index = 1
    for i in range(len(points)):
        label_mapping = {-1: 0}
        if len(points[i]) < 10:
            continue
        for j in range(int(np.max(clustering_results[i]))+1):
            label_mapping[j] = index
            index += 1
        for label, point in zip(clustering_results[i], points[i]):
            x = int(np.round(point[0] * width))
            y = int(np.round(point[1] * height))
            new_surface_image[y][x] = label_mapping[label]
    return new_surface_image

def fill_values(new_surface_image):
    surfaces = np.zeros(np.shape(new_surface_image))
    height, width = np.shape(new_surface_image)
    for y in range(1, height-1):
        for x in range(1, width-1):
            if new_surface_image[y][x] == 0:
                counts = {}
                for y2 in range(y-1, y+2):
                    for x2 in range(x-1, x+2):
                        index = new_surface_image[y2][x2]
                        try:
                            counts[index] += 1
                        except:
                            counts[index] = 1
                best_value = -1
                best_count = 0
                for v, c in counts.items():
                    if c > best_count and v != 0:
                        best_value = v
                surfaces[y][x] = best_value
            else:
                surfaces[y][x] = new_surface_image[y][x]
    return surfaces

def find_smooth_surfaces_new(depth_image, log_depth_image):
    depth_edges = find_edges.find_edges_from_depth_image(depth_image)
    depth_edges = do_iteration_2(depth_edges, 3, 2)
    indexed_surfaces, segment_count = measure.label(depth_edges, background=1, return_num=True)
    segment_count += 1
    plot_surface_image(indexed_surfaces)

    normal_vectors = calculate_normals_as_angles_final(depth_image)
    print("Normals calculated.")
    dataset = create_datasets(normal_vectors, indexed_surfaces, segment_count)
    print("Dataset created.")
    clusters = cluster_points(dataset)
    print("Clusters found.")
    surfaces = create_new_surface_image(dataset, clusters, *np.shape(depth_image))
    print("New surfaces indexed.")
    surfaces = fill_values(surfaces)
    print("Values filled.")
    plot_surface_image(surfaces)

if __name__ == '__main__':
    images = load_image(110)
    find_smooth_surfaces_new(images[0], None)