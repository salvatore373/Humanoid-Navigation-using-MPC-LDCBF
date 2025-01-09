import math
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from HumanoidNavigation.Utils.obstacles_no_sympy import line_polygon_intersection, plot_polygon, generate_obstacles, segment_intersects_polygon


def get_closest_point(points_list, point, lidar_range):
    closest_point = None
    distance = lidar_range

    for p in points_list:
        curr = float(np.linalg.norm(np.array(p) - point))
        if curr < distance:
            distance = curr
            closest_point = p

    return closest_point, distance


def compute_lidar_readings(position, obstacles, lidar_range, resolution=360):
    x, y = position
    resolution_step = (2 * math.pi / resolution)

    # get all angles that a lidar must check
    angles = [i * resolution_step for i in range(resolution)]
    detected_points = []

    for angle in angles:
        # ray from actual position to the maximum range
        ray_x = lidar_range * math.cos(angle)
        ray_y = lidar_range * math.sin(angle)

        # ending point of current ray
        ray_end = (x + ray_x, y + ray_y)

        ray = [position, ray_end]

        nearest_point = None
        min_distance = lidar_range

        # for each obstacle
        for o in obstacles:
            # get intersection between current ray and obstacles
            intersections = line_polygon_intersection(ray, o)

            if len(intersections) == 0: continue

            # intersections between a ray and a polygon may be more than 1
            closest, distance = get_closest_point(intersections, position, lidar_range)

            if distance <= lidar_range and distance < min_distance:
                nearest_point = closest
                min_distance = distance

        detected_points.append(nearest_point)

    return detected_points

def create_convex_hull(points):
    """Create a convex polygon from a cluster of points."""
    if len(points) < 3: return None  # Not enough points for a polygon
    hull = ConvexHull(points)
    return points[hull.vertices]

def retrieve_clusters(points, eps=0.2, min_samples=3):
    filtered_points = []
    for point in points:
        if point is not None: filtered_points.append(point)
    filtered_points = np.array(filtered_points)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_points)
    labels = clustering.labels_
    clusters = [filtered_points[labels == i] for i in set(labels) if i != -1]
    return clusters


def build_local_obstacles(clusters):
    obstacle_polygons = []
    for cluster in clusters:
        polygon = create_convex_hull(cluster)
        if polygon is not None:
            polygon = np.append(polygon, [polygon[0]], axis=0)
            obstacle_polygons.append(polygon)
    return obstacle_polygons


def display_lidar_readings(lidar_position, readings, with_range=True, with_grid=True):
    fig = plt.gcf()
    ax = fig.gca()

    # readings
    xs = []
    ys = []
    for point in readings:
        # ignore None points (i.e. no obstacles)
        if point:
            xs.append(point[0])
            ys.append(point[1])

    # LiDAR range
    if with_range:
        range = plt.Circle((lidar_position[0], lidar_position[1]), radius=LIDAR_RANGE, color='tomato',
                           label='LiDAR range', fill=False, linewidth=2, alpha=1.0)
        ax.add_patch(range)

    # utils
    plt.scatter(xs, ys, color="green", label="Detected Points")
    plt.legend() # check if needed to remove
    plt.grid(with_grid)
    plt.axis("equal")




# ===== TO RUN SIMULATION FROM THIS FILE =====
if __name__ == "__main__":
    LIDAR_RANGE = 2.0
    RESOLUTION = 360 # 90 -> a ray each 4 degree

    # LiDAR 2d position
    lidar_position = [0, 0]

    # LiDAR
    plt.scatter(lidar_position[0], lidar_position[1], color="red", label="LiDAR Position")

    # obstacles (i.e. polygons)
    obstacles = generate_obstacles(
        start=np.array(lidar_position),
        goal=np.array([1, 1]),
        num_obstacles=3,
        num_points=5,
        x_range=(-3, 3),
        y_range=(-3, 3)
    )

    # ===== LIDAR READINGS =====
    readings = compute_lidar_readings(lidar_position, obstacles, lidar_range=LIDAR_RANGE, resolution=RESOLUTION)

    # ===== CLUSTER WITH DBSCAN =====
    clusters = retrieve_clusters(readings)

    # ===== OBSTACLES FROM CLUSTERS =====
    local_obstacles = build_local_obstacles(clusters)

    # ===== PLOTTING STUFFS =====
    for obstacle in local_obstacles:
        plot_polygon(obstacle, color='blue', label='Polygon')

    display_lidar_readings(lidar_position, readings, with_range=True, with_grid=True)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("LiDAR Simulation")
    plt.show()
