import math
import numpy as np
from matplotlib import pyplot as plt
from sympy import Point, Ray, intersection

from Source.obstacles import GenerateObstacles

LIDAR_RANGE = 2.0
RESOLUTION = 360

def get_closest_point(points_list, point):
    closest_point = None
    distance = LIDAR_RANGE

    for p in points_list:
        curr = float(p.distance(point))
        if curr < distance:
            distance = curr
            closest_point = p

    return closest_point, distance


def compute_lidar_readings(position, obstacles, lidar_range, resolution=360):
    x, y = position
    point = Point(x, y)
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

        ray = Ray(point, Point(ray_end[0], ray_end[1]))

        nearest_point = None
        min_distance = lidar_range

        # for each obstacle
        for o in obstacles:
            # get intersection between current ray and obstacles
            intersections = ray.intersection(o)

            if len(intersections) == 0: continue

            # intersections between a ray and a polygon may be more than 1
            closest, distance = get_closest_point(intersections, point)

            if distance <= lidar_range and distance < min_distance:
                nearest_point = closest
                min_distance = distance

        detected_points.append(nearest_point)

    return detected_points



# Example Usage
if __name__ == "__main__":
    # LiDAR 2d position
    lidar_position = [0, 0]

    # obstacles (i.e. polygons)
    obstacles = GenerateObstacles(
        start=np.array(lidar_position),
        goal=np.array([1, 1]),
        num_obs=5,
        range_min=-3,
        range_max=3,
    )

    # get LiDAR readings
    readings = compute_lidar_readings(lidar_position, obstacles, lidar_range=LIDAR_RANGE, resolution=RESOLUTION)

    # ===== PLOTTING PHASE =====
    # LiDAR
    plt.scatter(lidar_position[0], lidar_position[1], color="red", label="LiDAR Position")

    # readings
    xs = []
    ys = []
    for point in readings:
        # ignore None points (i.e. no obstacles)
        if point:
            xs.append(point[0])
            ys.append(point[1])

    # LiDAR range
    robot = plt.Circle((lidar_position[0], lidar_position[1]), radius=LIDAR_RANGE, color='tomato', label='LiDAR range',
                       fill=False, linewidth=2, alpha=1.0)
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(robot)

    # utils
    plt.scatter(xs, ys, color="green", label="Detected Points")
    plt.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("LiDAR Simulation")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
