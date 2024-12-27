import math
import numpy as np
from matplotlib import pyplot as plt

LIDAR_RANGE = 1

def compute_lidar_readings(position, lines, lidar_range, resolution=360):
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

        nearest_point = None
        min_distance = lidar_range

        # for each obstacle
        for line in lines:
            p1, p2 = line
            # get intersection between current ray and obstacles
            intersection = compute_intersection((x, y), ray_end, p1, p2)

            if intersection:
                # euclidean distance from intersection and position
                distance = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip((x, y), intersection)))

                if distance <= lidar_range and distance < min_distance:
                    nearest_point = intersection
                    min_distance = distance

        detected_points.append(nearest_point)

    return detected_points


def compute_intersection(ray_start, ray_end, line_start, line_end):
    x1, y1 = ray_start
    x2, y2 = ray_end
    x3, y3 = line_start
    x4, y4 = line_end

    """
        THEORETIC RECALL: that a line from (x_1, y_1) to (x_2, y_2) can be written
        in a parametric form as (De Luca style)
        
            (x, y) = (x_1, y_1) + t * (x_2 - x_1, y_2 - y_1)
                   = (x_1 + t*(x_2 - x_1),  y_1 + t*(y_2 - y_1))
            
        to find the insersection of those two lines, we need to seek when:
        
            1. the x coord of the first is equal to the x of the second
                x_1 + t*(x_2 - x_1) = x_3 + u*(x_4 - x_3) 
            2. the y coord of the first is equal to the y of the second
                y_1 + t*(y_2 - y_1) = y_3 + u*(y_4 - y_3)
            
        we have a system of 2 equations and 2 unknowns, so we can compute t and u
        
        (in the following code, the second is parametrized using 'u' 
        while the first using 't')
    """
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # parallel/coincident lines
    if denom == 0: return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return intersection_x, intersection_y

    return None


# Example Usage
if __name__ == "__main__":
    # LiDAR 2d position
    lidar_position = (0, 0)

    # obstacles (i.e. lines)
    obstacle_lines = [
        ((0.7, -1), (1.4, 1)),
        ((-0.9, 0.7), (0.9, 0.2)),
        ((-0.5, -0.5), (-0.5, 0.5))
    ]

    # get LiDAR readings
    readings = compute_lidar_readings(lidar_position, obstacle_lines, lidar_range=LIDAR_RANGE, resolution=90)

    # ===== PLOTTING PHASE =====
    # LiDAR
    plt.scatter(lidar_position[0], lidar_position[1], color="red", label="LiDAR Position")

    # obstacles
    for line in obstacle_lines:
        x_coords, y_coords = zip(*line)
        plt.plot(x_coords, y_coords, color="blue", label="Obstacle")

    # readings
    xs = []
    ys = []
    for point in readings:
        # ignore None points (i.e. no obstacles)
        if point:
            xs.append(point[0])
            ys.append(point[1])

    # LiDAR range
    robot = plt.Circle((lidar_position[0], lidar_position[1]), radius=LIDAR_RANGE, color='tomato', label='LiDAR range', fill=False, linewidth=2, alpha=1.0)
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
