import random
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

# generate ONE random convex polygon
def generate_random_convex_polygon(num_points, x_range, y_range):
    points = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]
    hull = ConvexHull(points)
    return [points[v] for v in hull.vertices]


# compute 2D cross product between OA and OB
# positive value --> B is ccw to A around O
# negative value --> B in cw to A around O
# zero value --> O, A and B are collinear
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# check if a point is inside a convex polygon
# if a point lies always on the same side of all edges of a
# convex polygon then it is inside it
# to check it, for each edge (V_i, V_i+1) compute cross product
# if all the results are negative, or positive, the point is inside
def is_point_inside_polygon(point, polygon):
    n = len(polygon)
    for i in range(n):
        if cross(polygon[i], polygon[(i + 1) % n], point) < 0:
            return False
    return True



# shortest distance from a point P to a segment VW
def point_to_segment_distance(p, v, w):
    # segment len
    l2 = (w[0] - v[0])**2 + (w[1] - v[1])**2

    # avoid zero division (I think can be deleted)
    if l2 == 0: return np.hypot(p[0] - v[0], p[1] - v[1])

    # similar logic to CBF constraint, compute the projection t of P in VW
    # t = (P-W)*(W-V)/norm(W-V)^2
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))

    # t can be seen as a percentage where the point lies in the egment
    projection = (v[0]+t*(w[0]-v[0]), v[1]+t*(w[1]-v[1]))
    return np.hypot(p[0]-projection[0], p[1]-projection[1])


# compute the distance from a point to a polygon
# by computing the distance from each edge and taking
# the minimum value
def point_to_polygon_distance(point, polygon):
    return min(
        point_to_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )

# if three points P, Q, R are arranged in a counter-clockwise order
def ccw(p, q, r):
    return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])


# test if the outmost points of one segment AB straddle the other segment CD
def segment_intersection(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

# check if a segment intersect with another polygon by checking
# for each edge of the polygon, if it intersects the given segment
def segment_intersects_polygon(segment, polygon):
    intersections = []
    for i in range(len(polygon)):
        if segment_intersection(segment[0], segment[1], polygon[i], polygon[(i + 1) % len(polygon)]):
            intersections.append((polygon[i], polygon[(i + 1) % len(polygon)]))
    return intersections


# checks if a polygon intersects with any polygon in a list
def polygon_intersect_with_list_of_polygons(polygon, polygons):
    for p in polygons:
        if polygons_intersect(p, polygon):
            return True
    return False


def line_polygon_intersection(line, polygon):
    """
    Computes the explicit intersection points between a line and a polygon.

    Args:
        line: A tuple of two points (p1, p2) representing the line segment.
        polygon: A list of points [(x1, y1), (x2, y2), ...] representing the polygon.

    Returns:
        A list of intersection points.
    """

    def compute_intersection(p1, p2, q1, q2):
        """Helper function to compute the intersection of two line segments (p1p2 and q1q2)."""
        a1, b1 = p1, p2
        a2, b2 = q1, q2

        denom = (b2[1] - a2[1]) * (b1[0] - a1[0]) - (b2[0] - a2[0]) * (b1[1] - a1[1])
        if denom == 0:  # Parallel lines
            return None

        ua = ((b2[0] - a2[0]) * (a1[1] - a2[1]) - (b2[1] - a2[1]) * (a1[0] - a2[0])) / denom
        ub = ((b1[0] - a1[0]) * (a1[1] - a2[1]) - (b1[1] - a1[1]) * (a1[0] - a2[0])) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:  # Intersection within the segment bounds
            x = a1[0] + ua * (b1[0] - a1[0])
            y = a1[1] + ua * (b1[1] - a1[1])
            return (x, y)
        return None

    intersections = []
    for i in range(len(polygon)):
        edge = (polygon[i], polygon[(i + 1) % len(polygon)])
        intersection = compute_intersection(line[0], line[1], edge[0], edge[1])
        if intersection:
            intersections.append(intersection)

    return intersections


# check if two polygons intersect by checking if any edge of one
# polygon intersects any edge of the other
def polygons_intersect(polygon1, polygon2):
    for i in range(len(polygon1)):
        segment1 = (polygon1[i], polygon1[(i + 1) % len(polygon1)])
        for j in range(len(polygon2)):
            segment2 = (polygon2[j], polygon2[(j + 1) % len(polygon2)])
            # FIXME: maybe the segment check can be fully replaced by the point check
            if segment_intersects_polygon(segment1, [segment2[0], segment2[1]]) or \
                is_point_inside_polygon(polygon1[i], polygon2) or \
                is_point_inside_polygon(polygon2[j], polygon1):
                return True
    return False

# plotting funtion
def plot_polygon(polygon, color='blue', label=None):
    # to 'close' the polygon, we need to append the first vertex
    # to the end of the vertex list
    polygon = np.append(polygon, [polygon[0]], axis=0)
    plt.plot(polygon[:, 0], polygon[:, 1], '-', color=color, label=label)
    plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.2, color=color)


# Generate small obstacles randomly placed around the map
def generate_polygons(start, goal, num_obstacles, num_points, x_range, y_range):
    polygons = []
    attempts = 0
    max_attempts = 500

    while len(polygons) < num_obstacles and attempts < max_attempts:
        attempts += 1

        x_center = random.uniform(*x_range)
        y_center = random.uniform(*y_range)
        poly = generate_random_convex_polygon(num_points, (x_center - 1, x_center + 1), (y_center - 1, y_center + 1))

        if is_point_inside_polygon(start, poly):
            continue

        if is_point_inside_polygon(goal, poly):
            continue

        if any(polygons_intersect(poly, p) for p in polygons):
            continue

        polygons.append(poly)


    return polygons





# ===== PLOTTING OBSTACLES FROM OTHER FILES =====
def generate_obstacles(start, goal, num_obstacles=10, num_points=5, x_range=(-10, 10), y_range=(-10, 10)):
    # generate the polygons
    polygons = generate_polygons(start, goal, num_obstacles, num_points, x_range, y_range)

    # Plot obstacles
    for i, obs in enumerate(polygons):
        plot_polygon(obs, color='tomato', label='obstacle' if i == 0 else None)

    return polygons







# ===== PLOTTING OBSTACLES FROM CURRENT FILE =====
if __name__ == "__main__":
    # Define a segment
    segment = ((2, 2), (6, 8))

    # Generate a test polygon
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]  # Square

    # Compute intersection points
    intersection_points = line_polygon_intersection(segment, polygon)

    # Print results
    print("Polygon vertices:", polygon)
    print("Segment:", segment)
    print("Intersection points:", intersection_points)

    # Plot the segment and polygon
    plot_polygon(polygon, color='blue', label='Polygon')
    plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r-', label='Segment')
    if intersection_points:
        x, y = zip(*intersection_points)
        plt.scatter(x, y, color='green', label='Intersection points')
    plt.legend()
    plt.show()




# ===== TESTING FUNCTION =====
# if __name__ == "__main__":
#     # Generate a random convex polygon
#     polygon = generate_random_convex_polygon(10, (0, 10), (0, 10))
#     print("Polygon vertices:", polygon)
#
#     # Check if a point is inside the polygon
#     point = (5, 5)
#     inside = is_point_inside_polygon(point, polygon)
#     print("Point inside polygon:", inside)
#
#     # Compute distance from a point to the polygon
#     distance = point_to_polygon_distance(point, polygon)
#     print("Distance from point to polygon:", distance)
#
#     # Compute intersection of a segment with the polygon
#     segment = ((2, 2), (8, 8))
#     intersections = segment_intersects_polygon(segment, polygon)
#     print("Segment intersections with polygon:", intersections)
#
#     # Generate another random convex polygon
#     polygon2 = generate_random_convex_polygon(10, (5, 15), (5, 15))
#     print("Second polygon vertices:", polygon2)
#
#     # Check if the two polygons intersect
#     intersect = polygons_intersect(polygon, polygon2)
#     print("Polygons intersect:", intersect)
#
#     # Plot the polygons
#     plot_polygon(polygon)
#     plot_polygon(polygon2)
#     plt.show()
