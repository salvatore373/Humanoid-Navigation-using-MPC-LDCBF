import random
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

# DISCLAIMER: QUESTO FILE NON È STATO FATTO DA DAMIANO
# damiano non aveva sbatti di andarsi a rispolverare altra geometria
# dopo quella che si è rivisto per il LiDAR

# Function to generate a random convex polygon
def generate_random_convex_polygon(num_points, x_range, y_range):
    points = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]
    hull = ConvexHull(points)
    return [points[v] for v in hull.vertices]

# Helper function to compute cross product
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

# Function to check if a point is inside a convex polygon
def is_point_inside_polygon(point, polygon):
    n = len(polygon)
    for i in range(n):
        if cross(polygon[i], polygon[(i + 1) % n], point) < 0:
            return False
    return True

# Function to compute the distance from a point to a polygon
def point_to_polygon_distance(point, polygon):
    def point_to_segment_distance(p, v, w):
        l2 = (w[0] - v[0])**2 + (w[1] - v[1])**2
        if l2 == 0:
            return np.hypot(p[0] - v[0], p[1] - v[1])
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return np.hypot(p[0] - projection[0], p[1] - projection[1])

    return min(point_to_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))

# Function to compute the intersection between a polygon and a segment
def segment_intersects_polygon(segment, polygon):
    def segment_intersection(a, b, c, d):
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    intersections = []
    for i in range(len(polygon)):
        if segment_intersection(segment[0], segment[1], polygon[i], polygon[(i + 1) % len(polygon)]):
            intersections.append((polygon[i], polygon[(i + 1) % len(polygon)]))

    return intersections

def polygon_intersect_with_list_of_polygons(polygon, polygons):
    for p in polygons:
        if polygons_intersect(p, polygon):
            return True
    return False


# Function to check if two polygons intersect
def polygons_intersect(polygon1, polygon2):
    for i in range(len(polygon1)):
        segment1 = (polygon1[i], polygon1[(i + 1) % len(polygon1)])
        for j in range(len(polygon2)):
            segment2 = (polygon2[j], polygon2[(j + 1) % len(polygon2)])
            if segment_intersects_polygon(segment1, [segment2[0], segment2[1]]):
                return True
    return False

# Function to plot a polygon
def plot_polygon(polygon, point=None, segment=None):
    polygon = np.array(polygon + [polygon[0]])  # Close the polygon by adding the first point at the end
    plt.plot(polygon[:, 0], polygon[:, 1], 'b-', label="Polygon")
    plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.2, color='blue')

    if point:
        plt.plot(point[0], point[1], 'ro', label="Point")

    if segment:
        seg = np.array(segment)
        plt.plot(seg[:, 0], seg[:, 1], 'g--', label="Segment")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()


# Example usage
if __name__ == "__main__":
    # Generate a random convex polygon
    polygon = generate_random_convex_polygon(10, (0, 10), (0, 10))
    print("Polygon vertices:", polygon)

    # Check if a point is inside the polygon
    point = (5, 5)
    inside = is_point_inside_polygon(point, polygon)
    print("Point inside polygon:", inside)

    # Compute distance from a point to the polygon
    distance = point_to_polygon_distance(point, polygon)
    print("Distance from point to polygon:", distance)

    # Compute intersection of a segment with the polygon
    segment = ((2, 2), (8, 8))
    intersections = segment_intersects_polygon(segment, polygon)
    print("Segment intersections with polygon:", intersections)

    # Generate another random convex polygon
    polygon2 = generate_random_convex_polygon(10, (5, 15), (5, 15))
    print("Second polygon vertices:", polygon2)

    # Check if the two polygons intersect
    intersect = polygons_intersect(polygon, polygon2)
    print("Polygons intersect:", intersect)

    # Plot the polygons
    plot_polygon(polygon)
    plot_polygon(polygon2)
    plt.show()
