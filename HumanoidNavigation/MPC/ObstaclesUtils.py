import random

import numpy as np
from scipy.spatial import ConvexHull


class ObstaclesUtils:
    """
    A class to generate obstacles and operate with them.
    """

    @staticmethod
    def generate_random_convex_polygon(num_points: int, x_range: tuple[float, float],
                                       y_range: tuple[float, float]) -> ConvexHull:
        """
        Generates a convex polygon with num_points vertices, that fits the area delimited by x_range and y_range.

        :return: The ConvexHull object that represents the generated convex polygon.
        """
        points = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]
        return ConvexHull(points)

    @staticmethod
    def get_closest_point_and_normal_vector_from_obs(x: np.ndarray, polygon: ConvexHull,
                                                     unitary_normal_vector: bool = False) \
            -> tuple[np.ndarray[(1, 2)], np.ndarray[(2, 2)]]:
        """
        Given a point X and a ConvexHull polygon, it returns the point on the boundary of polygon closest to X and the
        normal vector connecting X to the polygon.

        :param unitary_normal_vector: Whether the returned normal vector should be a unitary vector.
        """
        # Compute the projection of X on the polygon and the closest point on the edge
        closest_point: np.ndarray = None
        min_dist: float = float('inf')
        for visible_facet in polygon.simplices:  # Iterate throught the polygon edges visible from X
            # Get the coords of the endpoints of the edge
            endpoint1, endpoint2 = polygon.points[visible_facet[0]], polygon.points[visible_facet[1]]

            # In comments below assume that A and B are the endpoints of the edge and P~X.
            # Compute the point on AB closest to P
            ep1ToP = x - endpoint1  # Compute vector AP
            ep1ToEp2 = endpoint2 - endpoint1  # Compute vector AB
            # Compute (AP dot AB) / (norm(AB))^2, i.e. the projection of AP on AB as a percentage of the distance from A
            projection_param = np.dot(ep1ToP, ep1ToEp2) / np.power(np.linalg.norm(ep1ToEp2), 2)
            # Make sure that projection_param is in 0, 1 to get a point on AB
            projection_param = max(0, min(1, projection_param))
            # Compute the point C corresponding to the projection of AP on AB
            point_c = endpoint1 + projection_param * ep1ToEp2
            point_c = point_c.reshape(2, 1)

            # Compute the length of the segment from X to C
            dist = np.linalg.norm(point_c - x)

            # While considering many edges of the polygon, take only the C closest to X
            if dist < min_dist:
                closest_point = point_c
                min_dist = dist

        # Define the vector from X to C
        normal_vector = np.array([
            x[0] - closest_point[0],
            x[1] - closest_point[1]
        ]).reshape(2, 1)

        # Make the normal vector unitary if requested
        if unitary_normal_vector:
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

        return closest_point, normal_vector

    @staticmethod
    def transform_obstacle_from_glob_to_loc_coords(obstacle: ConvexHull, transformation_matrix: np.ndarray) \
            -> ConvexHull:
        # Get the vertices of the obstacle
        glob_vertices = obstacle.points
        # Convert all the vertices from global to local coordinates with the given transformation matrix
        loc_vertices = transformation_matrix @ np.insert(glob_vertices.T, 2, 1, axis=0)
        # Create a new obstacle with the new vertices (excluding the last component of the just computed vertices)
        return ConvexHull(loc_vertices[:-1, :].T)
