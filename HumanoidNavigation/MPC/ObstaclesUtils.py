import random

import casadi as cs
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
        generators = np.concatenate((polygon.points, np.expand_dims(x, axis=0)), axis=0)

        hull = ConvexHull(points=generators,
                          qhull_options=f'QG{len(polygon.vertices)}')
        # Compute the projection of X on the polygon and the closest point on the edge
        closest_point: np.ndarray = None
        min_dist: float = float('inf')
        for visible_facet in hull.simplices[hull.good]:  # Iterate throught the polygon edges visible from X
            # Get the coords of the endpoints of the edge
            endpoint1, endpoint2 = hull.points[visible_facet[0]], hull.points[visible_facet[1]]

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

            # Compute the length of the segment from X to C
            dist = np.linalg.norm(point_c - x)

            # While considering many edges of the polygon, take only the C closest to X
            if dist < min_dist:
                closest_point = point_c
                min_dist = dist

        # Define the vector from X to C
        normal_vector = np.array([
            closest_point[0] - x[0],
            closest_point[1] - x[1]
        ]).reshape(2, 1)

        # Make the normal vector unitary if requested
        if unitary_normal_vector:
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

        return closest_point, normal_vector

    @staticmethod
    def get_closest_point_and_normal_vector_from_each_edge(x: np.ndarray[cs.MX], polygon: ConvexHull,
                                                           unitary_normal_vector: bool = False) \
            -> tuple[list[cs.MX], list[cs.MX]]:
        """
        Given a point X and a ConvexHull polygon, for each edge of the polygon it returns the point closest to X and the
        normal vector connecting X to that edge.

        :param x: The 2D vector representing the point X. The components of this vector are casadi variables.
        :param polygon: The polygon from which to compute the closest point and normal vector to X.
        :param unitary_normal_vector: Whether the normal vectors should a unitary vectors.
        :returns: A tuple where the first element is the list of all the closest points to X, and the second one is the
         list of the normal vectors.
        """
        list_c = []
        list_norm_vec = []
        # Compute the projection of X on the polygon and the closest point on the edge
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
            projection_param = cs.fmax(0, cs.fmin(1, projection_param))
            # Compute the point C corresponding to the projection of AP on AB
            point_c = endpoint1 + projection_param * ep1ToEp2

            # Define the vector from X to C
            normal_vector = cs.vertcat(
                point_c[0] - x[0],
                point_c[1] - x[1]
            )
            # Make the normal vector unitary if requested
            if unitary_normal_vector:
                normal_vector = normal_vector / cs.norm_2(normal_vector)

            list_c.append(point_c)
            list_norm_vec.append(normal_vector)

        return list_c, list_norm_vec

    @staticmethod
    def transform_obstacle_from_glob_to_loc_coords(obstacle: ConvexHull, transformation_matrix: np.ndarray) \
            -> ConvexHull:
        # Get the vertices of the obstacle
        glob_vertices = obstacle.points
        # Convert all the vertices from global to local coordinates with the given transformation matrix
        loc_vertices = transformation_matrix @ np.insert(glob_vertices.T, 2, 1, axis=0)
        # Create a new obstacle with the new vertices (excluding the last component of the just computed vertices)
        return ConvexHull(loc_vertices[:-1, :].T)


if __name__ == "__main__":
    # Test: find the point on a polygon closest to X
    from scipy.spatial import convex_hull_plot_2d
    import matplotlib.pyplot as plt

    x = np.array([0.3, 0.6])
    obstacle = ConvexHull(np.array([[0.2, 0.2],
                                    [0.2, 0.4],
                                    [0.4, 0.4],
                                    [0.4, 0.2], ]))

    # c, normal_vec = ObstaclesUtils.get_closest_point_and_normal_vector_from_obs(x, obstacle, unitary_normal_vector=True)
    list_c, list_norm_vec = ObstaclesUtils.get_closest_point_and_normal_vector_from_each_edge(x, obstacle,
                                                                                              unitary_normal_vector=True)

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Plot the XC vector
    # plt.plot([x[0], c[0]], [x[1], c[1]], "c:")
    for c in list_c:
        plt.plot([x[0], c[0]], [x[1], c[1]], "c:")
    # Plot the obstacle
    convex_hull_plot_2d(obstacle, ax=ax)
    plt.xlim(0.18, 1)
    plt.ylim(0.18, 1)
    plt.show()
