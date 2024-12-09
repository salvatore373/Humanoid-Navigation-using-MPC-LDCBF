import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sympy import Point, Polygon
import random


def CreateConvexObstacle(n):
    mu_x = random.uniform(1, 20)
    mu_y = random.uniform(1, 20)
    std = 0.7
    rnd_points_x = np.random.normal(mu_x, std, size=(n)) # random x coordinate
    rnd_points_y = np.random.normal(mu_y, std, size=(n)) # random y coordinate
    rnd_points = np.stack((rnd_points_x, rnd_points_y), axis=1) # random points
    hull = ConvexHull(rnd_points) # create a convex hull that contains the n random points
    vertices=[]
    for simplex in hull.simplices:
        vertices+=list(simplex)
    return rnd_points[hull.vertices]

num_obs = 20
poly_list = []
for j in range(0, num_obs):
    n = np.random.randint(3, 10) # num of random_points inside an obstacle
    vertices = CreateConvexObstacle(n)
    poly_list.append(Polygon(*map(Point, vertices)))
    plt.fill(vertices[:,0], vertices[:,1], 'k', alpha=0.3)

plt.show()
