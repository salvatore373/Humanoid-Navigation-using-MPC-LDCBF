import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sympy import Point, Polygon 

def CreateConvexObstacle(n):
    rnd_points = np.random.rand(n,2) # random_points inside the obstacle
    hull = ConvexHull(rnd_points) # create a convex hull that contains the n random points
    vertices=[]
    for simplex in hull.simplices:
        vertices+=list(simplex)
    return rnd_points[hull.vertices]

num_obs = 4
poly_list = []
for j in range(0, num_obs):
    n = np.random.randint(3, 10) # num of random_points inside an obstacle
    vertices = CreateConvexObstacle(n)
    poly_list.append(Polygon(*map(Point, vertices)))
    plt.fill(vertices[:,0], vertices[:,1], 'k', alpha=0.3)

plt.show()
