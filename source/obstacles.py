import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sympy import Point, Polygon
import random

def RandomPoints(min, max, n, std=0.7):
    # use this to sample from a gaussian distribution with random mean between min and max and a std=0.7
    mu_x = random.uniform(min, max)
    mu_y = random.uniform(min, max)
    rnd_x = np.random.normal(mu_x, std, size=(n)) # random x coordinate
    rnd_y = np.random.normal(mu_y, std, size=(n)) # random y coordinate
    rnd_points = np.stack((rnd_x, rnd_y), axis=1) # random points
    return rnd_points

def CreateStartAndGoal(start, goal):
    plt.plot(start[0], start[1], marker="o", color="tomato", label="start")
    plt.plot(goal[0], goal[1], marker="o", color="royalblue", label="end")
    return 0

def CreateConvexObstacle(n, start, goal):

    rnd_points = RandomPoints(min=1, max=20, n=n)
    
    # this can be done more efficiently
    if start in rnd_points: 
        rnd_points.remove(start)
    if goal in rnd_points:
        rnd_points.remove(goal)
    
    hull = ConvexHull(rnd_points) # create a convex hull that contains the n random points
    vertices=[]
    for simplex in hull.simplices:
        vertices+=list(simplex)
    return rnd_points[hull.vertices]

num_obs = 30
poly_list = []
start = RandomPoints(min=0, max=8, n=1)[0]
goal = RandomPoints(min=15, max=21, n=1)[0]
CreateStartAndGoal(start,goal)

for j in range(0, num_obs):
    n = np.random.randint(3, 10) # num of random_points inside an obstacle
    vertices = CreateConvexObstacle(n, start, goal)
    poly_list.append(Polygon(*map(Point, vertices)))
    plt.fill(vertices[:,0], vertices[:,1], 'k', alpha=0.3)

# print(poly_list)
plt.legend()
plt.show()

# TODO 
# delete obstacle if an intersection occurs (not a good idea to merge since it's not s)
# handle the presence of start and goal during the creation of the obstacle (we are not completely robust right now)
# handle the orientation
# differential drive representation
# check if something can be done more efficiently
# test everything with a simple differential drive
# if everything works correctly implement humanoid model