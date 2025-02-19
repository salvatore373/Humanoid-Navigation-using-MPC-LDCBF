from enum import Enum

import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.Utils.obstacles import generate_obstacles

import matplotlib.pyplot as plt
class Scenario(Enum):
    CROWDED = 0
    CROWDED_START = 1
    CROWDED_END = 2
    START_CLOSE_TO_OBSTACLE = 3
    END_CLOSE_TO_OBSTACLE = 4
    HORIZONTAL_WALL = 5
    VERTICAL_SLALOM = 6
    MAZE = 7
    EMPTY = 8
    FEW_OBSTACLES = 9


def load_scenario(scenario, start, goal,
                  num_max_obstacles=5, min_distance=2.0,
                  delta=1.0, range_x=None, range_y=None):
    """
    Returns the list of random obstacles and if MAZE scenario is selected, also the initial and final position.

    :param scenario: selected scenario
    :param start: initial (x, y) coordinates
    :param goal: final (x, y) coordinates
    :param num_max_obstacles: maximum number of obstacles to generate
    :param delta: minimum distance between each obstacle

    # === for auto range generation ===
    :param min_distance: minimum distance from objectives

    # === for imposed range ===
    :param range_x: custom range of x coordinates
    :param range_y: custom range of y coordinates
    """
    obstacles = None

    if scenario == Scenario.CROWDED:
        dist_factor = min_distance

        min_x = min(start[0]+dist_factor, goal[0]-dist_factor)
        max_x = max(start[0]+dist_factor, goal[0]-dist_factor)
        min_y = min(start[1]+dist_factor, goal[1]-dist_factor)
        max_y = max(start[1]+dist_factor, goal[1]-dist_factor)

        obstacles = generate_obstacles(
            start=start,
            goal=goal,
            num_obstacles=num_max_obstacles,
            x_range=(min_x, max_x) if range_x is None else range_x,
            y_range=(min_y, max_y) if range_y is None else range_y,
            delta=delta
        )
    if scenario == Scenario.CROWDED_START:
        dist_factor = min_distance

        min_x = min(start[0]-dist_factor, start[0]+dist_factor)
        max_x = max(start[0]-dist_factor, start[0]+dist_factor)
        min_y = min(start[1]+dist_factor, start[1]-dist_factor)
        max_y = max(start[1]+dist_factor, start[1]-dist_factor)

        obstacles = generate_obstacles(
            start=start,
            goal=goal,
            num_obstacles=num_max_obstacles,
            x_range=(min_x, max_x) if range_x is None else range_x,
            y_range=(min_y, max_y) if range_y is None else range_y,
            delta=delta
        )
    if scenario == Scenario.CROWDED_END:
        dist_factor = min_distance

        min_x = min(goal[0]-dist_factor, goal[0]+dist_factor)
        max_x = max(goal[0]-dist_factor, goal[0]+dist_factor)
        min_y = min(goal[1]+dist_factor, goal[1]-dist_factor)
        max_y = max(goal[1]+dist_factor, goal[1]-dist_factor)

        obstacles = generate_obstacles(
            start=start,
            goal=goal,
            num_obstacles=num_max_obstacles,
            x_range=(min_x, max_x) if range_x is None else range_x,
            y_range=(min_y, max_y) if range_y is None else range_y,
            delta=delta
        )
    if scenario == Scenario.START_CLOSE_TO_OBSTACLE:
        obstacles = [
            ConvexHull(np.array([
                [start[0] + 0.1, -3],
                [start[0] + 0.1, 3],
                [start[0] + 0.3, 3],
                [start[0] + 0.3, -3]
            ]))
        ]
    if scenario == Scenario.END_CLOSE_TO_OBSTACLE:
        obstacles = [
            ConvexHull(np.array([
                [goal[0] + 0.1, -3],
                [goal[0] + 0.1, 3],
                [goal[0] + 0.3, 3],
                [goal[0] + 0.3, -3]
            ]))
        ]
    if scenario == Scenario.HORIZONTAL_WALL:
        obstacles = [
            ConvexHull(np.array([[1, -10], [1, 10], [3, 10], [3, -10]]))
        ]
    if scenario == Scenario.VERTICAL_SLALOM:
        obstacles = [
            ConvexHull(np.array([[1, -1], [1, 10], [2, 10], [2, -1]])),
            ConvexHull(np.array([[3, 1], [3, -10], [4, -10], [4, 1]]))
        ]
    if scenario == Scenario.MAZE:
        start = (0.5, 0.5)
        goal = (5.5, 5.5)
        obstacles = [
            ConvexHull(np.array([[-1, 0], [6, 0],
                                 [-1,-1], [6, -1]])), # low wall
            
            ConvexHull(np.array([[-1, 7], [7, 7],
                                 [-1, 6], [7, 6]])), # high wall
            
            ConvexHull(np.array([[-1, -1], [0, -1],
                                 [-1, 6], [0, 6]])), # left wall
            
            ConvexHull(np.array([[7, -1], [6, -1],
                                 [7, 6], [6, 6]])), # left wall
            
            ConvexHull(np.array([[4, 0], [6, 0],
                                 [4, 2], [6, 2]])), # low_right_block
            
            ConvexHull(np.array([[0, 1], [1, 1],
                                 [0, 6], [1, 6]])), # upper_left_block
            
            ConvexHull(np.array([[1, 1], [3, 1],
                                 [1, 2], [3, 2]])), # low_left_block
            
            ConvexHull(np.array([[2, 3], [5, 3],
                                 [2, 4], [5, 4]])), # mid_block_1
            
            ConvexHull(np.array([[4, 4], [5, 4],
                                 [4, 5], [5, 5]])), # mid_block_2
            
            ConvexHull(np.array([[1, 5], [3, 5],
                                 [1, 6], [3, 6]])), # upper_medium
        ]
    if scenario == Scenario.FEW_OBSTACLES:
        obstacles = [
            ConvexHull(np.array([[3, 2], [5, 4], [2, 2], [2, 4]])),
            ConvexHull(np.array([[4, 1], [5, 0.5], [7, 3], [6, 2.5]]))
        ]
    if scenario == Scenario.EMPTY:
        obstacles = []

    return start, goal, obstacles

if __name__ == "__main__":
         
    scenario = load_scenario(Scenario.MAZE)
    fig, ax = plt.subplots()
    
    start = scenario[0]
    goal  = scenario[1]
    obstacles = scenario[2]
    
    # Plot each obstacle (assuming obstacles are ConvexHull objects)
    for obs in obstacles:
        # Get the vertices in order
        vertices = obs.vertices
        points = obs.points[vertices]
        # Close the polygon by appending the first point to the end
        points = np.vstack([points, points[0]])
        ax.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
        ax.fill(points[:, 0], points[:, 1], color='black')
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Maze")
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()