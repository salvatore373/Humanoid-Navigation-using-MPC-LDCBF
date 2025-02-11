from enum import Enum

import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.Utils.obstacles import generate_obstacles


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


def load_scenario(scenario, start=(0, 0), goal=(5, 0)):
    start, goal, obstacles = start, goal, None

    if scenario == Scenario.CROWDED:
        start = (0, 0)
        goal = (5, 5)
        obstacles = generate_obstacles(
            start=start,
            goal=goal,
            num_obstacles=10,  # 100?
            x_range=(0.1, 5),
            y_range=(0.1, 5)
        )
    if scenario == Scenario.CROWDED_START:
        start = (0, 0)
        goal = (5, 5)
        obstacles = generate_obstacles(
            start=start,
            goal=goal,
            num_obstacles=10,
            x_range=(0.1, 2),
            y_range=(0.1, 2)
        )
    if scenario == Scenario.CROWDED_END:
        start = (0, 0)
        goal = (5, 5)
        obstacles = generate_obstacles(
            start=start,
            goal=goal,
            num_obstacles=10,
            x_range=(3, 4.9),
            y_range=(3, 4.9)
        )
    if scenario == Scenario.START_CLOSE_TO_OBSTACLE:
        start = (0, 0)
        goal = (5, 0)
        obstacles = [
            ConvexHull(np.array([[0.1, -3], [0.1, 3], [1, 3], [1, -3]]))
        ]
    if scenario == Scenario.END_CLOSE_TO_OBSTACLE:
        start = (0, 0)
        goal = (5, 0)
        obstacles = [
            ConvexHull(np.array([[4.9, -3], [4.9, 3], [4, 3], [4, -3]]))
        ]
    if scenario == Scenario.HORIZONTAL_WALL:
        start = (0, 0)
        goal = (5, 0)
        obstacles = [
            ConvexHull(np.array([[1, -10], [1, 10], [3, 10], [3, -10]]))
        ]
    if scenario == Scenario.VERTICAL_SLALOM:
        start = (0, 0)
        goal = (5, 0)
        obstacles = [
            ConvexHull(np.array([[1, -1], [1, 10], [2, 10], [2, -1]])),
            ConvexHull(np.array([[3, 1], [3, -10], [4, -10], [4, 1]]))
        ]
    if scenario == Scenario.MAZE:
        raise NotImplementedError()
    
    if scenario == Scenario.EMPTY:
        start = (0,0)
        goal = (5,5)
        obstacles = []
        
    if scenario == Scenario.FEW_OBSTACLES:
        start = (0,0)
        goal = (5,5)
        obstacles = [
            ConvexHull(np.array([[3, 2], [5, 4], [2, 2], [2, 4]])),
            ConvexHull(np.array([[4, 1], [5, 0.5], [7, 3], [6, 2.5]]))
        ]

    return start, goal, obstacles