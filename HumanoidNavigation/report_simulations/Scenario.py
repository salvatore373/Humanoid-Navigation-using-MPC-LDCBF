from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.obstacles import generate_obstacles, set_seed


class Scenario(Enum):
    CROWDED = 0
    CROWDED_START = 1
    CROWDED_END = 2
    START_CLOSE_TO_OBSTACLE = 3
    END_CLOSE_TO_OBSTACLE = 4
    HORIZONTAL_WALL = 5
    VERTICAL_SLALOM = 6
    EMPTY = 7
    FEW_OBSTACLES = 8
    CIRCLE_OBSTACLES = 9
    MAIN_PAPER = 10
    BASE = 11
    MAZE_1 = 12
    MAZE_2 = 13

    @staticmethod
    def load_scenario(scenario, start, goal,
                      num_max_obstacles=5, min_distance=2.0,
                      delta=1.0, range_x=None, range_y=None, seed: int = None):
        """
        Returns the list of random obstacles and if MAZE scenario is selected, also the initial and final position.

        :param scenario: selected scenario
        :param start: initial (x, y) coordinates
        :param goal: final (x, y) coordinates
        :param num_max_obstacles: maximum number of obstacles to generate
        :param delta: minimum distance between each obstacle
        :param seed: The seed to control random generations.

        # === for auto range generation ===
        :param min_distance: minimum distance from objectives

        # === for imposed range ===
        :param range_x: custom range of x coordinates
        :param range_y: custom range of y coordinates
        """
        obstacles = None

        if seed is not None:
            ObstaclesUtils.set_random_seed(seed)
            set_seed(seed)

        if scenario == Scenario.CROWDED:
            dist_factor = min_distance

            min_x = min(start[0] + dist_factor, goal[0] - dist_factor)
            max_x = max(start[0] + dist_factor, goal[0] - dist_factor)
            min_y = min(start[1] + dist_factor, goal[1] - dist_factor)
            max_y = max(start[1] + dist_factor, goal[1] - dist_factor)

            obstacles = generate_obstacles(
                start=start,
                goal=goal,
                num_obstacles=num_max_obstacles,
                x_range=(min_x, max_x) if range_x is None else range_x,
                y_range=(min_y, max_y) if range_y is None else range_y,
                delta=delta
            )
        elif scenario == Scenario.CROWDED_START:
            dist_factor = min_distance

            min_x = min(start[0] - dist_factor, start[0] + dist_factor)
            max_x = max(start[0] - dist_factor, start[0] + dist_factor)
            min_y = min(start[1] + dist_factor, start[1] - dist_factor)
            max_y = max(start[1] + dist_factor, start[1] - dist_factor)

            obstacles = generate_obstacles(
                start=start,
                goal=goal,
                num_obstacles=num_max_obstacles,
                x_range=(min_x, max_x) if range_x is None else range_x,
                y_range=(min_y, max_y) if range_y is None else range_y,
                delta=delta
            )
        elif scenario == Scenario.CROWDED_END:
            dist_factor = min_distance

            min_x = min(goal[0] - dist_factor, goal[0] + dist_factor)
            max_x = max(goal[0] - dist_factor, goal[0] + dist_factor)
            min_y = min(goal[1] + dist_factor, goal[1] - dist_factor)
            max_y = max(goal[1] + dist_factor, goal[1] - dist_factor)

            obstacles = generate_obstacles(
                start=start,
                goal=goal,
                num_obstacles=num_max_obstacles,
                x_range=(min_x, max_x) if range_x is None else range_x,
                y_range=(min_y, max_y) if range_y is None else range_y,
                delta=delta
            )
        elif scenario == Scenario.START_CLOSE_TO_OBSTACLE:
            obstacles = [
                ConvexHull(np.array([
                    [start[0] + 0.1, -3],
                    [start[0] + 0.1, 3],
                    [start[0] + 0.3, 3],
                    [start[0] + 0.3, -3]
                ]))
            ]
        elif scenario == Scenario.END_CLOSE_TO_OBSTACLE:
            obstacles = [
                ConvexHull(np.array([
                    [goal[0] + 0.1, -3],
                    [goal[0] + 0.1, 3],
                    [goal[0] + 0.3, 3],
                    [goal[0] + 0.3, -3]
                ]))
            ]
        elif scenario == Scenario.HORIZONTAL_WALL:
            obstacles = [
                ConvexHull(np.array([[1, -10], [1, 10], [3, 10], [3, -10]]))
            ]
        elif scenario == Scenario.VERTICAL_SLALOM:
            obstacles = [
                ConvexHull(np.array([[1, -1], [1, 10], [2, 10], [2, -1]])),
                ConvexHull(np.array([[3, 1], [3, -10], [4, -10], [4, 1]]))
            ]
        elif scenario == Scenario.MAZE_1:
            start = (0.5, 0.5) if start is None else start
            goal = (7.5, 7.5) if goal is None else goal
            obstacles = [
                ConvexHull(np.array([[-1, -0.5], [3.5, -0.5],
                                     [-1, -1], [3.5, -1]])),  # low wall

                ConvexHull(np.array([[-0.5, -0.5], [-0.5, 6],
                                     [-1, -0.5], [-1, 6]])),  # left wall

                ConvexHull(np.array([[8.5, 2.5], [9, 2.5],
                                     [8.5, 8.5], [9, 8.5]])),  # right wall
                                    
                ConvexHull(np.array([[3.5, 8.5], [9, 8.5],
                                     [3.5, 9], [9, 9]])),  # high wall
                                     
                ConvexHull(np.array([[1, 1.5], [2.5, 2.5], 
                                     [3.5, 3.5], [3, 5],
                                     [1, 4], [7, 4], [7, 4.5]])),  # mid_left_block

                ConvexHull(np.array([[5, 6.5], [8.5, 6.5],
                                     [5, 6], [8.5, 6]])),  # upper_right_block
                
                ConvexHull(np.array([[-1, 6], [3.5, 6],
                                     [-1, 9], [3.5, 9]])),  # upper_left_block
                                    
                ConvexHull(np.array([[3.5, -1], [3.5, 0],
                                     [9, -1],  
                                     [7, 2.5], [9, 2.5]])),  # lower_right_block

            ]
        elif scenario == Scenario.MAZE_2:
            start = (0.5, 0.5) if start is None else start
            goal = (0.5, 7.5) if goal is None else goal
            obstacles = [
                ConvexHull(np.array([[-1, -0.5], [3.5, -0.5],
                                     [-1, -1], [3.5, -1]])),  # low wall

                ConvexHull(np.array([[-0.5, -0.5], [-0.5, 8.5],
                                     [-1, -0.5], [-1, 8.5]])),  # left wall

                ConvexHull(np.array([[8.5, 2.5], [9, 2.5],
                                     [8.5, 6], [9, 6]])),  # right wall
                                    
                ConvexHull(np.array([[-1, 8.5], [5, 8.5],
                                     [-1, 9], [5, 9]])),  # high wall

                ConvexHull(np.array([[-0.5, 2.5], [1, 2.5], 
                                     [-0.5, 4.5], [1, 4.5]])),  # mid_left_block_1
                                     
                ConvexHull(np.array([[1, 2.5], [3.5, 3.5], 
                                     [3, 5], [1, 4],
                                     [7, 4], [7, 4.5]])),  # mid_left_block_2

                ConvexHull(np.array([[-0.5, 6.5], [3.5, 6.5],
                                     [-0.5, 5.5], [3.5, 6]])),  # upper_left_block
                
                ConvexHull(np.array([[5, 6], [9, 6],
                                     [5, 9], [9, 9]])),  # upper_right_block
                                    
                ConvexHull(np.array([[3.5, -1], [3.5, 0],
                                     [9, -1],  
                                     [7, 2.5], [9, 2.5]])),  # lower_right_block

            ]
        elif scenario == Scenario.FEW_OBSTACLES:
            obstacles = [
                ConvexHull(np.array([[3, 2], [5, 4], [2, 2], [2, 4]])),
                ConvexHull(np.array([[4, 1], [5, 0.5], [7, 3], [6, 2.5]]))
            ]
        elif scenario == Scenario.EMPTY:
            obstacles = []

        elif scenario == Scenario.CIRCLE_OBSTACLES:
            obstacles = [
                ObstaclesUtils.generate_circle_like_polygon(10, 0.5, (5, -1)),
                ObstaclesUtils.generate_circle_like_polygon(20, 1, (4, 2)),
                ObstaclesUtils.generate_circle_like_polygon(25, 1.2, (1.5, -1)),
            ]
        elif scenario == Scenario.BASE:
            obstacles = generate_obstacles(
                start=start,
                goal=goal,
                num_obstacles=5,
                x_range=(0, 5),
                y_range=(0, 5),
                delta=delta
            )
        elif scenario == Scenario.MAIN_PAPER:
            start = (0, 0)
            goal = (10, 10)

            obstacles = [
                ConvexHull(np.array([[2.0,7.5], [1.5,7.0], [1.8,6.5]])),
                ConvexHull(np.array([[4.0,6.5], [4.3,6.8], [4.7,6.5], [4.5,6.2], [4.1,6.2]])),
                ConvexHull(np.array([[7.0,7.0], [7.5,7.5], [8.0,7.0], [7.5,6.5]])),
                ConvexHull(np.array([[6.0,2.5], [6.5,2.0], [7.0,2.5]])),
                ConvexHull(np.array([[1.5,3.0], [1.8,3.3], [2.2,3.0], [2.0,2.6], [1.6,2.6]])),
                ConvexHull(np.array([[2.5,3.5], [2.8,3.8], [3.2,3.5], [3.0,3.1], [2.6,3.1]]))
            ]

        return start, goal, obstacles


if __name__ == "__main__":
    scenario = Scenario.load_scenario(Scenario.MAZE_2, start=(0.5,0.5), goal=(0.5,7.5))
    fig, ax = plt.subplots()

    start = scenario[0]
    goal = scenario[1]
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
