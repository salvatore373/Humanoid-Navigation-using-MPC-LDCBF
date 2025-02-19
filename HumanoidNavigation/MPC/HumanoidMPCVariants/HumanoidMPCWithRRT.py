import math
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from rrtplanner import plot_rrt_lines, plot_path, plot_og, plot_start_goal
from rrtplanner import r2norm, RRTStarInformed
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Delaunay

from HumanoidNavigation.MPC.HumanoidMPCVariants.HumanoidMPCCustomLCBF import HumanoidMPCCustomLCBF
from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC
from HumanoidNavigation.Utils.HumanoidAnimationUtils import HumanoidAnimationUtils


class HumanoidMPCWithRRT(HumanoidMPC):
    """
    A subclass of HumanoidMPC, where the user-given goal is reached by navigating though the sub-goals provided by the
     RRT algorithm.
    """

    def _build_occupancy_grid(self, width_grid_size: int) -> \
            tuple[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[
                [np.ndarray, np.ndarray], np.ndarray]]:
        """
        It returns the environment as an occupancy grid, i.e. a matrix where each cell is 1 if it contains an obstacle
        and 0 otherwise. Furthermore, it returns a function to convert from world's coordinates to occupancy grid
        coordinates, and another function to compute the inverse transformation.

        :param width_grid_size: The number of cells that the grid should have as width. The height will be chosen
         appropriately to preserve the original ratio.
        """
        # Find the minimum and maximum X and Y coordinates of the obstacles
        min_obs_x, min_obs_y = float('inf'), float('inf')
        max_obs_x, max_obs_y = float('-inf'), float('-inf')
        for o in self.obstacles:
            obs_vertices = o.points[o.vertices]
            obs_vertices_x, obs_vertices_y = obs_vertices[:, 0], obs_vertices[:, 1]
            min_obs_x = min(obs_vertices_x.min(), min_obs_x)
            max_obs_x = max(obs_vertices_x.max(), max_obs_x)
            min_obs_y = min(obs_vertices_y.min(), min_obs_y)
            max_obs_y = max(obs_vertices_y.max(), max_obs_y)
        # Find the minimum and maximum coordinates to include in the occupancy grid, considering the initial position,
        # the goal position and the obstacles. (initial position is always (0,0), regardless of
        # the provided initial state).
        min_x = min(0, self.goal[0], min_obs_x)
        min_y = min(0, self.goal[1], min_obs_y)
        max_x = max(0, self.goal[0], max_obs_x)
        max_y = max(0, self.goal[1], max_obs_y)

        # Initialize the occupancy grid
        height_grid_size = math.ceil(width_grid_size * ((max_y - min_y) / (max_x - min_x)))
        # Create the grid filled with zeros
        occupancy_grid = np.zeros((width_grid_size + 1, height_grid_size + 1))

        # Define the function to perform the global to occupancy grid coordinates
        transformation_fun = lambda x_glob, y_glob: np.array([
            np.round(((x_glob - min_x) / (max_x - min_x)) * width_grid_size),
            np.round(((y_glob - min_y) / (max_y - min_y)) * height_grid_size),
        ]).astype(int)
        # Define the function to perform the occupancy grid to global coordinates
        inverse_transformation_fun = lambda x_og, y_og: np.array([
            min_x + ((x_og * (max_x - min_x)) / width_grid_size),
            min_y + ((y_og * (max_y - min_y)) / height_grid_size),
        ])

        # For each obstacle, fill with 1s the area it occupies in the grid
        for o in self.obstacles:
            obs_vertices = o.points[o.vertices]
            # Convert the coordinates of the obstacle to the grid space
            obs_vertices = transformation_fun(obs_vertices[:, 0], obs_vertices[:, 1]).T
            # obs_vertices[:, 0] = np.round(((obs_vertices[:, 0] - min_x) / max_x) * (width_grid_size-1))
            # obs_vertices[:, 1] = np.round(((obs_vertices[:, 1] - min_y) / max_y) * (height_grid_size-1))
            # obs_vertices = obs_vertices.astype(int)
            # Initialize a Delaunay tessellation of the discrete obstacle
            hull_delaunay_discrete = Delaunay(obs_vertices)

            # Find the coordinates of the points in the convex hull in the grid
            obs_inner_coords = []
            for x_coord in range(obs_vertices[:, 0].min(), obs_vertices[:, 0].max()):
                for y_coord in range(obs_vertices[:, 1].min(), obs_vertices[:, 1].max()):
                    discr_coord = [x_coord, y_coord]
                    if hull_delaunay_discrete.find_simplex(discr_coord) >= 0:
                        obs_inner_coords.append(discr_coord)

            # Set to 1 all the grid cells containing the obstacle
            obs_inner_coords = np.array(obs_inner_coords)
            occupancy_grid[obs_inner_coords[:, 0], obs_inner_coords[:, 1]] = 1

        return occupancy_grid, transformation_fun, inverse_transformation_fun

    def run_simulation(self, path_to_gif: str, make_fast_plot: bool = True, plot_animation: bool = False,
                       fill_animator: bool = True, initial_animator: HumanoidAnimationUtils = None,
                       visualize_rrt_path: bool = False) -> \
            tuple[np.ndarray, np.ndarray, Union[None, HumanoidAnimationUtils]]:
        """
        :param visualize_rrt_path: Whether to visualize or not the plot of the RRT computation and path
        """
        # Convert the environment to an occupancy grid
        occupancy_grid, transformation_fun, inverse_transformation_fun = \
            self._build_occupancy_grid(width_grid_size=250)
        # Convert the start and goal positions to the occupancy grid coordinates
        goal_og_coords = transformation_fun(self.goal[0], self.goal[1])
        start_og_coords = transformation_fun(0, 0)

        # For each cell of the occupancy grid, compute the distance from the closest obstacle
        dst_from_obs = distance_transform_edt(1 - occupancy_grid)
        # Compute the matrix of the cells costs by assigning the minimum cost to the cell with maximum distance
        # from the obstacles. The exponential is to guarantee asymptotic optimality (ref:
        # https://robotics.stackexchange.com/questions/649/does-rrt-guarantee-asymptotic-optimality-for-a-minimum-clearance-cost-metric).
        costs_matrix = np.exp(-dst_from_obs)

        # Define the cost function that the RRT* algorithm should minimize, including the travelled distance
        # and the clearance.
        def cost_fn(vcosts: np.ndarray,
                    points: np.ndarray,
                    v: int,
                    x: np.ndarray, ):
            return vcosts[v] + costs_matrix[x[0], x[1]] * r2norm(points[v] - x)

        # Find a path from the initial to the goal position using RRT
        # og: np matrix with 1 if the cell contains an obstacle, 0 otherwise
        # n: the maximum number of points that can be sampled in plan()
        # r_rewire: value of delta in RRT algorithm
        # pbar: whthere to display a progress bar
        rrts = RRTStarInformed(og=occupancy_grid, n=1500, r_rewire=10, pbar=False, costfn=cost_fn, r_goal=10)
        T, gv = rrts.plan(start_og_coords, goal_og_coords)
        # From the RRT result, get the sequence of occupancy grid cells that must be reached in order to reach the goal
        tree_path_start2goal = rrts.route2gv(T, gv)
        sub_goals_og_seq = rrts.vertices_as_ndarray(T, tree_path_start2goal)
        # Convert the sub goals from occupancy grid to global coordinates
        sub_goals = np.zeros((len(sub_goals_og_seq), 2))
        for sub_goal_ind, start2goal_mat in enumerate(sub_goals_og_seq):
            sub_goals[sub_goal_ind, :] = inverse_transformation_fun(start2goal_mat[1, 0], start2goal_mat[1, 1])

        # Visualize the results
        if visualize_rrt_path:
            # create figure and ax.
            fig = plt.figure()
            ax = fig.add_subplot()
            # these functions alter ax in-place.
            plot_og(ax, occupancy_grid)
            plot_start_goal(ax, start_og_coords, goal_og_coords)
            plot_rrt_lines(ax, T)
            plot_path(ax, sub_goals_og_seq)
            ax.set_aspect('equal')  # Set equal aspect ratio for accurate proportions
            plt.show()

        # Reach all the sub-goals sequentially
        X_pred_glob, U_pred_glob = None, None
        start_state = (0, 0, 0, 0, 0)
        animator = initial_animator
        for i, sub_goal in enumerate(sub_goals):
            # Add the current sub goal to the animation data
            if animator is not None:
                animator.add_goal(sub_goal)
            # Starting from the current position, reach the next sub-goal in the path
            curr_mpc = HumanoidMPCCustomLCBF(
                goal=sub_goal,
                init_state=start_state,
                obstacles=self.obstacles,
                N_horizon=self.N_horizon,
                N_mpc_timesteps=self.N_simul,
                sampling_time=self.sampling_time,
                start_with_right_foot=self.start_with_right_foot,
                verbosity=self.verbosity,
                distance_from_obstacles=0.3,
            )
            curr_X_pred, curr_U_pred, animator = (
                curr_mpc.run_simulation(path_to_gif=path_to_gif,
                                        make_fast_plot=i == len(sub_goals) - 1 and make_fast_plot,
                                        plot_animation=i == len(sub_goals) - 1 and plot_animation,
                                        fill_animator=fill_animator, initial_animator=animator))
            # Set the last state of this run as initial state of the next one
            start_state = tuple(curr_X_pred[:, -1])
            # Update the evolution of the state and input from the provided initial state to the end
            X_pred_glob = curr_X_pred if X_pred_glob is None else np.concatenate((X_pred_glob, curr_X_pred), axis=1)
            U_pred_glob = curr_U_pred if U_pred_glob is None else np.concatenate((U_pred_glob, curr_U_pred), axis=1)

        return X_pred_glob, U_pred_glob, animator
