import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC, conf, ASSETS_PATH
from HumanoidNavigation.RangeFinder.range_finder_wth_polygons_dbscan import range_finder
from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.obstacles import set_seed
from HumanoidNavigation.report_simulations.Scenario import load_scenario, Scenario


class HumanoidMPCUnknownEnvironment(HumanoidMPC):
    """
    A subclass of HumanoidMPC, where the robot is not aware of the full map, but it can only perceive the environment
     through a LiDAR system.
    """

    def _get_list_c_and_eta(self, loc_x_k: float, loc_y_k: float, glob_theta_k: float, glob_x_km1: float,
                            glob_y_km1: float):
        """
        It computes the list of the points C and the normal vectors Eta, which are defined for each obstacle as the
        point on the obstacle's edge that is closest to the robot's position, and the vector from the robot's position
        to C.

        :param loc_x_k: The CoM X-coordinate of the humanoid w.r.t the local RF at time step k in the simulation.
        :param loc_y_k: The CoM Y-coordinate of the humanoid w.r.t the local RF at time step k in the simulation.
        :param glob_theta_k: The orientation of the humanoid w.r.t the inertial RF at time step k in the simulation.
        :param glob_x_km1: The CoM X-coordinate of the humanoid w.r.t the inertial RF at time step k-1 in the simulation
        :param glob_y_km1: The CoM Y-coordinate of the humanoid w.r.t the inertial RF at time step k-1 in the simulation
        """
        list_c, list_norm_vec = [], []
        # Get the vector of the CoM position from the current state
        pos_from_state = np.array([loc_x_k, loc_y_k])
        # pos_from_state = [loc_x_k, loc_y_k]

        # Add one constraint for each obstacle in the map
        local_obstacles = []
        for obstacle in self.obstacles:
            #  Convert the obstacle's points in the local RF (i.e. the one of the state)
            local_obstacle = ObstaclesUtils.transform_obstacle_coords(
                obstacle=obstacle, transformation_matrix=self._get_glob_to_loc_rf_trans_mat(
                    glob_theta_k, glob_x_km1, glob_y_km1
                )
            )
            local_obstacles.append(local_obstacle)

        # perform range finder search
        lidar_readings, _, inferred_obstacles = range_finder(
            lidar_position=pos_from_state,
            obstacles=[ch.points for ch in local_obstacles],
            # obstacles = [ch.points for ch in self.obstacles],
            lidar_range=3.0,
            resolution=360
        )

        current_inferred_obstacles = []

        for obstacle in inferred_obstacles:
            local_obstacle = ConvexHull(obstacle)
            current_inferred_obstacles.append(local_obstacle)
            # Find c, i.e. the point on the obstacle's edge closest to (com_x, com_y) and compute
            # the normal vector from (com_x, com_y) to c
            c, normal_vector = ObstaclesUtils.get_closest_point_and_normal_vector_from_obs(
                x=pos_from_state, polygon=local_obstacle, unitary_normal_vector=True,
            )
            list_c.append(c)
            list_norm_vec.append(normal_vector)

        self.list_inferred_obstacles.append(current_inferred_obstacles)
        self.list_lidar_readings.append(lidar_readings)

        return list_c, list_norm_vec


if __name__ == "__main__":
    ObstaclesUtils.set_random_seed(1)
    set_seed(1)

    start, goal = (0, 0), (5, 0)

    start, goal, obstacles = load_scenario(Scenario.CROWDED, start, goal)

    initial_state = (start[0], 0, start[1], 0, np.pi * 3 / 2)

    # mpc = HumanoidMPC(
    mpc = HumanoidMPCUnknownEnvironment(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=conf["DELTA_T"],
        goal=goal,
        # goal=(5, 5),
        init_state=initial_state,
        # init_state=(0, 0, 0, 0, 0),
        obstacles=obstacles,
        # obstacles=[
        #     obstacle1,
        #     # obstacle2,
        #     # obstacle3,
        # ],
        verbosity=0
    )

    mpc.run_simulation(path_to_gif=ASSETS_PATH, make_fast_plot=True, plot_animation=False)
