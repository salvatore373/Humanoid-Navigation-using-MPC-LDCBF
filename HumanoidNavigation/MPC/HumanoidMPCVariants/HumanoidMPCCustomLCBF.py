from typing import Union

import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC


class HumanoidMPCCustomLCBF(HumanoidMPC):
    """
    A subclass of HumanoidMPC, where the paper's LCBF function is replaced with the one defined by us, that allows the
    to keep the robot at a user's defined distance from the obstacles.
    """

    def __init__(self, goal, obstacles, N_horizon=3, N_mpc_timesteps=100, sampling_time=1e-3,
                 init_state: Union[np.ndarray, tuple[float, float, float, float, float]] = None,
                 start_with_right_foot: bool = True, verbosity: int = 1,
                 distance_from_obstacles: float = 0.0):
        """
        :param distance_from_obstacles: The distance that the CoM will keep from all the obstacles, i.e. the value delta
         s.t. the position of the CoM will always be h(x) >= delta, where h(x) is the value of the LCBF computed in the
          current CoM position. It must be non-negative.
        """
        assert distance_from_obstacles >= 0.0, "distance_from_obstacles must be non-negative"

        super().__init__(goal, obstacles, N_horizon, N_mpc_timesteps, sampling_time,
                         init_state, start_with_right_foot, verbosity)
        self.distance_from_obstacles = distance_from_obstacles

    def _compute_single_lcbf(self, x, eta, c):
        return eta.T @ (x - c) - self.distance_from_obstacles


if __name__ == "__main__":
    # only one and very far away
    obstacle1 = ConvexHull(np.array([[0, 2], [0, 4], [2, 2], [2, 4]]))
    # obstacle1 = ConvexHull(np.array([[-0.5, 2], [-0.5, 4], [2, 2], [2, 4]]))
    # obstacle1 = ObstaclesUtils.generate_random_convex_polygon(5, (-0.5, 0.5), (2, 4))
    # obstacle2 = ObstaclesUtils.generate_random_convex_polygon(5, (-1.2, -0.5), (2, 4))
    # obstacle3 = ObstaclesUtils.generate_random_convex_polygon(5, (-0.1, 0.5), (2, 4))

    mpc = HumanoidMPCCustomLCBF(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=HumanoidMPC.DELTA_T,
        goal=(-1, 3),
        obstacles=[
            obstacle1,
            # obstacle2,
            # obstacle3,
        ],
        verbosity=0,
        distance_from_obstacles=.5
    )

    mpc.run_simulation(path_to_gif='/Users/salvatore/Downloads/re2.gif', make_fast_plot=True)
