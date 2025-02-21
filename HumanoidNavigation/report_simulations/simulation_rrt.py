import os

import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.MPC.HumanoidMPCVariants.HumanoidMPCWithRRT import HumanoidMPCWithRRT
from HumanoidNavigation.MPC.HumanoidMpc import conf
from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils
from HumanoidNavigation.Utils.obstacles import generate_obstacles, set_seed

PLOTS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/SimulationRRT/"


def run_simulation_rrt(start, goal, include_obstacles=False):
    """
    It runs the simulation described in paragraph 1 of the Simulation chapter, and generates all the associated plots.
    """
    ObstaclesUtils.set_random_seed(1)
    set_seed(1)

    if start is None:
        start = (0, 0, 0, 0, 0)
    if goal is None:
        goal = (5, 5)

    obstacles = generate_obstacles(
        start=(start[0], start[2]),
        goal=goal,
        num_obstacles=5,
        x_range=(0, 5),
        y_range=(0, 5)
    )

    # obstacles = [ConvexHull(o) for o in obstacles[1:2]]
    if include_obstacles:
        obstacles = [ConvexHull(o) for o in obstacles]
        # obstacles = [ConvexHull(np.array([[0,4], [0, 2], [2, 2], [2, 4]]))]
    else:
        obstacles = []

    # mpc = HumanoidMPC(
    mpc = HumanoidMPCWithRRT(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=conf["DELTA_T"],
        # sampling_time=4e-1,
        goal=goal,
        init_state=start,
        obstacles=obstacles,
        # obstacles=[
        #     obstacle1,
        #     # obstacle2,
        #     # obstacle3,
        # ],
        verbosity=0
    )

    X_pred_glob, U_pred_glob, _ = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif', make_fast_plot=False,
                                                     plot_animation=True, visualize_rrt_path=True)

    PlotUtils.plot_signals([
        (X_pred_glob[[0, 2], :] - np.array([[goal[0]], [goal[1]]]), "Position error", ['X error', 'Y error']),
        (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$")
    ], path_to_pdf=f"{PLOTS_PATH}/evolutions.pdf")


if __name__ == "__main__":
    run_simulation_rrt(start=(0, 0, 0, 0, 0), goal=(5, 0), include_obstacles=True)
