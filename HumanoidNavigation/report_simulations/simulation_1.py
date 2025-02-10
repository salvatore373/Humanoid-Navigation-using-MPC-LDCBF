import os

import numpy as np

from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC, conf
from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils
from HumanoidNavigation.Utils.obstacles import generate_obstacles, set_seed

PLOTS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/Simulation1/"


def run_simulation_1():
    """
    It runs the simulation described in paragraph 1 of the Simulation chapter, and generates all the associated plots.
    """
    ObstaclesUtils.set_random_seed(1)
    set_seed(1)

    start = (0, 0, 0, 0, np.pi * 3 / 2)
    goal = (5, 5)

    obstacles = generate_obstacles(
        start=(start[0], start[2]),
        goal=goal,
        num_obstacles=5,
        x_range=(0, 5),
        y_range=(0, 5)
    )

    obstacles = obstacles[1:2]

    mpc = HumanoidMPC(
        N_horizon=3,
        N_simul=300,
        sampling_time=conf["DELTA_T"],
        # sampling_time=1e-2,
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

    X_pred_glob, U_pred_glob = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif', make_fast_plot=True)

    PlotUtils.plot_signals([
        (X_pred_glob[[0, 2], :] - np.array([[goal[0]], [goal[1]]]), "Position error$", ['X error', 'Y error']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$")
    ], path_to_pdf=f"{PLOTS_PATH}/evolutions.pdf")


if __name__ == "__main__":
    run_simulation_1()
