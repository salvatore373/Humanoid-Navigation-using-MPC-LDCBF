import itertools
import os

import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.MPC import HumanoidMpc
from HumanoidNavigation.MPC.HumanoidMpc import conf, HumanoidMPC
from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils
from HumanoidNavigation.Utils.obstacles import generate_obstacles, set_seed

PLOTS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/Simulation1/"


def run_simulation_1(start, goal, include_obstacles=False):
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

    mpc = HumanoidMPC(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=conf["DELTA_T"],
        # sampling_time=1e-1,
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

    X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif', make_fast_plot=True,
                                                        plot_animation=False, fill_animator=True)

    PlotUtils.plot_signals([
        (X_pred_glob[[0, 2], :] - np.array([[goal[0]], [goal[1]]]), "Position error", ['X error', 'Y error']),
        (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$")
    ], path_to_pdf=f"{PLOTS_PATH}/evolutions.pdf")

    anim.plot_animation(path_to_gif=f'{PLOTS_PATH}/animation.gif')


def bounds_tuning():
    best_combin = None
    best_res = float('inf')

    v_max_x = np.arange(0.2, 1, 0.05)
    v_max_y = np.arange(0.2, 0.4, 0.05)
    alpha = np.arange(0.5, 4, 0.1)
    hyperparams_combinations = itertools.product(v_max_x, v_max_y, alpha)
    for vx_max, vy_max, a in hyperparams_combinations:
        HumanoidMpc.conf['ALPHA'] = a
        HumanoidMpc.conf['V_MAX'] = [vx_max, vy_max]

        mpc = HumanoidMPC(
            N_horizon=3,
            N_mpc_timesteps=300,
            # sampling_time=conf["DELTA_T"],
            sampling_time=1e-1,
            goal=(5, 5),
            init_state=(0, 0, 0, 0, 0),
            obstacles=[],
            verbosity=0
        )
        X_pred_glob, U_pred_glob, _ = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif', make_fast_plot=False,
                                                            plot_animation=False, fill_animator=False)

        if all((X_pred_glob[[0, 2], -1] - [5,5])**2 <= 1):
            val = max(X_pred_glob[3, :].max(), -X_pred_glob[3, :].min())
            if val < best_res:
                best_res = val
                best_combin = (vx_max, vy_max, a)

    print(f'best_combination: {best_combin}')
    print(f'best_res: {best_res}')
    HumanoidMpc.conf['ALPHA'] = best_combin[2]
    HumanoidMpc.conf['V_MAX'] = [best_combin[0], best_combin[1]]

    mpc = HumanoidMPC(
        N_horizon=3,
        N_mpc_timesteps=300,
        # sampling_time=conf["DELTA_T"],
        sampling_time=1e-1,
        goal=(5, 5),
        init_state=(0, 0, 0, 0, 0),
        obstacles=[],
        verbosity=0
    )
    X_pred_glob, U_pred_glob, _ = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif', make_fast_plot=True,
                                                     plot_animation=True, fill_animator=True)
    PlotUtils.plot_signals([
        (X_pred_glob[[0, 2], :] - np.array([[5], [5]]), "Position error", ['X error', 'Y error']),
        (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$")
    ], path_to_pdf=f"{PLOTS_PATH}/evolutions.pdf")



if __name__ == "__main__":
    bounds_tuning()
    exit(0)

    print("******* run_simulation_1(start=(0, 0, 0, 0, 0), goal=(5, 5), include_obstacles=Fasle) ****")
    run_simulation_1(start=(0, 0, 0, 0, 0), goal=(5, 5), include_obstacles=False)
    print("******* run_simulation_1(start=(0, 0, 0, 0, 0), goal=(5, 5), include_obstacles=True) ****")
    run_simulation_1(start=(0, 0, 0, 0, 0), goal=(5, 5), include_obstacles=True)
    print("******* run_simulation_1(start=(0, 0, 0, 0, 0), goal=(0, 3), include_obstacles=False) ****")
    run_simulation_1(start=(0, 0, 0, 0, 0), goal=(0, 3), include_obstacles=False)
    print("******* run_simulation_1(start=(0, 0, 0, 0, 0), goal=(0, -3), include_obstacles=False) ****")
    run_simulation_1(start=(0, 0, 0, 0, 0), goal=(0, -3), include_obstacles=False)
    print("******* run_simulation_1(start=(0, 0, 0, 0, 0), goal=(3, -3), include_obstacles=False) ****")
    run_simulation_1(start=(0, 0, 0, 0, 0), goal=(3, -3), include_obstacles=False)
    print("******* run_simulation_1(start=(0, 0, 0, 0, 0), goal=(3, 3), include_obstacles=False) ****")
    run_simulation_1(start=(0, 0, 0, 0, 0), goal=(3, 3), include_obstacles=False)
