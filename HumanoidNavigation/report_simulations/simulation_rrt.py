import os

import numpy as np
from scipy.spatial import ConvexHull

from HumanoidNavigation.MPC.HumanoidMPCVariants.HumanoidMPCWithRRT import HumanoidMPCWithRRT
from HumanoidNavigation.MPC.HumanoidMpc import conf, HumanoidMPC
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils

PLOTS_PATH_RRT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/SimulationRRT"

PLOTS_PATH_NO_RRT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/SimulationRRT-NoRRT"


def full_simulation():
    start = (0, 0, 0, 0, 0)
    goal_pos = (5, 0)

    sampling_time = conf['DELTA_T']  # 1e-1
    num_steps_per_second = 1 / sampling_time

    obstacles = [ConvexHull(np.array([[2, -3], [2, 3], [3, -3], [3, 3]]))]

    run_simulation_no_rrt(start, goal_pos, sampling_time, num_steps_per_second, obstacles)
    run_simulation_rrt(start, goal_pos, sampling_time, num_steps_per_second, obstacles)


def run_simulation_no_rrt(start, goal_pos, sampling_time, num_steps_per_second, obstacles):
    """
    It runs the simulation where the humanoid reaches the goal by following a sequence of sub-goals without using RRT,
     and generates all the associated plots.
    """
    mpc = HumanoidMPC(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=sampling_time,
        goal=goal_pos,
        init_state=start,
        obstacles=obstacles,
        verbosity=0
    )

    X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH_NO_RRT}/animation.gif',
                                                        make_fast_plot=True, plot_animation=False,
                                                        fill_animator=True)

    PlotUtils.plot_signals([
        (X_pred_glob[[0, 2], :] - np.array([[goal_pos[0]], [goal_pos[1]]]), "Position error", ['X error', 'Y error']),
        (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$")
    ], path_to_pdf=f"{PLOTS_PATH_NO_RRT}/evolutions", samples_per_second=num_steps_per_second)

    # anim.plot_animation(path_to_gif=f'{PLOTS_PATH_NO_RRT}/animation.gif')
    anim.plot_animation(path_to_gif=f'{PLOTS_PATH_NO_RRT}/animation.gif',
                        path_to_frames_folder=f'{PLOTS_PATH_NO_RRT}/grid_frames')


def run_simulation_rrt(start, goal_pos, sampling_time, num_steps_per_second, obstacles):
    """
    It runs the simulation where the humanoid reaches the goal by following a sequence of sub-goals with RRT,
     and generates all the associated plots.
    """
    mpc = HumanoidMPCWithRRT(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=sampling_time,
        # sampling_time=4e-1,
        goal=goal_pos,
        init_state=start,
        obstacles=obstacles,
        verbosity=0
    )

    X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH_RRT}/animation.gif',
                                                        make_fast_plot=True, visualize_rrt_path=True,
                                                        plot_animation=False, fill_animator=True,
                                                        path_to_rrt_pdf=f'{PLOTS_PATH_RRT}/rrt_res.pdf')

    PlotUtils.plot_signals([
        (X_pred_glob[[0, 2], :] - np.array([[goal_pos[0]], [goal_pos[1]]]), "Position error", ['X error', 'Y error']),
        (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$")
    ], path_to_pdf=f"{PLOTS_PATH_RRT}/evolutions", samples_per_second=num_steps_per_second)

    # anim.plot_animation(path_to_gif=f'{PLOTS_PATH_RRT}/animation.gif')
    anim.plot_animation(path_to_gif=f'{PLOTS_PATH_RRT}/animation.gif',
                        path_to_frames_folder=f'{PLOTS_PATH_RRT}/grid_frames')


if __name__ == "__main__":
    full_simulation()
