import os

import numpy as np

from HumanoidNavigation.MPC.HumanoidMPCVariants.HumanoidMPCCustomLCBF import HumanoidMPCCustomLCBF
from HumanoidNavigation.MPC.HumanoidMPCVariants.HumanoidMPCUnknownEnvironment import HumanoidMPCUnknownEnvironment
from HumanoidNavigation.MPC.HumanoidMpc import conf, HumanoidMPC
from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils
from HumanoidNavigation.Utils.obstacles import set_seed
from HumanoidNavigation.report_simulations.Scenario import Scenario

PLOTS_PATH_BASE = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/Simulation1"
PLOTS_PATH_CIRCLES = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/Simulation1Circles"
PLOTS_PATH_CIRCLES_DELTA = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/Simulation1CirclesDelta"
PLOTS_PATH_UNK_ENV = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/Simulation1UnkEnv"
PLOTS_PATH_UNK_ENV2 = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/Simulation2UnkEnv"
PLOTS_PATH_UNK_ENV3 = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/Simulation3UnkEnv"
PLOTS_PATH_UNK_ENV4 = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/Simulation4UnkEnv"


def run_simulation_1():
    """
    It runs the simulation described in paragraph 1 of the Simulation chapter, and generates all the associated plots.
    """
    initial_state = (0, 0, 0, 0, 0)
    goal_pos = (5, 5)

    sampling_time = conf['DELTA_T']  # 1e-1
    num_steps_per_second = 1 / sampling_time

    _, _, obstacles = Scenario.load_scenario(Scenario.BASE, start=(initial_state[0], initial_state[2]),
                                             goal=goal_pos, seed=7)

    mpc = HumanoidMPC(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=sampling_time,
        goal=goal_pos,
        init_state=initial_state,
        obstacles=obstacles,
        verbosity=0
    )

    X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH_BASE}/animation.gif',
                                                        make_fast_plot=True,
                                                        plot_animation=False, fill_animator=True)

    signals = [
        (X_pred_glob[[0, 2], :] - np.array([[goal_pos[0]], [goal_pos[1]]]), "Position error", ['X error', 'Y error']),
        (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$"),
        (np.concatenate([np.array(X_pred_glob[[0, 2], 10:20]), np.array(U_pred_glob[[0, 1], 9:19])]),
         "CoM and ZMP (foot stance)", ['CoM X', 'CoM Y', 'ZMP X', 'ZMP Y']),
        # (X_pred_glob[[0, 2], :], "CoM position", ['X', 'Y']),
    ]
    PlotUtils.plot_signals(signals, path_to_pdf=f"{PLOTS_PATH_BASE}/evolutions", samples_per_second=num_steps_per_second)

    # CoM and ZMP coordinates
    com_x = np.array(X_pred_glob[[0], 10:20]).squeeze()
    com_y = np.array(X_pred_glob[[2], 10:20]).squeeze()
    zmp_x = np.array(U_pred_glob[[0], 9:19]).squeeze()
    zmp_y = np.array(U_pred_glob[[1], 9:19]).squeeze()

    PlotUtils.plot_com_and_zmp(f"{PLOTS_PATH_BASE}/evolutions", len(signals), com_x, com_y, zmp_x, zmp_y)

    # anim.plot_animation(path_to_gif=f'{PLOTS_PATH}/animation.gif')
    anim.plot_animation(path_to_gif=f'{PLOTS_PATH_BASE}/animation.gif',
                        path_to_frames_folder=f'{PLOTS_PATH_BASE}/grid_frames')


def run_simulation_circles():
    """
    It runs the circles simulation described in paragraph 1 of the Simulation chapter, and generates all the
     associated plots.
    """
    initial_state = (0, 0, 3, 0, 0)
    goal_pos = (6, -3)

    sampling_time = conf['DELTA_T']  # 1e-1
    num_steps_per_second = 1 / sampling_time

    _, _, obstacles = Scenario.load_scenario(Scenario.CIRCLE_OBSTACLES, start=(initial_state[0], initial_state[2]),
                                             goal=goal_pos)

    mpc = HumanoidMPC(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=sampling_time,
        goal=goal_pos,
        init_state=initial_state,
        obstacles=obstacles,
        verbosity=0
    )

    X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH_CIRCLES}/animation.gif',
                                                        make_fast_plot=True,
                                                        plot_animation=False, fill_animator=True)

    # Compute the longitudinal and lateral velocities
    local_vel = PlotUtils.compute_local_velocities(X_pred_glob[4, :], X_pred_glob[[1, 3], :])

    signals = [
        (X_pred_glob[[0, 2], :] - np.array([[goal_pos[0]], [goal_pos[1]]]), "Position error", ['X error', 'Y error']),
        # (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (local_vel, "Translational velocity", ['Longitudinal velocity', 'Lateral velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$"),
        (np.concatenate((X_pred_glob[[0], :-1], U_pred_glob[[0]]), axis=0),
         "CoM and ZMP (foot stance)", ['CoM X', 'ZMP X', ], (2.5, 9), (0.5, 3.5)),
    ]
    PlotUtils.plot_signals(signals, path_to_pdf=f"{PLOTS_PATH_CIRCLES}/evolutions", samples_per_second=num_steps_per_second)

    # CoM and ZMP coordinates
    com_x = np.array(X_pred_glob[[0], 18:31]).squeeze()
    com_y = np.array(X_pred_glob[[2], 18:31]).squeeze()
    zmp_x = np.array(U_pred_glob[[0], 17:30]).squeeze()
    zmp_y = np.array(U_pred_glob[[1], 17:30]).squeeze()

    PlotUtils.plot_com_and_zmp(f"{PLOTS_PATH_CIRCLES}/evolutions", len(signals), com_x, com_y, zmp_x, zmp_y)

    anim.plot_animation(path_to_gif=f'{PLOTS_PATH_CIRCLES}/animation.gif',
                        path_to_frames_folder=f'{PLOTS_PATH_CIRCLES}/grid_frames',
                        min_max_coords=((-0.5, 6.25), (-3.25, 3.25)))


def run_simulation_circles_custom_ldcbf():
    """
    It runs the circles simulation with custom LDCBF described in paragraph 1 of the Simulation chapter, and generates
    all the associated plots.
    """
    initial_state = (0, 0, 3, 0, 0)
    goal_pos = (6, -3)

    sampling_time = conf['DELTA_T']  # 1e-1
    num_steps_per_second = 1 / sampling_time

    delta = 0.3

    _, _, obstacles = Scenario.load_scenario(Scenario.CIRCLE_OBSTACLES, start=(initial_state[0], initial_state[2]),
                                             goal=goal_pos)

    mpc = HumanoidMPCCustomLCBF(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=sampling_time,
        goal=goal_pos,
        init_state=initial_state,
        obstacles=obstacles,
        verbosity=0,
        distance_from_obstacles=delta,
    )

    X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH_CIRCLES_DELTA}/animation.gif',
                                                        make_fast_plot=True,
                                                        plot_animation=False, fill_animator=True)

    # Compute the longitudinal and lateral velocities
    local_vel = PlotUtils.compute_local_velocities(X_pred_glob[4, :], X_pred_glob[[1, 3], :])

    signals = [
        (X_pred_glob[[0, 2], :] - np.array([[goal_pos[0]], [goal_pos[1]]]), "Position error", ['X error', 'Y error']),
        # (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (local_vel, "Translational velocity", ['Longitudinal velocity', 'Lateral velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$"),
        (np.concatenate((X_pred_glob[[0], :-1], U_pred_glob[[0]]), axis=0),
         "CoM and ZMP (foot stance)", ['CoM X', 'ZMP X', ], (2.5, 9), (0.5, 3.5)),
    ]
    PlotUtils.plot_signals(signals, path_to_pdf=f"{PLOTS_PATH_CIRCLES_DELTA}/evolutions", samples_per_second=num_steps_per_second)

    # CoM and ZMP coordinates
    com_x = np.array(X_pred_glob[[0], 10:21]).squeeze()
    com_y = np.array(X_pred_glob[[2], 10:21]).squeeze()
    zmp_x = np.array(U_pred_glob[[0], 9:20]).squeeze()
    zmp_y = np.array(U_pred_glob[[1], 9:20]).squeeze()

    PlotUtils.plot_com_and_zmp(f"{PLOTS_PATH_CIRCLES_DELTA}/evolutions", len(signals), com_x, com_y, zmp_x, zmp_y)

    anim.delta = delta
    # anim.plot_animation(path_to_gif=f'{PLOTS_PATH_CIRCLES_DELTA}/animation.gif')
    anim.plot_animation(path_to_gif=f'{PLOTS_PATH_CIRCLES_DELTA}/animation.gif',
                        path_to_frames_folder=f'{PLOTS_PATH_CIRCLES_DELTA}/grid_frames',
                        min_max_coords=((-0.5, 6.25), (-3.25, 3.25)))


def run_simulation_unk_env():
    """
    It runs the circles simulation described in paragraph 1 of the Simulation chapter, and generates all the
     associated plots.
    """

    seed = 10
    # seed = 12
    ObstaclesUtils.set_random_seed(seed)
    set_seed(seed)

    start, goal_pos = (0, 0), (4, 3.5)

    sampling_time = conf['DELTA_T']  # 1e-1
    num_steps_per_second = 1 / sampling_time

    start, goal, obstacles = Scenario.load_scenario(
        Scenario.CROWDED,
        start,
        goal_pos,
        20,
        range_x=(-1, 6),
        range_y=(-1, 6)
    )

    initial_state = (start[0], 0, start[1], 0, np.pi/2)

    mpc = HumanoidMPCUnknownEnvironment(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=conf["DELTA_T"],
        goal=goal,
        init_state=initial_state,
        obstacles=obstacles,
        verbosity=0,
        lidar_range=1.5
    )

    X_pred_glob, U_pred_glob, anim = mpc.run_simulation(
        path_to_gif=f'{PLOTS_PATH_UNK_ENV4}/animation.gif',
        make_fast_plot=True,
        plot_animation=False,
        fill_animator=True)

    min_range = 10
    max_range = 20

    # Compute the longitudinal and lateral velocities
    local_vel = PlotUtils.compute_local_velocities(X_pred_glob[4, :], X_pred_glob[[1, 3], :])

    signals = [
        (X_pred_glob[[0, 2], :] - np.array([[goal_pos[0]], [goal_pos[1]]]), "Position error", ['X error', 'Y error']),
        (local_vel, "Translational velocity", ['Longitudinal velocity', 'Lateral velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$"),
        # (np.concatenate([np.array(X_pred_glob[[0, 2], min_range:max_range]),
        #                  np.array(U_pred_glob[[0, 1], min_range - 1:max_range - 1])]),
        #  "CoM and ZMP (foot stance)", ['CoM X', 'CoM Y', 'ZMP X', 'ZMP Y']),
        (np.concatenate((X_pred_glob[[0], :-1], U_pred_glob[[0]]), axis=0),
         "CoM and ZMP (foot stance)", ['CoM X', 'ZMP X'], (2.5, 9), (0.5, 3.5)),
    ]
    PlotUtils.plot_signals(signals, path_to_pdf=f"{PLOTS_PATH_UNK_ENV4}/evolutions", samples_per_second=num_steps_per_second)

    # CoM and ZMP coordinate
    com_x = np.array(X_pred_glob[[0], min_range:max_range+1]).squeeze()
    com_y = np.array(X_pred_glob[[2], min_range:max_range+1]).squeeze()
    zmp_x = np.array(U_pred_glob[[0], min_range - 1:max_range]).squeeze()
    zmp_y = np.array(U_pred_glob[[1], min_range - 1:max_range]).squeeze()

    PlotUtils.plot_com_and_zmp(f"{PLOTS_PATH_UNK_ENV4}/evolutions", len(signals), com_x, com_y, zmp_x, zmp_y)

    anim.plot_animation(path_to_gif=f'{PLOTS_PATH_UNK_ENV4}/animation.gif',
                        path_to_frames_folder=f'{PLOTS_PATH_UNK_ENV4}/grid_frames',
                        min_max_coords=((-0.25, 5.5), (-0.25, 5.5)))


if __name__ == "__main__":
    # run_simulation_1()
    # run_simulation_circles()
    run_simulation_unk_env()
    # run_simulation_circles_custom_ldcbf()
