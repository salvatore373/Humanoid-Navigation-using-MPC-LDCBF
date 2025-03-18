import os
import numpy as np

from HumanoidNavigation.MPC.HumanoidMpc import conf
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils
from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC
from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.report_simulations.Scenario import Scenario
from HumanoidNavigation.MPC.HumanoidMPCVariants.HumanoidMPCWithRRT import HumanoidMPCWithRRT

PLOTS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/ReportResults/SimulationMaze2/"


def run_simulation_maze(start, goal, rrt=False):
    """
    It runs the simulation described in paragraph of the Simulation chapter, and generates all the associated plots.
    """

    start, goal, obstacles = Scenario.load_scenario(
        Scenario.MAZE_2,
        (start[0], start[2]),
        goal,
        20,
        range_x=(-1, 6),
        range_y=(-1, 6)
    )

    initial_state = (start[0], 0, start[1], 0, 0)
    num_steps_per_second = 1 / conf["DELTA_T"]

    if not rrt:
        mpc = HumanoidMPC(
            N_horizon=2,
            N_mpc_timesteps=500,
            sampling_time=conf["DELTA_T"],
            goal=goal,
            init_state=initial_state,
            obstacles=obstacles,
            verbosity=0
        )

        X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif', make_fast_plot=True,
                                                     plot_animation=True, fill_animator=True)
        
    else:
        mpc = HumanoidMPCWithRRT(
            N_horizon=3,
            N_mpc_timesteps=300,
            sampling_time=conf["DELTA_T"],
            goal=goal,
            init_state=initial_state,
            obstacles=obstacles,
            verbosity=0
        )

        X_pred_glob, U_pred_glob, anim = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif', make_fast_plot=True,
                                                     visualize_rrt_path=True, plot_animation=True, fill_animator=True,
                                                     path_to_rrt_pdf=f'{PLOTS_PATH}/rrt_res.pdf')

    PlotUtils.plot_signals([
        (X_pred_glob[[0, 2], :] - np.array([[goal[0]], [goal[1]]]), "Position error", ['X error', 'Y error']),
        (X_pred_glob[[1, 3], :], "Translational velocity", ['X velocity', 'Y velocity']),
        (np.expand_dims(X_pred_glob[4, :], axis=0), "Orientation $\\theta$"),
        (np.expand_dims(U_pred_glob[2, :], axis=0), "Turning rate $\\omega$")
    ], path_to_pdf=f"{PLOTS_PATH}/evolutions.pdf", samples_per_second=num_steps_per_second)

    anim.plot_animation(path_to_gif=f'{PLOTS_PATH}/animation.gif',
                         path_to_frames_folder=f'{PLOTS_PATH}/grid_frames')


if __name__ == "__main__":
    # run_simulation_maze(start=(0.5, 0, 0.5, 0, 0), goal=(7.5, 7.5), rrt=True) # maze_1 without rrt
    run_simulation_maze(start=(0.5, 0, 0.5, 0, 0), goal=(0.5, 7.5), rrt=True) # maze_2 with rrt