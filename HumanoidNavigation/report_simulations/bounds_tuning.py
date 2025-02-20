import itertools
import os

import numpy as np

from HumanoidNavigation.MPC import HumanoidMpc
from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils

PLOTS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/Assets/Simulation1/"


def bounds_tuning():
    best_combin = None
    best_res = float('inf')

    v_max_x = np.arange(0.2, 1, 0.05)
    v_max_y = np.arange(0.2, 0.4, 0.05)
    alpha = np.arange(0.5, 4, 0.1)
    omega_range = np.arange(0.4, 1, 0.05)
    hyperparams_combinations = itertools.product(v_max_x, v_max_y, alpha, omega_range)
    for vx_max, vy_max, a, omega in hyperparams_combinations:
        HumanoidMpc.conf['ALPHA'] = a
        HumanoidMpc.conf['V_MAX'] = [vx_max, vy_max]
        HumanoidMpc.conf['OMEGA_MAX'] = omega
        HumanoidMpc.conf['OMEGA_MIN'] = -omega

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
        X_pred_glob, U_pred_glob, _ = mpc.run_simulation(path_to_gif=f'{PLOTS_PATH}/animation.gif',
                                                         make_fast_plot=False,
                                                         plot_animation=False, fill_animator=False)

        if all((X_pred_glob[[0, 2], -1] - [5, 5]) ** 2 <= 1):
            # val = max(X_pred_glob[3, :].max(), -X_pred_glob[3, :].min())
            val = np.average(np.absolute(X_pred_glob[3, :50]))
            if val < best_res:
                best_res = val
                best_combin = (vx_max, vy_max, a, omega)

    print(f'best_combination: {best_combin}')
    print(f'best_res: {best_res}')
    HumanoidMpc.conf['ALPHA'] = best_combin[2]
    HumanoidMpc.conf['V_MAX'] = [best_combin[0], best_combin[1]]
    HumanoidMpc.conf['OMEGA_MAX'] = best_combin[3]
    HumanoidMpc.conf['OMEGA_MIN'] = -best_combin[3]

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
    bounds_tuning()  # best_combin = (0.8499999999999999, 0.2, 2.3, 0.7999999999999999)
