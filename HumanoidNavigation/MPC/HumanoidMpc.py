import math
import os
import time
from typing import Union

import casadi as cs
import numpy as np
from scipy.spatial import ConvexHull
from yaml import safe_load

from HumanoidNavigation.Utils.HumanoidAnimationUtils import HumanoidAnimationUtils
from HumanoidNavigation.Utils.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.PlotsUtils import PlotUtils
from HumanoidNavigation.Utils.obstacles import set_seed

this_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.dirname(this_dir)
with open(config_dir + '/config.yml', 'r') as file:
    conf = safe_load(file)
conf["BETA"] = np.sqrt(conf["GRAVITY_CONST"] / conf["COM_HEIGHT"])
conf["OMEGA_MAX"] = 0.156 * cs.pi
conf["OMEGA_MIN"] = -conf["OMEGA_MAX"]

ASSETS_PATH = os.path.dirname(config_dir) + "/Assets/Animations/res.gif"


class HumanoidMPC:
    """
    The implementation of the MPC defined in the paper "Real-Time Safe Bipedal Robot Navigation using Linear Discrete
    Control Barrier Functions" by Peng et al.
    """

    # The drift matrix of the humanoid's dynamic model
    COSH = cs.cosh(conf["BETA"] * conf["DELTA_T"])
    SINH = cs.sinh(conf["BETA"] * conf["DELTA_T"])
    A_l = cs.vertcat(
        cs.horzcat(COSH, SINH / conf["BETA"], 0, 0),
        cs.horzcat(SINH * conf["BETA"], COSH, 0, 0),
        cs.horzcat(0, 0, COSH, SINH / conf["BETA"]),
        cs.horzcat(0, 0, SINH * conf["BETA"], COSH),
    )
    # The control matrix of the humanoid's dynamic model
    B_l = cs.vertcat(
        cs.horzcat(1 - COSH, 0),
        cs.horzcat(-conf["BETA"] * SINH, 0),
        cs.horzcat(0, 1 - COSH),
        cs.horzcat(0, -conf["BETA"] * SINH),
    )

    def __init__(self, goal, obstacles, N_horizon=3, N_mpc_timesteps=100, sampling_time=1e-3,
                 init_state: Union[np.ndarray, tuple[float, float, float, float, float]] = np.array([0, 0, 0, 0, 0]),
                 start_with_right_foot: bool = True, verbosity: int = 1):
        """
        Initialize the MPC.

        :param goal: The position that the humanoid should reach.
        :param obstacles: The obstacles in the environment.
        :param init_state: The initial position, velocity and orientation of the humanoid, as a 5D vector.
        If None, the initial state will be null
        :param N_horizon: The length of the prediction horizon.
        :param N_mpc_timesteps: The maximum number of times the MPC will be triggered in the simulation.
        :param sampling_time: The duration of a single time step in seconds.
        :param start_with_right_foot: Whether the first stance foot should be the right or left one.
        :param verbosity: The level of verbosity of the logs. It ranges between 1 and 3.
        """
        assert conf['DELTA_T'] % sampling_time <= 1e-8, \
            "The sampling time must be lower than and divisible by the duration of the step."

        self.N_horizon = N_horizon
        self.N_simul = N_mpc_timesteps
        self.sampling_time = sampling_time

        # Compute the number of inputs that will be provided to the robot during one simulation timestep
        self.mpc_step = int((conf['DELTA_T'] / self.sampling_time))
        self.mpc_step = 1 if self.mpc_step == 0 else self.mpc_step
        # Compute the number of inputs that will be provided throughout the simulation,
        # based on the robot's sampling time.
        self.num_inputs = self.mpc_step * self.N_simul

        self.start_with_right_foot = start_with_right_foot
        self.verbosity = verbosity

        self.goal = goal
        self.obstacles: list[ConvexHull] = obstacles
        self.list_inferred_obstacles = []
        self.list_lidar_readings = []
        self.precomputed_omega = None
        self.precomputed_theta = None

        # Make sure that the goal and the obstacles are given
        assert (self.goal is not None and self.obstacles is not None)

        self.state_dim = 4
        self.control_dim = 2

        # Create the instance of the optimization problem inside the MPC
        self.optim_prob = cs.Opti()
        p_opts = dict(print_time=False, verbose=True if verbosity > 1 else False, expand=True)
        s_opts = dict(print_level=verbosity, max_iter=5000, constr_viol_tol=1e-5, tol=1e-5)
        self.optim_prob.solver("ipopt", p_opts, s_opts)  # (NLP solver)

        # An array of constants such that the i-th element is 1 if the right foot is the stance at time instant i,
        # -1 if the stance is the left foot.
        self.s_v = []
        # Define which step should be right and which left
        self.s_v_param = self.optim_prob.parameter(1, self.N_horizon + 1)
        for i in range(self.num_inputs + self.N_horizon + 1):
            self.s_v.append(conf["RIGHT_FOOT"] if i % 2 == (0 if start_with_right_foot else 1) else conf["LEFT_FOOT"])

        # Define the state and the control variables (without theta and omega)
        self.X_mpc = self.optim_prob.variable(self.state_dim, self.N_horizon + 1)
        self.U_mpc = self.optim_prob.variable(self.control_dim, self.N_horizon)
        self.x0 = self.optim_prob.parameter(self.state_dim)
        # Define the parameters for theta and omega (that will be precomputed)
        self.X_mpc_theta = self.optim_prob.parameter(1, self.N_horizon + 1)
        self.U_mpc_omega = self.optim_prob.parameter(1, self.N_horizon)
        self.x0_theta = self.optim_prob.parameter(1)

        # Set the initial state (always in the origin)
        if isinstance(init_state, tuple):
            init_state = np.array(init_state)
        assert init_state.shape[0] == 5, "The initial state must be a vector with 5 components."
        self.optim_prob.set_value(self.x0, init_state[:self.state_dim])
        self.optim_prob.set_value(self.x0_theta, init_state[self.state_dim])

        # Define a vector of parameters that can be either 0 or 1, used to activate or deactivate the LCBFs of a
        # specific simulation timestep.
        self.lcbf_activation_params = self.optim_prob.parameter(1, self.num_inputs)
        self.optim_prob.set_value(self.lcbf_activation_params, np.ones(self.num_inputs))

        # Add the constraints to the objective function
        self._add_constraints()

        # Define the cost function of the objective function
        self._add_cost_function()

    def _precompute_theta_omega_naive(self, start_state, start_state_theta):
        """
        Computes the values that the humanoid's state and input should have for theta and omega for the prediction
        horizon in order to reach the goal position. It computes the target theta value as atan2(goal_y-p_y, goal_x-p_x)
        and computes the omega and theta values needed to reach that angle with the current velocity limits.

        :param start_state: The current state of the humanoid's system, defined as (com_x, vel_com_x, com_y, vel_com_y).
        :param start_state_theta: The current orientation of the humanoid.
        """
        # Compute the humanoid's orientation for this prediction horizon
        self.precomputed_theta = [start_state_theta]  # initial theta
        self.precomputed_omega = []
        for k in range(self.N_horizon):
            target_heading_angle = (cs.atan2(self.goal[1] - start_state[2], self.goal[0] - start_state[0])
                                    - self.precomputed_theta[-1])

            # Compute the turning rate for this prediction horizon
            self.precomputed_omega.append(
                cs.fmin(cs.fmax(target_heading_angle, conf["OMEGA_MIN"]), conf["OMEGA_MAX"])
            )

            self.precomputed_theta.append(
                self.precomputed_theta[-1] + self.precomputed_omega[k] * self.sampling_time
            )

    def _compute_walking_velocities_matrix(self, x_k_next, theta_k, k):
        """
        It computes the result of the matrix multiplication in the "walking velocities" constraint defined in the paper
        as the below expression (formula 8).

          | v_x_min | <= | cos(theta_k)     sin(theta_k) |  |   v_x_k+1     |  <= | v_x_max |
          | v_y_min |    | -sin(theta_k)    cos(theta_k) |  | s_v * v_y_k+1 |     | v_y_max |

        :param x_k_next: The state of the humanoid's system at time K+1.
        :param theta_k: The orientation of the humanoid at time K.
        :param k: The time step in the prediction horizon.
        """
        s_v = self.s_v_param[k]

        local_velocities = cs.vertcat(
            cs.cos(theta_k) * x_k_next[1] + cs.sin(theta_k) * 1 * x_k_next[3],
            -cs.sin(theta_k) * x_k_next[1] + cs.cos(theta_k) * s_v * x_k_next[3]
        )

        return local_velocities

    def _compute_leg_reachability_matrix(self, x_next, x_k, theta_k, k):
        """
        It computes the result of the matrix multiplication in the "Leg Reachability" constraint defined in the paper
        as the below expression (formula 9).

          | l_x,min | <= | cos(theta_k)     sin(theta_k) |  |  p_x_{k+1} - p_x_{k}  |  <= | l_max |
          | l_y,min |    | -sin(theta_k)    cos(theta_k) |  |  p_y_{k+1} - p_y_{k}  |     | l_max |

        :param x_k: The state of the humanoid's system at time K.
        :param x_next: The state of the humanoid's system at time K+1.
        :param theta_k: The orientation of the humanoid at time K.
        """
        p_diff = [x_next[0] - x_k[0], x_next[2] - x_k[2]]
        local_positions = cs.vertcat(
            cs.cos(theta_k) * p_diff[0] + cs.sin(theta_k) * p_diff[1],
            -cs.sin(theta_k) * p_diff[0] + cs.cos(theta_k) * p_diff[1]
        )
        local_positions += cs.vertcat(0, self.s_v_param[k] * 0.05)

        return local_positions

    def _compute_maneuverability_terms(self, x_k, theta_k, omega_k):
        """
        It computes the left and the right side of the "Maneuverability" constraint inequality defined in the paper
        as the below expression (formula 10).

          | cos(theta_k)     sin(theta_k) |  |  v_x_k  |   <=   v_x_max - (alpha/pi) * |omega_k|
                                             |  v_y_k  |

        :param x_k: The state of the humanoid's system at time K.
        :param theta_k: The orientation of the humanoid at time K.
        :param omega_k: The turning rate of the humanoid at time K.
        """
        velocity_term = cs.cos(theta_k) * x_k[1] + cs.sin(theta_k) * x_k[3]
        safety_term = conf["V_MAX"][0] - (conf["ALPHA"] / np.pi) * cs.fabs(omega_k)

        return velocity_term, safety_term

    def _add_constraints(self):
        """
        Defines and adds to the MPC the constraints that the solution should satisfy in order to be physically feasible.
        """
        # initial position constraint
        self.optim_prob.subject_to(self.X_mpc[:, 0] == self.x0)

        for k in range(self.N_horizon):
            integration_res = self._integrate(self.X_mpc[:, k], self.U_mpc[:, k])
            self.optim_prob.subject_to(self.X_mpc[:, k + 1] == integration_res)

            # leg reachability
            reachability = self._compute_leg_reachability_matrix(x_k=self.X_mpc[:, k], x_next=self.X_mpc[:, k + 1],
                                                                 theta_k=self.X_mpc_theta[k], k=k)
            self.optim_prob.subject_to(cs.le(reachability, cs.vertcat(conf['L_MAX_X'], conf['L_MAX_Y'])))
            self.optim_prob.subject_to(cs.ge(reachability, cs.vertcat(conf['L_MIN_X'], conf['L_MIN_Y'])))

        for k in range(self.N_horizon):
            # maneuverability constraint
            velocity_term, turning_term = self._compute_maneuverability_terms(self.X_mpc[:, k + 1],
                                                                              self.X_mpc_theta[k + 1],
                                                                              self.U_mpc_omega[k])
            self.optim_prob.subject_to(velocity_term <= turning_term)

        for k in range(1, self.N_horizon + 1):
            # walking velocities constraint
            local_velocities = self._compute_walking_velocities_matrix(self.X_mpc[:, k], self.X_mpc_theta[k], k)
            self.optim_prob.subject_to(local_velocities <= conf["V_MAX"])
            self.optim_prob.subject_to(local_velocities >= conf["V_MIN"])

    @staticmethod
    def _compute_single_lcbf(x, eta, c):
        """
        It returns the value of the LCBF h(x) as defined in formula (16) of the paper.

        :param x: The current position of the humanoid.
        :param eta: The vector normal to the line that defines the half plane of the LCBF, pointing towards the
         direction where h(x) > 0.
        :param c: The point on one obstacle's edge that is closest to x.
        """
        return eta.T @ (x - c)

    def _add_lcbf_constraint(self, simul_k: int, x_k: float, y_k: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Adds the constraint of the Linear Control Barrier Function (LCBF) to the optimization problem for the current
        simulation timestep K.
        This is based on the obstacles that are currently surrounding the robot and on the humanoid's position.

        :param simul_k: The current simulation timestep.
        :param x_k: The CoM X-coordinate of the humanoid at time step k in the simulation.
        :param y_k: The CoM Y-coordinate of the humanoid at time step k in the simulation.
        :returns: The list of the c and eta vectors computed for each obstacle.
        """
        # For each obstacle, determine the point C on the edge closest to the current humanoid's position X,
        # and the normal vector connecting X to C
        list_c, list_norm_vecs = self._get_list_c_and_eta(x_k=x_k, y_k=y_k)
        list_c_and_norm_vecs = list(zip(list_c, list_norm_vecs))

        # Deactivate all the LCBF constraints relative to the previous simulation timesteps
        if simul_k > 0:
            self.optim_prob.set_value(self.lcbf_activation_params[:simul_k], np.zeros(simul_k))

        # Add the control barrier functions constraint
        for k in range(self.N_horizon + 1):
            # Get the vector of the CoM position from the current state
            pos_from_state = np.array([self.X_mpc[0, k], self.X_mpc[2, k]])
            pos_from_state_cs = cs.vertcat(pos_from_state[0], pos_from_state[1])

            # Add one constraint for each obstacle in the map
            for c, normal_vector in list_c_and_norm_vecs:
                lcbf_constr = self._compute_single_lcbf(pos_from_state_cs, normal_vector, c)
                self.optim_prob.subject_to(cs.power(lcbf_constr, self.lcbf_activation_params[simul_k]) >= 0)

        return list_c_and_norm_vecs

    def _get_list_c_and_eta(self, x_k: float, y_k: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        It computes the list of the points C and the normal vectors Eta, which are defined for each obstacle as the
        point on the obstacle's edge that is closest to the robot's position, and the vector from the robot's position
        to C.

        :param x_k: The CoM X-coordinate of the humanoid at time step k in the simulation.
        :param y_k: The CoM Y-coordinate of the humanoid at time step k in the simulation.
        """
        list_c, list_norm_vec = [], []
        # Get the vector of the CoM position from the current state
        pos_from_state = np.array([x_k, y_k])

        # Add one constraint for each obstacle in the map
        for obstacle in self.obstacles:
            # Find c, i.e. the point on the obstacle's edge closest to (com_x, com_y) and compute
            # the normal vector from (com_x, com_y) to c
            c, normal_vector = ObstaclesUtils.get_closest_point_and_normal_vector_from_obs(
                x=pos_from_state, polygon=obstacle, unitary_normal_vector=True,
            )
            list_c.append(c)
            list_norm_vec.append(normal_vector)

        return list_c, list_norm_vec

    def _add_cost_function(self):
        """
        Defines and adds to the MPC the cost function to minimize.
        """
        # Compute state_N
        distance_cost = cs.power(self.X_mpc[0, 0] - self.goal[0], 2) + cs.power(
            self.X_mpc[2, 0] - self.goal[1], 2)
        for k in range(self.N_horizon):
            state_kp1 = self._integrate(self.X_mpc[:, k], self.U_mpc[:, k])
            distance_cost += cs.power(state_kp1[0] - self.goal[0], 2) + cs.power(
                state_kp1[2] - self.goal[1], 2)

        self.optim_prob.minimize(distance_cost)

    def _integrate(self, x_k, u_k):
        """
        Given the state and the input of the humanoid's system at time K, it computes the state at time K+1.

        :param x_k: The state at time K.
        :param u_k: The input at time K.
        :returns: The state at time K+1.
        """
        return (self.A_l @ x_k) + (self.B_l @ u_k)

    def run_simulation(self, path_to_gif: str, make_fast_plot: bool = True, plot_animation: bool = False,
                       fill_animator: bool = True, initial_animator: HumanoidAnimationUtils = None) -> \
            tuple[np.ndarray, np.ndarray, Union[None, HumanoidAnimationUtils]]:
        """
        It executes the MPC. It assumes that the initial state of the humanoid is 0, and it computes the optimal inputs
        to reach the goal. Then, it plots the obtained results.

        :param path_to_gif: The path to the GIF file where the animation of this simulation will be saved.
        :param make_fast_plot: Whether to show a static (though fast) plot of the simulation before the animation.
        :param plot_animation: Whether to show and save the animation or not.
        :param fill_animator: Whether it should provide frame-by-frame data to the animator.
        It must be true if plot_animation is true.
        :param initial_animator: If provided, it will add frame data to this animator, and show the overall frame data.
        :return: A tuple, where the first matrix is the evolution of the state throughout the simulation, while the
        second one is the evolution of the inputs computed by the MPC. The last element is the animator or
        fill_animator is True, None otherwise.
        """
        computation_time = np.zeros(self.num_inputs)  # DEBUG

        assert not plot_animation or fill_animator, "If plot_animation is True, fill_animator must be True too."

        # Initialize the matrices that will hold the evolution of the state and the input throughout the simulation
        X_pred = np.zeros(shape=(self.state_dim + 1, self.num_inputs + 1))
        U_pred = np.zeros(shape=(self.control_dim + 1, self.num_inputs))
        # Get the initial state from x0
        X_pred[:4, 0] = self.optim_prob.value(self.x0)
        X_pred[4, 0] = self.optim_prob.value(self.x0_theta)

        # The list of the lists of vectors c and eta computed for each obstacle at each simulation step
        c_and_eta_lists: list[list[tuple[np.ndarray, np.ndarray]]] = []

        self.optim_prob.set_initial(self.X_mpc[:, 0], X_pred[:4, 0])
        self.optim_prob.set_initial(self.U_mpc[:, 0], U_pred[:2, 0])

        last_obj_fun_val = float('inf')
        for k in range(self.num_inputs):
            starting_iter_time = time.time()  # CLOCK  #DEBUG

            # Check whether in this timestep an MPC solution will be computed
            is_mpc_timestep = k % self.mpc_step == 0

            # Add the LCBF constraints based on the current humanoid's position
            list_of_c_and_eta = self._add_lcbf_constraint(k, x_k=X_pred[0, k], y_k=X_pred[2, k])
            c_and_eta_lists.append(list_of_c_and_eta)

            # Stop searching for the solution if the value of the optimization function with the solution
            # of the previous step is low enough.
            if last_obj_fun_val < 0.05:
                break

            # Set the initial state
            self.optim_prob.set_value(self.x0, X_pred[:4, k])
            self.optim_prob.set_value(self.x0_theta, X_pred[4, k])

            if is_mpc_timestep:
                # Compute how many steps are before the current one
                step_number = math.floor(k / self.mpc_step)
                # Set whether the following steps should be with right or left foot
                self.optim_prob.set_value(self.s_v_param, self.s_v[step_number:step_number + self.N_horizon + 1])

            # Precompute theta and omega for the current prediction horizon
            self.optim_prob.set_value(self.X_mpc_theta[0], X_pred[4, k])
            self._precompute_theta_omega_naive(X_pred[:4, k], X_pred[4, k])
            for i in range(1, self.N_horizon + 1):
                self.optim_prob.set_value(self.X_mpc_theta[i], self.precomputed_theta[i])
            for i in range(self.N_horizon):
                self.optim_prob.set_value(self.U_mpc_omega[i], self.precomputed_omega[i])

            # Compute a new foot position by the optimization process only if in this step a new MPC solution
            # should be generated.
            if is_mpc_timestep:
                try:
                    kth_solution = self.optim_prob.solve()
                    last_obj_fun_val = self.optim_prob.debug.value(self.optim_prob.f)
                except Exception as e:
                    print(f"===== ERROR ({k}) =====")
                    print("===== STATES =====")
                    print(self.optim_prob.debug.value(X_pred))
                    print("===== CONTROLS =====")
                    print(self.optim_prob.debug.value(U_pred))
                    print("===== EXCEPTION =====")
                    print(e)
                    print("===== INFEASIBILITIES =====")
                    print(self.optim_prob.debug.show_infeasibilities())
                    break

            # get u_k
            U_pred[:2, k] = kth_solution.value(self.U_mpc[:, 0])
            U_pred[2, k] = self.precomputed_omega[0]

            # ===== DEBUGGING =====
            # print(kth_solution.value(self.X_mpc))
            # print(kth_solution.value(self.U_mpc))

            if is_mpc_timestep:
                # A new MPC solution has just been computed, then compute x_k_next using x_k and u_k
                state_res = self._integrate(X_pred[:4, k], U_pred[:2, k])
                X_pred[:4, k + 1] = state_res.full().squeeze(-1)
            else:
                # At this timestep only a new omega has been provided to the robot, then the CoM position and velocity
                # did not change.
                X_pred[:4, k + 1] = X_pred[:4, k]
            X_pred[4, k + 1] = self.precomputed_theta[1]

            # Set the initial guesses only if a new MPC solution will be generated at the next timestep
            if k % self.mpc_step == self.mpc_step - 1:
                # Set X_mpc to the state derived from the computed inputs
                self.optim_prob.set_initial(self.X_mpc[:, 0], X_pred[:4, k + 1])
                for i in range(self.N_horizon - 1):
                    state_res = self._integrate(state_res, kth_solution.value(self.U_mpc[:, i + 1]))
                    self.optim_prob.set_initial(self.X_mpc[:, i + 1], state_res)

        # Remove empty portions from X_pred and U_pred
        X_pred = X_pred[:, :k + 1]
        U_pred = U_pred[:, :k]

        computation_time = computation_time[:k + 1]
        computation_time[k] = time.time() - starting_iter_time  # CLOCK
        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")  # DEBUG

        # Get the c and vectors of each simulation timestep
        c_lists = []
        for k, list_k in enumerate(c_and_eta_lists):
            list_c_k = []
            for obs_i_c, obs_i_eta in list_k:
                list_c_k.append(obs_i_c.squeeze())
            c_lists.append(list_c_k)

        # Display the obtained results
        if make_fast_plot:
            HumanoidAnimationUtils.plot_fast_static(X_pred, U_pred, self.goal, self.obstacles,
                                                    np.repeat(self.s_v, self.mpc_step))
        animator = initial_animator
        if fill_animator:
            animator = HumanoidAnimationUtils(goal_position=self.goal,
                                              obstacles=self.obstacles) if animator is None else initial_animator
            for k in range(X_pred.shape[1]):
                animator.add_frame_data(
                    com_position=[X_pred[0, k], X_pred[2, k]],
                    humanoid_orientation=X_pred[4, k],
                    footstep_position=U_pred[:2, k] if k < X_pred.shape[1] - 1 else [None, None],
                    which_footstep=self.s_v[math.floor(k / self.mpc_step)],
                    list_point_c=c_lists[k] if k < X_pred.shape[1] - 1 else c_lists[k - 1],
                    inferred_obstacles=self.list_inferred_obstacles[k] if len(self.list_inferred_obstacles) > 0 else [],
                    lidar_readings=self.list_lidar_readings[k] if len(self.list_lidar_readings) > 0 else []
                )
        if plot_animation:
            animator.plot_animation(path_to_gif)

        return X_pred, U_pred, animator


def main():
    ObstaclesUtils.set_random_seed(4)
    set_seed(4)

    initial_state = (0, 0, 3, 0, 0)
    goal_pos = (6, -3)

    mpc = HumanoidMPC(
        N_horizon=3,
        N_mpc_timesteps=300,
        sampling_time=conf["DELTA_T"],
        # sampling_time=1e-2,
        goal=goal_pos,
        init_state=initial_state,
        obstacles=[],
        verbosity=0,
    )

    X_pred_glob, U_pred_glob, _ = mpc.run_simulation(path_to_gif=ASSETS_PATH, make_fast_plot=True, plot_animation=True)

    diff = X_pred_glob[[0, 2], 1:] - X_pred_glob[[0, 2], :-1]
    rot_diff = PlotUtils.compute_local_velocities(X_pred_glob[4, :-1], diff).T
    print(rot_diff)


if __name__ == "__main__":
    main()
