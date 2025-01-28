import time

import casadi as cs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull

from HumanoidNavigation.MPC.ObstaclesUtils import ObstaclesUtils
from HumanoidNavigation.Utils.obstacles_no_sympy import plot_polygon

""" The width and the height of the rectangle representing the feet in the plot"""
FOOT_RECT_WIDTH = 0.5
FOOT_RECT_HEIGHT = 0.2

# Definition of some constants related to the humanoid's motion
DELTA_T = 0.4
GRAVITY_CONST = 9.81
COM_HEIGHT = 1
BETA = np.sqrt(GRAVITY_CONST / COM_HEIGHT)
ALPHA = 3.66  # (1.44 or 3.6)
L_MAX = 0.17320508075  # 0.1*sqrt(3)
V_MIN = [-0.1, 0.1]
V_MAX = [0.8, 0.4]


class HumanoidMPC:
    """
    The implementation of the MPC defined in the paper "Real-Time Safe Bipedal Robot Navigation using Linear Discrete
    Control Barrier Functions" by Peng et al.
    """

    # The constants used to represent the left and right feet
    RIGHT_FOOT = 1
    LEFT_FOOT = -1

    # The minimum and maximum turning rate allowed by the humanoid's system
    OMEGA_MAX = 0.156 * cs.pi  # unit: rad/s
    OMEGA_MIN = -OMEGA_MAX

    # The drift matrix of the humanoid's dynamic model
    COSH = cs.cosh(BETA * DELTA_T)
    SINH = cs.sinh(BETA * DELTA_T)
    A_l = cs.vertcat(
        cs.horzcat(COSH, SINH / BETA, 0, 0),
        cs.horzcat(SINH * BETA, COSH, 0, 0),
        cs.horzcat(0, 0, COSH, SINH / BETA),
        cs.horzcat(0, 0, SINH * BETA, COSH),
    )
    # The control matrix of the humanoid's dynamic model
    B_l = cs.vertcat(
        cs.horzcat(1 - COSH, 0),
        cs.horzcat(-BETA * SINH, 0),
        cs.horzcat(0, 1 - COSH),
        cs.horzcat(0, -BETA * SINH),
    )

    def __init__(self, goal, obstacles, N_horizon=3, N_simul=100, sampling_time=1e-3,
                 start_with_right_foot: bool = True, verbosity: int = 1):
        """
        Initialize the MPC.

        :param goal: The position that the humanoid should reach.
        :param obstacles: The obstacles in the environment.
        :param N_horizon: The length of the prediction horizon.
        :param N_simul: The number of time steps in the simulation.
        :param sampling_time: The duration of a single time step in seconds.
        :param start_with_right_foot: Whether the first stance foot should be the right or left one.
        :param verbosity: The level of verbosity of the logs. It ranges between 1 and 3.
        """
        self.N_horizon = N_horizon
        self.N_simul = N_simul
        self.sampling_time = sampling_time

        self.goal = goal
        self.obstacles: list[ConvexHull] = obstacles
        self.precomputed_omega = None
        self.precomputed_theta = None

        # Make sure that the goal and the obstacles are given
        assert (self.goal is not None and self.obstacles is not None)

        self.state_dim = 4
        self.control_dim = 2

        # Create the instance of the optimization problem inside the MPC
        self.optim_prob = cs.Opti()
        p_opts = dict(print_time=False, verbose=True, expand=True)
        s_opts = dict(print_level=verbosity)
        self.optim_prob.solver("ipopt", p_opts, s_opts)  # (NLP solver)

        # An array of constants such that the i-th element is 1 if the right foot is the stance at time instant i,
        # -1 if the stance is the left foot.
        self.s_v = []
        # Define which step should be right and which left
        self.s_v_param = self.optim_prob.parameter(1, self.N_horizon)
        for i in range(self.N_simul + self.N_horizon - 1):
            self.s_v.append(self.RIGHT_FOOT if i % 2 == (0 if start_with_right_foot else 1) else self.LEFT_FOOT)

        # Define the state and the control variables (without theta and omega)
        self.X_mpc = self.optim_prob.variable(self.state_dim, self.N_horizon + 1)
        self.U_mpc = self.optim_prob.variable(self.control_dim, self.N_horizon)
        self.x0 = self.optim_prob.parameter(self.state_dim)
        # Define the parameters for theta and omega (that will be precomputed)
        self.X_mpc_theta = self.optim_prob.parameter(1, self.N_horizon + 1)
        self.U_mpc_omega = self.optim_prob.parameter(1, self.N_horizon)
        self.x0_theta = self.optim_prob.parameter(1)

        # Set the initial state
        self.optim_prob.set_value(self.x0, np.zeros((self.state_dim, 1)))  # TODO: get initial state from input
        self.optim_prob.set_value(self.x0_theta, 0)

        # Define the goal local coordinates as a parameter
        glob_to_loc_mat = HumanoidMPC._get_glob_to_loc_rf_trans_mat(0, 0, 0)  # TODO: get theta,x,y from init state
        self.goal_loc_coords = self.optim_prob.parameter(2, 1)  # goal_x, goal_y
        self.optim_prob.set_value(self.goal_loc_coords, (glob_to_loc_mat @ [self.goal[0], self.goal[1], 1])[:2])

        # Define a vector of parameters that can be either 0 or 1, used to activate or deactivate the LCBFs of a
        # specific simulation timestep.
        self.lcbf_activation_params = self.optim_prob.parameter(1, self.N_simul)
        self.optim_prob.set_value(self.lcbf_activation_params, np.ones(self.N_simul))

        # Add the constraints to the objective function
        self._add_constraints()

        # Define the cost function of the objective function
        self._add_cost_function()

    @staticmethod
    def _get_local_to_glob_rf_trans_mat(theta_k, x_k, y_k):
        """
        Returns the homogeneous matrix to transform a vector from the RF relative to the humanoid's CoM and orientation
        to the inertial RF.

        :param theta_k: The global orientation of the humanoid at step K.
        :param x_k: The global x-coordinate position of the humanoid at step K.
        :param y_k: The global y-coordinate position of the humanoid at step K.
        """
        return np.array([
            [np.cos(theta_k), -np.sin(theta_k), x_k],
            [np.sin(theta_k), np.cos(theta_k), y_k],
            [0, 0, 1, ],
        ])

    @staticmethod
    def _get_glob_to_loc_rf_trans_mat(theta_k, x_k, y_k):
        """
        Returns the homogeneous matrix to transform a vector from the inertial RF
        to the RF relative to the humanoid's CoM and orientation.

        :param theta_k: The global orientation of the humanoid at step K.
        :param x_k: The global x-coordinate position of the humanoid at step K.
        :param y_k: The global y-coordinate position of the humanoid at step K.
        """
        return np.linalg.inv(HumanoidMPC._get_local_to_glob_rf_trans_mat(theta_k, x_k, y_k))

    def _precompute_theta_omega_naive(self, start_state, start_state_theta):
        """
        Computes the values that the humanoid's state and input should have for theta and omega for the prediction
        horizon in order to reach the goal position. It computes the target theta value as atan2(goal_y-p_y, goal_x-p_x)
        and computes the omega and theta values needed to reach that angle with the current velocity limits.

        :param start_state: The current state of the humanoid's system, defined as (com_x, vel_com_x, com_y, vel_com_y).
        :param start_state_theta: The current orientation of the humanoid.
        """
        # we are pre-computing the heading angle as the direction from the current position towards the goal position
        goal_loc_coords = self.optim_prob.value(self.goal_loc_coords)
        target_heading_angle = (cs.atan2(goal_loc_coords[1] - start_state[2], goal_loc_coords[0] - start_state[0])
                                - start_state_theta)

        # Compute the turning rate for this prediction horizon
        self.precomputed_omega = [
            cs.fmin(cs.fmax(target_heading_angle, self.OMEGA_MIN), self.OMEGA_MAX)
            for _ in range(self.N_horizon)
        ]

        # Compute the humanoid's orientation for this prediction horizon
        self.precomputed_theta = [start_state_theta]  # initial theta
        for k in range(self.N_horizon):
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
            cs.cos(theta_k) * x_k_next[1] + cs.sin(theta_k) * s_v * x_k_next[3],
            -cs.sin(theta_k) * x_k_next[1] + cs.cos(theta_k) * s_v * x_k_next[3]
        )

        return local_velocities

    def _compute_leg_reachability_matrix(self, x_k, theta_k):
        """
        It computes the result of the matrix multiplication in the "Leg Reachability" constraint defined in the paper
        as the below expression (formula 9).

          | l_min | <= | cos(theta_k)     sin(theta_k) |  |  p_x_k  |  <= | l_max |
          | l_max |    | -sin(theta_k)    cos(theta_k) |  |  p_y_k  |     | l_max |

        :param x_k: The state of the humanoid's system at time K.
        :param theta_k: The orientation of the humanoid at time K.
        """
        local_positions = cs.vertcat(
            cs.cos(theta_k) * x_k[0] + cs.sin(theta_k) * x_k[2],
            -cs.sin(theta_k) * x_k[0] + cs.cos(theta_k) * x_k[2]
        )

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
        safety_term = V_MAX[0] - (ALPHA / np.pi) * cs.fabs(omega_k)

        return velocity_term, safety_term

    def _add_constraints(self):
        """
        Defines and adds to the MPC the constraints that the solution should satisfy in order to be physically feasible.
        """
        # initial position constraint
        self.optim_prob.subject_to(self.X_mpc[:, 0] == self.x0)

        # horizon constraint (via dynamics)
        for k in range(self.N_horizon):
            integration_res = self._integrate(self.X_mpc[:, k], self.U_mpc[:, k])
            self.optim_prob.subject_to(self.X_mpc[:, k + 1] == integration_res)

            # leg reachability -> prevent the over-extension of the swing leg
            reachability = self._compute_leg_reachability_matrix(self.X_mpc[:, k], self.X_mpc_theta[k])
            self.optim_prob.subject_to(cs.le(reachability, cs.vertcat(L_MAX, L_MAX)))
            self.optim_prob.subject_to(cs.ge(reachability, cs.vertcat(-L_MAX, -L_MAX)))

            # walking velocities constraint
            local_velocities = self._compute_walking_velocities_matrix(self.X_mpc[:, k + 1], self.X_mpc_theta[k], k)
            self.optim_prob.subject_to(local_velocities <= V_MAX)
            self.optim_prob.subject_to(local_velocities >= V_MIN)

            # maneuverability constraint (right now using the same v_max of the walking constraint)
            velocity_term, turning_term = self._compute_maneuverability_terms(self.X_mpc[:, k], self.X_mpc_theta[k],
                                                                              self.U_mpc_omega[k])
            self.optim_prob.subject_to(velocity_term <= turning_term)

    def _add_lcbf_constraint(self, simul_k: int, loc_x_k: float, loc_y_k: float, glob_theta_k: float, glob_x_k: float,
                             glob_y_k: float):
        """
        Adds the constraint of the Linear Control Barrier Function (LCBF) to the optimization problem for the current
        simulation timestep K.
        This is based on the obstacles that are currently surrounding the robot and on the humanoid's position.

        :param simul_k: The current simulation timestep.
        :param loc_x_k: The CoM X-coordinate of the humanoid w.r.t the local RF at time step k in the simulation.
        :param loc_y_k: The CoM Y-coordinate of the humanoid w.r.t the local RF at time step k in the simulation.
        :param glob_theta_k: The orientation of the humanoid w.r.t the inertial RF at time step k in the simulation.
        :param glob_x_k: The CoM X-coordinate of the humanoid w.r.t the inertial RF at time step k in the simulation.
        :param glob_y_k: The CoM Y-coordinate of the humanoid w.r.t the inertial RF at time step k in the simulation.

        :param simul_k: The current simulation timestep.
        :param list_c: The list of the closest point of each obstacle's edge from the humanoid's position. This is the
        result of self._get_list_c_and_eta(). The obstacle should be considered in local coordinates.
        :param list_normal_vectors: The list of the vectors of the humanoid's positions from each obstacle's C point.
        This is the result of self._get_list_c_and_eta(). The obstacle should be considered in local coordinates.
        """
        # For each obstacle, determine the point C on the edge closest to the current humanoid's position X,
        # and the normal vector connecting X to C
        list_c, list_norm_vecs = self._get_list_c_and_eta(
            loc_x_k=loc_x_k, loc_y_k=loc_y_k,
            glob_x_k=glob_x_k, glob_y_k=glob_y_k, glob_theta_k=glob_theta_k,
        )
        list_c_and_norm_vecs = list(zip(list_c, list_norm_vecs))

        # Deactivate all the LCBF constraints relative to the previous simulation timesteps
        if simul_k > 0:
            self.optim_prob.set_value(self.lcbf_activation_params[:simul_k], np.zeros(simul_k))

        # Add the control barrier functions constraint
        for k in range(self.N_horizon):
            # Get the vector of the CoM position from the current state
            pos_from_state = np.array([self.X_mpc[0, k], self.X_mpc[2, k]])
            pos_from_state_cs = cs.vertcat(pos_from_state[0], pos_from_state[1])

            # Add one constraint for each obstacle in the map
            for c, normal_vector in list_c_and_norm_vecs:
                lcbf_constr = normal_vector.T @ (pos_from_state_cs - c)
                self.optim_prob.subject_to(cs.power(lcbf_constr, self.lcbf_activation_params[simul_k]) >= 0)

    def _get_list_c_and_eta(self, loc_x_k: float, loc_y_k: float, glob_theta_k: float, glob_x_k: float,
                            glob_y_k: float):
        """
        It computes the list of the points C and the normal vectors Eta, which are defined for each obstacle as the
        point on the obstacle's edge that is closest to the robot's position, and the vector from the robot's position
        to C.

        :param loc_x_k: The CoM X-coordinate of the humanoid w.r.t the local RF at time step k in the simulation.
        :param loc_y_k: The CoM Y-coordinate of the humanoid w.r.t the local RF at time step k in the simulation.
        :param glob_theta_k: The orientation of the humanoid w.r.t the inertial RF at time step k in the simulation.
        :param glob_x_k: The CoM X-coordinate of the humanoid w.r.t the inertial RF at time step k in the simulation.
        :param glob_y_k: The CoM Y-coordinate of the humanoid w.r.t the inertial RF at time step k in the simulation.
        """
        list_c, list_norm_vec = [], []
        # Get the vector of the CoM position from the current state
        pos_from_state = np.array([loc_x_k, loc_y_k])

        # Add one constraint for each obstacle in the map
        for obstacle in self.obstacles:
            #  Convert the obstacle's points in the local RF (i.e. the one of the state)
            local_obstacle = ObstaclesUtils.transform_obstacle_from_glob_to_loc_coords(
                obstacle=obstacle, transformation_matrix=self._get_glob_to_loc_rf_trans_mat(
                    glob_theta_k, glob_x_k, glob_y_k
                )
            )
            # Find c, i.e. the point on the obstacle's edge closest to (com_x, com_y) and compute
            # the normal vector from (com_x, com_y) to c
            c, normal_vector = ObstaclesUtils.get_closest_point_and_normal_vector_from_obs(
                x=pos_from_state, polygon=local_obstacle, unitary_normal_vector=True,
            )
            list_c.append(c)
            list_norm_vec.append(normal_vector)

        return list_c, list_norm_vec

    def _add_cost_function(self):
        """
        Defines and adds to the MPC the cost function to minimize.
        """
        # control actions cost
        # control_cost = cs.sumsqr(cs.vertcat(self.U_mpc, self.U_mpc_omega))
        control_cost = cs.sumsqr(self.U_mpc)

        # (p_x - g_x)^2 + (p_y - g_y)^2 distance of a sequence of predicted states from the goal position
        # distance_cost = (cs.sumsqr(self.X_mpc[0, :-1] - cs.DM.ones(1, self.N_horizon) * self.goal[0])
        #                  + cs.sumsqr(self.X_mpc[2, :-1] - cs.DM.ones(1, self.N_horizon) * self.goal[2]))

        # Compute state_N
        distance_cost = cs.power(self.X_mpc[0, 0] - self.goal_loc_coords[0], 2) + cs.power(
            self.X_mpc[2, 0] - self.goal_loc_coords[1], 2)
        for k in range(self.N_horizon):
            state_kp1 = self._integrate(self.X_mpc[:, k], self.U_mpc[:, k])
            distance_cost += cs.power(state_kp1[0] - self.goal_loc_coords[0], 2) + cs.power(
                state_kp1[2] - self.goal_loc_coords[1], 2)
        # distance_cost = (cs.sumsqr(self.X_mpc[0, 1:] - cs.DM.ones(1, self.N_horizon) * self.goal[0])
        #                  + cs.sumsqr(self.X_mpc[2, 1:] - cs.DM.ones(1, self.N_horizon) * self.goal[2]))

        self.optim_prob.minimize(distance_cost)
        # self.optim_prob.minimize(distance_cost + control_cost)

    def _integrate(self, x_k, u_k):
        """
        Given the state and the input of the humanoid's system at time K, it computes the state at time K+1.

        :param x_k: The state at time K.
        :param u_k: The input at time K.
        :returns: The state at time K+1.
        """
        return (self.A_l @ x_k) + (self.B_l @ u_k)

    def _plot(self, state_glob, input_glob):
        """
        It plots the trajectory of the CoM contained in state_glob and the footsteps prints contained in input_glob.

        :param state_glob: A matrix of shape (5xN_simul+1) that represents the state of the humanoid dynamic system at
        any instant of the simulation. The coordinates system is the one of the inertial RF.
        :param input_glob: A matrix of shape (3xN_simul) that represents the input of the humanoid dynamic system at
        any instant of the simulation. The coordinates system is the one of the inertial RF.
        """
        fix, ax = plt.subplots()

        # Plot the start position
        plt.plot(state_glob[0, 0], state_glob[2, 0], marker='o', color="cornflowerblue", label="Start")

        # Plot the goal position
        plt.plot(self.goal[0], self.goal[1], marker='o', color="darkorange", label="Goal")

        # Plot the obstacles
        for obstacle in self.obstacles:
            plot_polygon(obstacle.points[obstacle.vertices])

        # Plot the trajectory of the CoM computed by the MPC
        plt.plot(state_glob[0, :], state_glob[2, :], color="mediumpurple", label="Predicted Trajectory")

        # Plot the footsteps plan computed by the MPC
        for time_instant, (step_x, step_y, _) in enumerate(input_glob.T):
            foot_orient = state_glob[4, time_instant]

            # Create rectangle centered at the position
            rect = Rectangle((-FOOT_RECT_WIDTH / 2, -FOOT_RECT_HEIGHT / 2), FOOT_RECT_WIDTH, FOOT_RECT_HEIGHT,
                             color='blue' if self.s_v[time_instant] == self.RIGHT_FOOT else 'green', alpha=0.7)
            # Apply rotation
            t = (matplotlib.transforms.Affine2D().rotate(foot_orient) +
                 matplotlib.transforms.Affine2D().translate(step_x, step_y) + ax.transData)
            rect.set_transform(t)

            # Add rectangle to the plot
            ax.add_patch(rect)

        plt.legend()
        plt.xlim(-5, 7)
        plt.ylim(-2, 12)
        plt.show()

    def run_simulation(self):
        """
        It executes the MPC. It assumes that the initial state of the humanoid is 0, and it computes the optimal inputs
        to reach the goal. Then, it plots the obtained results.
        """
        # Initialize the matrices that will hold the evolution of the state and the input throughout the simulation
        X_pred = np.zeros(shape=(self.state_dim + 1, self.N_simul + 1))
        U_pred = np.zeros(shape=(self.control_dim + 1, self.N_simul))
        computation_time = np.zeros(self.N_simul)

        # The position of the CoM in the global frame at each step of the simulation
        X_pred_glob = np.zeros(shape=(self.state_dim + 1, self.N_simul + 1))
        # The position of the footsteps in the global frame at each step of the simulation
        U_pred_glob = np.zeros(shape=(self.control_dim + 1, self.N_simul))

        last_obj_fun_val = float('inf')
        for k in range(self.N_simul):
            # Stop searching for the solution if the value of the optimization function with the solution
            # of the previous step is low enough.
            if last_obj_fun_val < 0.05:
                X_pred_glob = X_pred_glob[:, :k + 1]
                U_pred_glob = U_pred_glob[:, :k + 1]
                break

            starting_iter_time = time.time()  # CLOCK

            # Add the LCBF constraints based on the current humanoid's position
            self._add_lcbf_constraint(k, loc_x_k=X_pred[0, k], loc_y_k=X_pred[2, k],
                                      glob_x_k=X_pred_glob[0, k], glob_y_k=X_pred_glob[2, k],
                                      glob_theta_k=X_pred_glob[4, k])

            # Set the initial state
            self.optim_prob.set_value(self.x0, X_pred[:4, k])
            self.optim_prob.set_value(self.x0_theta, X_pred[4, k])

            # Set whether the following steps should be with right or left foot
            self.optim_prob.set_value(self.s_v_param, self.s_v[k:k + self.N_horizon])

            # Precompute theta and omega for the current prediction horizon
            self._precompute_theta_omega_naive(X_pred[:4, k], X_pred[4, k])
            for i in range(self.N_horizon + 1):
                self.optim_prob.set_value(self.X_mpc_theta[i], self.precomputed_theta[i])
            for i in range(self.N_horizon):
                self.optim_prob.set_value(self.U_mpc_omega[i], self.precomputed_omega[i])

            # solve
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
                exit(1)

            # get u_k
            U_pred[:2, k] = kth_solution.value(self.U_mpc[:, 0])
            U_pred[2, k] = self.precomputed_omega[0]

            # Compute u_k in the global frame
            trans_mat_loc_to_glob = self._get_local_to_glob_rf_trans_mat(X_pred_glob[4, k],
                                                                         X_pred_glob[0, k],
                                                                         X_pred_glob[2, k])
            U_pred_glob[:2, k] = (trans_mat_loc_to_glob @ np.append(U_pred[:2, k], 1))[:2]
            U_pred_glob[2, k] = self.precomputed_omega[0]

            # ===== DEBUGGING =====
            # print(kth_solution.value(self.X_mpc))
            # print(kth_solution.value(self.U_mpc))

            # compute x_k_next using x_k and u_k
            state_res = self._integrate(X_pred[:4, k], U_pred[:2, k])
            X_pred[:4, k + 1] = state_res.full().squeeze(-1)
            X_pred[4, k + 1] = self.precomputed_theta[1]

            # Compute x_k_next in the global frame
            rot_mat_loc_to_glob = self._get_local_to_glob_rf_trans_mat(X_pred_glob[4, k],
                                                                       X_pred_glob[0, k],
                                                                       X_pred_glob[2, k])[:2, :2]
            glob_pos = (trans_mat_loc_to_glob @ [X_pred[0, k + 1], X_pred[2, k + 1], 1])[:2]
            glob_vel = (rot_mat_loc_to_glob @ [X_pred[1, k + 1], X_pred[3, k + 1]])
            X_pred_glob[:4, k + 1] = [glob_pos[0], glob_vel[0], glob_pos[1], glob_vel[1]]
            X_pred_glob[4, k + 1] = self.precomputed_theta[1] + X_pred_glob[4, k]

            # Set X_mpc to the state derived from the computed inputs
            self.optim_prob.set_initial(self.X_mpc[:, 0], X_pred[:4, k + 1])
            for i in range(self.N_horizon - 1):
                state_res = self._integrate(state_res, kth_solution.value(self.U_mpc[:, i + 1]))
                self.optim_prob.set_initial(self.X_mpc[:, i + 1], state_res)

            # Move the goal wrt the RF with origin in p_k_next and orientation theta_next
            glob_to_loc_trans_mat = self._get_glob_to_loc_rf_trans_mat(X_pred_glob[4, k + 1],
                                                                       X_pred_glob[0, k + 1],
                                                                       X_pred_glob[2, k + 1])
            goal_loc_coords = (glob_to_loc_trans_mat @ [self.goal[0], self.goal[1], 1])[:2]
            self.optim_prob.set_value(self.goal_loc_coords, goal_loc_coords)

            computation_time[k] = time.time() - starting_iter_time  # CLOCK

        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")
        self._plot(X_pred_glob, U_pred_glob)


if __name__ == "__main__":
    # only one and very far away
    # obstacle1 = ConvexHull(np.array([[-0.5, 2], [-0.5, 4], [2, 2], [2, 4]]))
    obstacle1 = ObstaclesUtils.generate_random_convex_polygon(5, (-0.5, 0.5), (2, 4))
    # obstacle2 = ObstaclesUtils.generate_random_convex_polygon(5, (-0.5, 0.5), (2, 4))

    mpc = HumanoidMPC(
        N_horizon=3,
        N_simul=300,
        sampling_time=DELTA_T,
        goal=(0, 5),
        obstacles=[
            obstacle1,
            # obstacle2,
        ],
        verbosity=1
    )

    mpc.run_simulation()
