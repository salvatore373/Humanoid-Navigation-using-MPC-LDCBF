import time

import casadi as cs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from HumanoidNavigation.Utils.obstacles_no_sympy import generate_random_convex_polygon, plot_polygon

""" The width and the height of the rectangle representing the feet in the plot"""
FOOT_RECT_WIDTH = 0.5
FOOT_RECT_HEIGHT = 0.2

"""
    This is the implementation related to our reference paper (3D Lip Dynamics with Heading angle and control barrier functions)
    humanoid state: [p_x, v_x, p_y, v_y, theta] = [com_pos_x, com_vel_x, com_pos_y, com_vel_y, heading_angle]
    humanoid control: [f_x, f_y, omega] = [stance_foot_pos_x, stance_foot_pos_y, turning_rate]
"""
# SALVO DELTA_T = 1e-3
DELTA_T = 0.4  # SALVO
GRAVITY_CONST = 9.81
COM_HEIGHT = 1
BETA = np.sqrt(GRAVITY_CONST / COM_HEIGHT)
M_CONVERSION = 1  # everything is expressed wrt meters -> use this to change the unit measure
ALPHA = 3.6  # paper refers to Digit robot (1.44 or 3.6?)
GAMMA = 0.3  # used in CBF
L_MAX = 0.17320508075 * M_CONVERSION  # 0.1*sqrt(3)
V_MIN = [M_CONVERSION * -0.1, M_CONVERSION * 0.1]  # [-0.1, 0.1]
V_MAX = [M_CONVERSION * 0.8, M_CONVERSION * 0.4]  # [0.8, 0.4]

COSH = cs.cosh(BETA * DELTA_T)
SINH = cs.sinh(BETA * DELTA_T)

AD = cs.vertcat(
    cs.horzcat(COSH, SINH / BETA, 0, 0, 0),
    cs.horzcat(SINH * BETA, COSH, 0, 0, 0),
    cs.horzcat(0, 0, COSH, SINH / BETA, 0),
    cs.horzcat(0, 0, SINH * BETA, COSH, 0),
    cs.horzcat(0, 0, 0, 0, 1)
)

BD = cs.vertcat(
    cs.horzcat(1 - COSH, 0, 0),
    cs.horzcat(-BETA * SINH, 0, 0),
    cs.horzcat(0, 1 - COSH, 0),
    cs.horzcat(0, -BETA * SINH, 0),
    cs.horzcat(0, 0, DELTA_T)
)


class HumanoidMPC:
    def __init__(self, goal, obstacles, N_horizon=5, N_simul=100, sampling_time=1e-3):
        self.N_horizon = N_horizon
        self.N_simul = N_simul
        self.sampling_time = sampling_time

        self.optim_prob = cs.Opti()
        p_opts = dict(print_time=False, verbose=True, expand=True)
        s_opts = dict(print_level=3)
        self.optim_prob.solver("ipopt", p_opts, s_opts)  # (NLP solver)

        self.goal = goal
        self.obstacles = obstacles
        self.precomputed_omega = None
        self.precomputed_theta = None
        # An array of constants such that the i-th element is 1 if the right foot is the stance at time instant i,
        # -1 if the stance is the left foot.

        # to be sure they are defined
        assert (self.goal is not None and self.obstacles is not None)

        self.state_dim = 4
        self.control_dim = 2

        # Define the state and the control variables (without theta and omega)
        self.X_mpc = self.optim_prob.variable(self.state_dim, self.N_horizon + 1)
        self.U_mpc = self.optim_prob.variable(self.control_dim, self.N_horizon)
        self.x0 = self.optim_prob.parameter(self.state_dim)
        # Define the parameters for theta and omega (that will be precomputed)
        self.X_mpc_theta = self.optim_prob.parameter(1, self.N_horizon + 1)
        self.U_mpc_omega = self.optim_prob.parameter(1, self.N_horizon)
        self.x0_theta = self.optim_prob.parameter(1)

        # Set the initial state
        self.optim_prob.set_value(self.x0, np.zeros((self.state_dim, 1)))  # DEBUG
        self.optim_prob.set_value(self.x0_theta, 0)

        # Define the goal local coordinates as a parameter
        glob_to_loc_mat = HumanoidMPC.get_glob_to_loc_rf_trans_mat(0, 0, 0)  # TODO: get theta,x,y from init state
        self.goal_loc_coords = self.optim_prob.parameter(2, 1)  # goal_x, goal_y
        self.optim_prob.set_value(self.goal_loc_coords, (glob_to_loc_mat @ [self.goal[0], self.goal[1], 1])[:2])

        # Precompute theta and omega
        self.parameters_precalculation(self.x0, self.x0_theta)

        # Add the constraints to the objective function
        self.add_constraints()

        # Define the cost function of the objective function
        self.cost_function()

    @staticmethod
    def get_local_to_glob_rf_trans_mat(theta_k, x_k, y_k):
        return np.array([
            [np.cos(theta_k), -np.sin(theta_k), x_k],
            [np.sin(theta_k), np.cos(theta_k), y_k],
            [0, 0, 1, ],
        ])

    @staticmethod
    def get_glob_to_loc_rf_trans_mat(theta_k, x_k, y_k):
        return np.linalg.inv(HumanoidMPC.get_local_to_glob_rf_trans_mat(theta_k, x_k, y_k))

    def parameters_precalculation(self, start_state, start_state_theta):
        # we are pre-computing the heading angle as the direction from the current position towards the goal position
        omega_max = 0.156 * cs.pi  # unit: rad/s
        omega_min = -omega_max
        goal_loc_coords = self.optim_prob.value(self.goal_loc_coords)
        target_heading_angle = cs.atan2(goal_loc_coords[1] - start_state[2], goal_loc_coords[0] - start_state[0])

        # omegas (turning rate)
        self.precomputed_omega = [
            cs.fmin(cs.fmax((target_heading_angle - start_state_theta) / self.N_horizon, omega_min), omega_max)
            # avoid sharp turns
            for _ in range(self.N_horizon)
        ]

        # thetas (heading angles)
        self.precomputed_theta = [start_state_theta]  # initial theta
        for k in range(self.N_horizon):
            self.precomputed_theta.append(
                self.precomputed_theta[-1] + self.precomputed_omega[k] * self.sampling_time
            )

    def add_constraints(self):
        # initial position constraint
        self.optim_prob.subject_to(self.X_mpc[:, 0] == self.x0)

        # goal constraint (only in position)
        # self.optim_prob.subject_to(self.X_mpc[0, self.N_horizon] == self.goal[0])
        # self.optim_prob.subject_to(self.X_mpc[2, self.N_horizon] == self.goal[2])

        # horizon constraint (via dynamics)
        for k in range(self.N_horizon):
            integration_res = self.integrate(self.X_mpc[:, k], self.U_mpc[:, k],
                                             self.X_mpc_theta[k], self.U_mpc_omega[k])
            self.optim_prob.subject_to(self.X_mpc[:, k + 1] == integration_res)
            # self.optim_prob.subject_to(
            #     cs.eq(
            #         cs.vertcat(self.X_mpc[:, k + 1]),
            #         integration_res
            #     )
            # )

            # leg reachability -> prevent the over-extension of the swing leg
            reachability = self.leg_reachability(self.X_mpc[:, k], self.X_mpc_theta[k])
            self.optim_prob.subject_to(cs.le(reachability, cs.vertcat(L_MAX, L_MAX)))
            self.optim_prob.subject_to(cs.ge(reachability, cs.vertcat(-L_MAX, -L_MAX)))

            # walking velocities constraint
            # local_velocities = self.walking_velocities(self.X_mpc[:, k + 1], self.X_mpc_theta[k], k)
            # self.optim_prob.subject_to(local_velocities <= V_MAX)
            # self.optim_prob.subject_to(local_velocities >= V_MIN)

            # maneuverability constraint (right now using the same v_max of the walking constraint)
            # velocity_term, turning_term = self.maneuverability(self.X_mpc[:, k], self.X_mpc_theta[k],
            #                                                    self.U_mpc_omega[k])
            # self.optim_prob.subject_to(velocity_term <= turning_term)

        # control barrier functions constraint
        # for k in range(self.N_horizon):
        #     ldcbf_constraints = self.compute_ldcbf(self.X_mpc[:, k], self.obstacles)
        #
        #     for constraint in ldcbf_constraints:
        #         self.optim_prob.subject_to(constraint)

    def cost_function(self):
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
            state_kp1 = self.integrate(self.X_mpc[:, k], self.U_mpc[:, k],
                                       self.X_mpc_theta[k], self.U_mpc_omega[k])
            distance_cost += cs.power(state_kp1[0] - self.goal_loc_coords[0], 2) + cs.power(
                state_kp1[2] - self.goal_loc_coords[1], 2)
        # distance_cost = (cs.sumsqr(self.X_mpc[0, 1:] - cs.DM.ones(1, self.N_horizon) * self.goal[0])
        #                  + cs.sumsqr(self.X_mpc[2, 1:] - cs.DM.ones(1, self.N_horizon) * self.goal[2]))

        self.optim_prob.minimize(distance_cost)
        # self.optim_prob.minimize(distance_cost + control_cost)

    def integrate(self, x_k, u_k, theta_k, omega_k):
        """
        :returns Al[:4,:4] * X_woTheta + Bl[:4, :] * U_woOmega, Al[4,4]*theta + Bl[4,2]*U_omega
        """
        Al1 = cs.vertcat(
            cs.horzcat(COSH, SINH / BETA, 0, 0),
            cs.horzcat(SINH * BETA, COSH, 0, 0),
            cs.horzcat(0, 0, COSH, SINH / BETA),
            cs.horzcat(0, 0, SINH * BETA, COSH),
            # DEBUG cs.horzcat(COSH, SINH / BETA, 0, 0),
            # DEBUG cs.horzcat(SINH / BETA, COSH, 0, 0),
            # DEBUG cs.horzcat(0, 0, COSH, SINH / BETA),
            # DEBUG cs.horzcat(0, 0, SINH / BETA, COSH),
        )
        Al2 = 1

        Bl1 = cs.vertcat(
            cs.horzcat(1 - COSH, 0),
            cs.horzcat(-BETA * SINH, 0),
            cs.horzcat(0, 1 - COSH),
            cs.horzcat(0, -BETA * SINH),
            # DEBUG cs.horzcat(1 + COSH, 0),
            # DEBUG cs.horzcat(+BETA * SINH, 0),
            # DEBUG cs.horzcat(0, 1 + COSH),
            # DEBUG cs.horzcat(0, +BETA * SINH),
        )
        Bl2 = DELTA_T

        # return (Al1 @ x_k) + (Bl1 @ u_k), Al2 * theta_k + Bl2 * omega_k
        return (Al1 @ x_k) + (Bl1 @ u_k)

    def plot(self, X_pred, U_pred):
        fix, ax = plt.subplots()

        # Plot the start position
        plt.plot(X_pred[0, 0], X_pred[2, 0], marker='o', color="cornflowerblue", label="Start")

        # Plot the goal position
        plt.plot(self.goal[0], self.goal[1], marker='o', color="darkorange", label="Goal")

        # Plot the obstacles
        plot_polygon(obstacles)

        # Plot the trajectory of the CoM computed by the MPC
        plt.plot(X_pred[0, :], X_pred[2, :], color="mediumpurple", label="Predicted Trajectory")

        # Plot the footsteps plan computed by the MPC
        # plt.scatter(U_pred[0, :], U_pred[1, :], marker='o', color="forestgreen", label="ZMP")
        for time_instant, (step_x, step_y, _) in enumerate(U_pred.T):
            foot_orient = X_pred[4, time_instant]

            # Create rectangle centered at the position
            rect = Rectangle((-FOOT_RECT_WIDTH / 2, -FOOT_RECT_HEIGHT / 2), FOOT_RECT_WIDTH, FOOT_RECT_HEIGHT,
                             color='blue' if time_instant % 2 == 0 else 'green', alpha=0.7)
            # Apply rotation
            t = (matplotlib.transforms.Affine2D().rotate(foot_orient) +
                 matplotlib.transforms.Affine2D().translate(step_x, step_y) + ax.transData)
            rect.set_transform(t)

            # Add rectangle to the plot
            ax.add_patch(rect)

        plt.legend()
        # plt.xlim(-1 - self.goal[0], self.goal[0] + 1)
        # plt.ylim(-1 - self.goal[2], self.goal[2] + 1)
        plt.xlim(-5, 7)
        plt.ylim(-2, 12)
        plt.show()

    def simulation(self):
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
            if last_obj_fun_val < 1e-2:
                break

            starting_iter_time = time.time()  # CLOCK

            # set x_0
            self.optim_prob.set_value(self.x0, X_pred[:4, k])
            self.optim_prob.set_value(self.x0_theta, X_pred[4, k])

            # precompute compute theta and omega
            self.parameters_precalculation(X_pred[:4, k], X_pred[4, k])
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
                exit(1)

            # get u_k
            U_pred[:2, k] = kth_solution.value(self.U_mpc[:, 0])
            U_pred[2, k] = self.precomputed_omega[0]

            # Compute u_k in the global frame
            trans_mat_loc_to_glob = self.get_local_to_glob_rf_trans_mat(X_pred_glob[4, k],
                                                                        X_pred_glob[0, k],
                                                                        X_pred_glob[2, k])
            U_pred_glob[:2, k] = (trans_mat_loc_to_glob @ np.append(U_pred[:2, k], 1))[:2]
            # U_pred_glob[2, k] = self.precomputed_omega[0] + (
            #     U_pred[2, k - 1] if k > 0 else 0)  # TODO: check this formula
            U_pred_glob[2, k] = self.precomputed_omega[0]

            # ===== DEBUGGING =====
            # print(kth_solution.value(self.X_mpc))
            # print(kth_solution.value(self.U_mpc))

            # assign to X_mpc and U_mpc the relative values
            # self.optim_prob.set_initial(self.X_mpc, kth_solution.value(self.X_mpc))
            # self.optim_prob.set_initial(self.U_mpc, kth_solution.value(self.U_mpc))

            # compute x_k_next using x_k and u_k
            state_res = self.integrate(X_pred[:4, k], U_pred[:2, k],
                                       self.precomputed_theta[0], self.precomputed_omega[0])
            X_pred[:4, k + 1] = state_res.full().squeeze(-1)
            X_pred[4, k + 1] = self.precomputed_theta[1]

            # Compute x_k_next in the global frame
            rot_mat_loc_to_glob = self.get_local_to_glob_rf_trans_mat(X_pred_glob[4, k],
                                                                      X_pred_glob[0, k],
                                                                      X_pred_glob[2, k])[:2, :2]
            glob_pos = (trans_mat_loc_to_glob @ [X_pred[0, k + 1], X_pred[2, k + 1], 1])[:2]
            glob_vel = (rot_mat_loc_to_glob @ [X_pred[1, k + 1], X_pred[3, k + 1]])
            X_pred_glob[:4, k + 1] = [glob_pos[0], glob_vel[0], glob_pos[1], glob_vel[1]]
            X_pred_glob[4, k + 1] = self.precomputed_theta[1] + X_pred_glob[4, k]  # TODO: check this formula

            # Set X_mpc to the state derived from the computed inputs
            self.optim_prob.set_initial(self.X_mpc[:, 0], X_pred[:4, k + 1])
            for i in range(self.N_horizon - 1):
                state_res = self.integrate(state_res, kth_solution.value(self.U_mpc[:, i + 1]),
                                           self.precomputed_theta[0], self.precomputed_omega[0])
                self.optim_prob.set_initial(self.X_mpc[:, i + 1], state_res)

            # Move the goal wrt the RF with origin in p_k_next and orientation theta_next
            glob_to_loc_trans_mat = self.get_glob_to_loc_rf_trans_mat(X_pred_glob[4, k + 1],
                                                                      X_pred_glob[0, k + 1],
                                                                      X_pred_glob[2, k + 1])
            goal_loc_coords = (glob_to_loc_trans_mat @ [self.goal[0], self.goal[1], 1])[:2]
            self.optim_prob.set_value(self.goal_loc_coords, goal_loc_coords)

            computation_time[k] = time.time() - starting_iter_time  # CLOCK

        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")
        self.plot(X_pred_glob, U_pred_glob)

    # ===== PAPER-SPECIFIC CONSTRAINTS =====
    def walking_velocities(self, x_k_next, theta_k, k):
        s_v = 1 if k % 2 == 0 else -1  # s_v = 1 right foot, s_v = -1 left foot

        local_velocities = cs.vertcat(
            cs.cos(theta_k) * x_k_next[1] + cs.sin(theta_k) * s_v * x_k_next[3],
            -cs.sin(theta_k) * x_k_next[1] + cs.cos(theta_k) * s_v * x_k_next[3]
        )

        return local_velocities

    def leg_reachability(self, x_k, theta_k):
        local_positions = cs.vertcat(
            cs.cos(theta_k) * x_k[0] + cs.sin(theta_k) * x_k[2],
            -cs.sin(theta_k) * x_k[0] + cs.cos(theta_k) * x_k[2]
        )

        return local_positions

    def maneuverability(self, x_k, theta_k, omega_k):
        velocity_term = cs.cos(theta_k) * x_k[1] + cs.sin(theta_k) * x_k[3]
        safety_term = V_MAX[0] - (ALPHA / np.pi) * cs.fabs(omega_k)

        return velocity_term, safety_term

    def compute_ldcbf(self, x_k, obstacle_vertices):
        robot_position = cs.vertcat(x_k[0], x_k[2])  # [px, py]

        constraints = []
        for i in range(len(obstacle_vertices)):
            # i-th and (i+1)-th vertices
            vertex = cs.MX(obstacle_vertices[i])
            next_vertex = cs.MX(obstacle_vertices[(i + 1) % len(obstacle_vertices)])

            # get the edge between previous vertices
            edge_vector = next_vertex - vertex

            # get the normal vector to such edge vector
            normal = cs.vertcat(-edge_vector[1], edge_vector[0])
            normal /= cs.norm_2(normal)

            to_robot = robot_position - vertex
            # closest point to the robot in the edge:
            #       0=cos(90°)=vertex, 1=cos(0°)=next_vertex
            # all middle values are the point (in percentage)
            # that lies on edge and is the closest to the robot
            projection = cs.dot(to_robot, edge_vector) / cs.norm_2(edge_vector) ** 2
            projection = cs.fmax(0.0, cs.fmin(projection, 1.0))

            # moving along edge_vector from vertex by a projection
            # between 0 and 1, so it is a percentage of edge_vector
            closest_point = vertex + projection * edge_vector

            # LDCBF condition: normal vector points away from edge (obv)
            # so, the following dot product tells us if we are in the safe
            # region (>0) or not (<0)
            h = cs.dot(normal, (robot_position - closest_point))

            h_next = h + GAMMA * h
            constraints.append(h_next >= 0)

        return constraints


if __name__ == "__main__":
    # only one and very far away
    obstacles = generate_random_convex_polygon(5, (3, 4), (13, 4))

    mpc = HumanoidMPC(
        N_horizon=3,
        N_simul=150,
        sampling_time=DELTA_T,
        goal=(3, 0),
        obstacles=obstacles
    )

    step0 = [0, 0, 0, 0]
    input0 = [-0.19396392946460198, 0.0]
    step1 = mpc.integrate(step0, input0, 0, 0)
    input1 = [0.7343380395416466, 0]
    step2 = mpc.integrate(step1, input1, 0, 0)
    input2 = [-0.3879278589277293, 0]
    step3 = mpc.integrate(step2, input2, 0, 0)
    print()

    mpc.simulation()
