import time
from abc import ABC

import matplotlib
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sympy.diffgeom.rn import theta

from BaseMpc import MpcSkeleton

from HumanoidNavigation.Utils.obstacles_no_sympy import generate_random_convex_polygon, plot_polygon

""" The width and the height of the rectangle representing the feet in the plot"""
FOOT_RECT_WIDTH = 0.5
FOOT_RECT_HEIGHT = 0.2

"""
    This is the implementation related to our reference paper (3D Lip Dynamics with Heading angle and control barrier functions)
    humanoid state: [p_x, v_x, p_y, v_y, theta] = [com_pos_x, com_vel_x, com_pos_y, com_vel_y, heading_angle]
    humanoid control: [f_x, f_y, omega] = [stance_foot_pos_x, stance_foot_pos_y, turning_rate]
"""
DELTA_T = 1e-3
GRAVITY_CONST = 9.81
COM_HEIGHT = 1
BETA = np.sqrt(GRAVITY_CONST / COM_HEIGHT)
M_CONVERSION = 1000  # everything is expressed wrt meters -> use this to change the unit measure
ALPHA = 3.6  # paper refers to Digit robot (1.44 or 3.6?)
GAMMA = 0.3  # used in CBF
L_MAX = 0.17320508075 * M_CONVERSION  # 0.1*sqrt(3)
V_MIN = [M_CONVERSION * -0.1, M_CONVERSION * -0.1]  # [-0.1, -0.1]
V_MAX = [M_CONVERSION * 0.8, M_CONVERSION * 0.4]  # [0.8, 0.4]

COSH = np.cosh(BETA * DELTA_T)
SINH = np.sinh(BETA * DELTA_T)

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


class HumanoidMPC(MpcSkeleton):
    def __init__(self, goal, obstacles, state_dim=5, control_dim=3, N_horizon=5, N_simul=100, sampling_time=1e-3):
        super().__init__(state_dim, control_dim, N_horizon, N_simul, sampling_time)
        self.goal = goal
        self.obstacles = obstacles
        self.precomputed_omega = None
        self.precomputed_theta = None
        # An array of constants such that the i-th element is 1 if the right foot is the stance at time instant i,
        # -1 if the stance is the left foot.
        self.s_v = [0 for _ in range(self.N_simul)]

        # to be sure they are defined
        assert (self.goal is not None and self.obstacles is not None)

        self.X_mpc = cs.vertcat(self.X_mpc[:4, :], cs.DM([[0 for _ in range(N_horizon + 1)]]))  # SALVO
        self.U_mpc = cs.vertcat(self.U_mpc[:2, :], cs.DM([[0 for _ in range(N_horizon)]]))  # SALVO
        self.x0 = self.optim_prob.parameter(5)  # SALVO
        self.optim_prob.set_value(self.x0, np.zeros((5, 1)))  # SALVO
        self.state_dim = 5  # SALVO
        self.control_dim = 3  # SALVO

        self.parameters_precalculation(self.x0)

        self.add_constraints()
        self.cost_function()

    def parameters_precalculation(self, start_state):
        # we are pre-computing the heading angle as the direction from the current position towards the goal position
        omega_max = 0.156 * cs.pi  # unit: rad/s
        omega_min = -omega_max
        # target_heading_angle = cs.atan2(self.goal[1] - start_state[2], self.goal[0] - start_state[0])
        target_heading_angle = cs.atan2(self.goal[1] - start_state[2], self.goal[0] - start_state[0]) + cs.pi  # SALVO

        # omegas (turning rate)
        self.precomputed_omega = [
            cs.fmin(cs.fmax((target_heading_angle - start_state[4]) / self.N_horizon, omega_min), omega_max)
            # avoid sharp turns
            for _ in range(self.N_horizon)
        ]
        print(self.optim_prob.value(self.precomputed_omega[0]))  # DEBUG

        # thetas (heading angles)
        self.precomputed_theta = [start_state[4]]  # initial theta
        for k in range(self.N_horizon - 1):
            self.precomputed_theta.append(
                self.precomputed_theta[-1] + self.precomputed_omega[k] * self.sampling_time
            )

    def add_constraints(self):
        # initial position constraint
        self.optim_prob.subject_to(self.X_mpc[:, 0] == self.x0)

        # goal constraint (only in position)
        self.optim_prob.subject_to(self.X_mpc[0, self.N_horizon] == self.goal[0])
        self.optim_prob.subject_to(self.X_mpc[2, self.N_horizon] == self.goal[2])

        # horizon constraint (via dynamics)
        for k in range(self.N_horizon):
            self.optim_prob.subject_to(self.X_mpc[:, k + 1] == self.integrate(self.X_mpc[:, k], self.U_mpc[:, k]))

            # leg reachability -> prevent the over-extension of the swing leg
            reachability = self.leg_reachability(self.X_mpc[:, k], k)
            self.optim_prob.subject_to(cs.le(reachability, cs.vertcat(L_MAX, L_MAX)))
            self.optim_prob.subject_to(cs.ge(reachability, cs.vertcat(-L_MAX, -L_MAX)))

            if k > 0:  # defined from 1 to N_horizon (you start still)
                # walking velocities constraint
                local_velocities = self.walking_velocities(self.X_mpc[:, k], k)
                self.optim_prob.subject_to(local_velocities <= V_MAX)
                self.optim_prob.subject_to(local_velocities >= V_MIN)

            # maneuverability constraint (right now using the same v_max of the walking constraint)
            velocity_term, turning_term = self.maneuverability(self.X_mpc[:, k], self.U_mpc[:, k], k)
            self.optim_prob.subject_to(velocity_term <= -turning_term + V_MAX)

            # TODO probably missing constraint related to the foot stance (u[0], u[1]) ~ zmp wrt to com

        # # control barrier functions constraint
        # for k in range(self.N_horizon):
        #     ldcbf_constraints = self.compute_ldcbf(self.X_mpc[:, k], self.obstacles)
        #
        #     for constraint in ldcbf_constraints:
        #         self.optim_prob.subject_to(constraint)

    def cost_function(self):
        # control actions cost
        control_cost = cs.sumsqr(self.U_mpc)
        # (p_x - g_x)^2 + (p_y - g_y)^2 distance of a sequence of predicted states from the goal position
        distance_cost = cs.sumsqr(self.X_mpc[0, :] - self.goal[0]) + cs.sumsqr(
            self.X_mpc[2, :] - self.goal[1])  # not needed to work!
        self.optim_prob.minimize(distance_cost + control_cost)

    def integrate(self, x_k, u_k):
        # discretized model xdot = AD*x + BD*u
        return (AD @ x_k) + (BD @ u_k)

    def plot(self, X_pred, U_pred):
        fix, ax = plt.subplots()

        # TODO: get x0 from the input
        # Plot the start position
        plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")

        # Plot the goal position
        plt.plot(self.goal[0], self.goal[2], marker='o', color="darkorange", label="Goal")

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
                             color='blue' if self.s_v[time_instant] == 1 else 'green', alpha=0.7)
            # Apply rotation
            t = (matplotlib.transforms.Affine2D().rotate(foot_orient) +
                 matplotlib.transforms.Affine2D().translate(step_x, step_y) + ax.transData)
            rect.set_transform(t)

            # Add rectangle to the plot
            ax.add_patch(rect)

        plt.legend()
        # plt.xlim(-1 - self.goal[0], self.goal[0] + 1)
        # plt.ylim(-1 - self.goal[2], self.goal[2] + 1)
        plt.xlim(-1, 12)
        plt.ylim(-1, 12)
        plt.show()

    def simulation(self):
        X_pred = np.zeros(shape=(self.state_dim, self.N_simul + 1))
        U_pred = np.zeros(shape=(self.control_dim, self.N_simul))
        computation_time = np.zeros(self.N_simul)

        for k in range(self.N_simul):
            starting_iter_time = time.time()  # CLOCK

            # set x_0
            self.optim_prob.set_value(self.x0, X_pred[:, k])

            # precompute compute theta and omega
            self.parameters_precalculation(X_pred[:, k])
            for i in range(self.N_horizon):
                self.X_mpc[4, i + 1] = self.precomputed_theta[i]
                self.U_mpc[2, i] = self.precomputed_omega[i]

            # solve
            try:
                kth_solution = self.optim_prob.solve()
            except Exception as e:
                print(f"===== ERROR ({k}) =====")
                print("===== STATES =====")
                print(self.optim_prob.debug.value(X_pred))
                print("===== CONTROLS =====")
                print(self.optim_prob.debug.value(U_pred))
                print("===== EXCEPTION =====")
                print(e)
                exit(1)

            # get u_0 for x_1
            U_pred[:, k] = kth_solution.value(self.U_mpc[:, 0])

            # ===== DEBUGGING =====
            # print(kth_solution.value(self.X_mpc))
            # print(kth_solution.value(self.U_mpc))

            # assign to X_mpc and U_mpc the relative values
            self.optim_prob.set_initial(self.X_mpc, kth_solution.value(self.X_mpc))
            self.optim_prob.set_initial(self.U_mpc, kth_solution.value(self.U_mpc))

            # compute x_k_next using x_k and u_k
            X_pred[:, k + 1] = self.integrate(X_pred[:, k], U_pred[:, k]).full().squeeze(-1)

            computation_time[k] = time.time() - starting_iter_time  # CLOCK

        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")
        self.plot(X_pred, U_pred)

    # ===== PAPER-SPECIFIC CONSTRAINTS =====
    def walking_velocities(self, x_k_next, k):
        theta = self.precomputed_theta[k]
        s_v = 1 if k % 2 == 0 else -1  # s_v = 1 right foot, s_v = -1 left foot
        self.s_v[k] = s_v

        local_velocities = cs.vertcat(
            cs.cos(theta) * x_k_next[1] + cs.sin(theta) * s_v * x_k_next[3],
            -cs.sin(theta) * x_k_next[1] + cs.cos(theta) * s_v * x_k_next[3]
        )

        return local_velocities

    def leg_reachability(self, x_k, k):
        theta = self.precomputed_theta[k]

        local_positions = cs.vertcat(
            cs.cos(theta) * x_k[0] + cs.sin(theta) * x_k[2],
            -cs.sin(theta) * x_k[0] + cs.cos(theta) * x_k[2]
        )

        return local_positions

    def maneuverability(self, x_k, u_k, k):
        # u_k is not used, so we should omit it!
        theta = self.precomputed_theta[k]  # theta = x_k[4]
        omega = self.precomputed_omega[k]  # omega = u_k[2]

        velocity_term = cs.cos(theta) * x_k[1] + cs.sin(theta) * x_k[3]
        safety_term = ALPHA / np.pi * cs.fabs(omega)

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
        # SALVO state_dim=5,
        # SALVO control_dim=3,
        state_dim=4,
        control_dim=2,
        N_horizon=3,
        N_simul=300,
        sampling_time=DELTA_T,
        goal=(1, 0, 10, 0, 0),  # position=(4, 0), velocity=(0, 0) theta=0
        obstacles=obstacles
    )

    mpc.simulation()
