import time
from abc import ABC

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from sympy.diffgeom.rn import theta

from BaseMpc import MpcSkeleton

from HumanoidNavigation.Utils.obstacles_no_sympy import generate_random_convex_polygon, plot_polygon, generate_obstacles

"""
    This is the implementation related to our reference paper (3D Lip Dynamics with Heading angle and control barrier functions)
    humanoid state: [p_x, v_x, p_y, v_y, theta]
    humanoid control: [f_x, f_y, omega]
"""

GRAVITY_CONST = 9.81
COM_HEIGHT = 1

class HumanoidMPC(MpcSkeleton):
    def __init__(self, goal, obstacles, state_dim=5, control_dim=3, N_horizon=5, N_simul=100, sampling_time=1e-3):
        super().__init__(state_dim, control_dim, N_horizon, N_simul, sampling_time)
        self.goal = goal
        self.obstacles = obstacles
        self.precomputed_omega = None
        self.precomputed_theta = None

        # to be sure they are defined
        assert(self.goal is not None and self.obstacles is not None)

        self.parameters_precalculation()

        self.add_constraints()
        self.cost_function()

    def parameters_precalculation(self):
        omega_max = 0.156 * cs.pi # unit: rad/s
        omega_min = -omega_max
        target_heading_angle = cs.atan2(self.goal[1]-self.x0[2], self.goal[0]-self.x0[0])
        # target_heading_angle = cs.atan2(self.goal[1] - self.x0[2], self.goal[0] - self.x0[0]) + cs.pi  # SALVO

        # omegas (turning rate)
        self.precomputed_omega = [
            cs.fmin(cs.fmax((target_heading_angle - self.x0[4]) / self.N_horizon, omega_min), omega_max)
            for _ in range(self.N_horizon)
        ]

        # thetas (heading angles)
        self.precomputed_theta = [self.x0[4]]  # initial theta
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
            self.optim_prob.subject_to(self.X_mpc[:, k+1] == self.integrate(self.X_mpc[:, k], self.U_mpc[:, k]))


        # leg reachability
        l_max = 0.17320508075 * 1000 # = 0.1*sqrt(3)
        l_min = -l_max
        for k in range(self.N_horizon):
            reachability = self.leg_reachability(self.X_mpc[:, k], k)
            self.optim_prob.subject_to(cs.le(reachability, cs.vertcat(l_max, l_max)))
            self.optim_prob.subject_to(cs.ge(reachability, cs.vertcat(l_min, l_min)))



        # walking velocities constraint
        v_min = [1000*-0.1, 1000*-0.1] # [-0.1, -0.1]
        v_max = [1000*0.8, 1000*0.4] # [0.8, 0.4]
        for k in range(1, self.N_horizon):
            local_velocities = self.walking_velocities(self.X_mpc[:, k], k)
            self.optim_prob.subject_to(cs.le(local_velocities, v_max))
            self.optim_prob.subject_to(cs.ge(local_velocities, v_min))



        # maneuverability constraint
        v_x_max = 1000*0.8
        for k in range(self.N_horizon):
            velocity_term, turning_term = self.maneuverability(self.X_mpc[:, k], k)
            self.optim_prob.subject_to(cs.le(velocity_term, cs.minus(v_x_max, turning_term)))


        # control barrier functions constraint
        # for k in range(self.N_horizon):
        #     ldcbf_constraints = self.compute_ldcbf(self.X_mpc[:, k], self.obstacles)
        #
        #     for constraint in ldcbf_constraints:
        #         self.optim_prob.subject_to(constraint)


    def cost_function(self):
        # control actions cost
        control_cost = cs.sumsqr(self.U_mpc)
        # (p_x - g_x)^2 + (p_y - g_y)^2
        distance_cost = cs.sumsqr(self.X_mpc[0] - self.goal[0]) + cs.sumsqr(self.X_mpc[2] - self.goal[1])
        self.optim_prob.minimize(distance_cost + control_cost)

    def integrate(self, x_k, u_k):
        beta = cs.sqrt(GRAVITY_CONST / COM_HEIGHT)

        Ad11 = cs.cosh(beta * self.sampling_time)
        Ad12 = cs.sinh(beta * self.sampling_time) / beta
        Ad21 = beta * cs.sinh(beta * self.sampling_time)
        Ad22 = cs.cosh(beta * self.sampling_time)

        Al = cs.vertcat(
            cs.horzcat(Ad11, Ad12, 0, 0, 0),
            cs.horzcat(Ad21, Ad22, 0, 0, 0),
            cs.horzcat(0, 0, Ad11, Ad12, 0),
            cs.horzcat(0, 0, Ad21, Ad22, 0),
            cs.horzcat(0, 0, 0, 0, 1)
        )

        Bd1 = 1 - cs.cosh(beta * self.sampling_time)
        Bd2 = -beta * cs.sinh(beta * self.sampling_time)

        Bl = cs.vertcat(
            cs.horzcat(Bd1, 0, 0),
            cs.horzcat(Bd2, 0, 0),
            cs.horzcat(0, Bd1, 0),
            cs.horzcat(0, Bd2, 0),
            cs.horzcat(0, 0, self.sampling_time)
        )

        first_term = cs.mtimes(Al, x_k)
        second_term = cs.mtimes(Bl, u_k)

        return first_term + second_term

    def plot(self, X_pred):
        plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")

        if self.goal is not None:
            plt.plot(self.goal[0], self.goal[2], marker='o', color="darkorange", label="Goal")

        if self.obstacles is not None:
            for obstacle in self.obstacles:
                plot_polygon(obstacle)

        plt.plot(X_pred[0,:], X_pred[2,:], color="mediumpurple", label="Predicted Trajectory")
        plt.legend()
        # plt.axis("equal")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(left=-7, right=7)
        plt.ylim(bottom=-0.4, top=10.4)
        plt.show()

    def simulation(self):
        X_pred = np.zeros(shape=(self.state_dim, self.N_simul + 1))
        U_pred = np.zeros(shape=(self.control_dim, self.N_simul))
        computation_time = np.zeros(self.N_simul)

        for k in range(self.N_simul):
            starting_iter_time = time.time() # CLOCK

            # set x_0
            self.optim_prob.set_value(self.x0, X_pred[:, k])

            # solve
            kth_solution = self.optim_prob.solve()


            # get u_0 for x_1
            U_pred[:, k] = kth_solution.value(self.U_mpc[:, 0])

            # ===== DEBUGGING =====
            # print(kth_solution.value(self.X_mpc))
            # print(kth_solution.value(self.U_mpc))

            # assign to X_mpc and U_mpc the relative values
            self.optim_prob.set_initial(self.X_mpc, kth_solution.value(self.X_mpc))
            self.optim_prob.set_initial(self.U_mpc, kth_solution.value(self.U_mpc))

            # compute x_k_next using x_k and u_k
            X_pred[:, k+1] = self.integrate(X_pred[:, k], U_pred[:, k]).full().squeeze(-1)

            computation_time[k] = time.time() - starting_iter_time  # CLOCK

        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")
        self.plot(X_pred)   


    # ===== PAPER-SPECIFIC CONSTRAINTS =====
    def walking_velocities(self, x_k_next, k):
        theta = self.precomputed_theta[k]
        s_v = 1 if k%2==0 else -1

        local_velocities = cs.vertcat(
            cs.cos(theta)*x_k_next[1] + cs.sin(theta)*s_v*x_k_next[3],
            -cs.sin(theta)*x_k_next[1] + cs.cos(theta)*s_v*x_k_next[3]
        )

        return local_velocities

    def leg_reachability(self, x_k, k):
        theta = self.precomputed_theta[k]

        local_positions = cs.vertcat(
            cs.cos(theta)*x_k[0] + cs.sin(theta)*x_k[2],
            -cs.sin(theta)*x_k[0] + cs.cos(theta)*x_k[2]
        )

        return local_positions

    def maneuverability(self, x_k, k):
        alpha = 1.44 # 1.44 or 3.6?
        theta = self.precomputed_theta[k] # theta = x_k[4]
        omega = self.precomputed_omega[k] # omega = u_k[2]

        velocity_term = cs.cos(theta)*x_k[1] + cs.sin(theta)*x_k[3]
        safety_term = alpha/cs.pi * cs.fabs(omega)

        return velocity_term, safety_term

    def compute_ldcbf(self, x_k, obstacles):
        raise NotImplemented()




if __name__ == "__main__":
    obstacles = generate_obstacles(
        start=(0, 0, 0, 0, 0),
        goal=(1, 0, 10, 0, 0),
        num_obstacles=1,
        num_points=5,
        x_range=(1, 2),
        y_range=(3, 8)
    )

    mpc = HumanoidMPC(
        state_dim=5,
        control_dim=3,
        N_horizon=50,
        N_simul=300,
        sampling_time=1e-3,
        goal=(1, 0, 10, 0, 0), # position=(4, 0), velocity=(0, 0) theta=0
        obstacles=obstacles
    )

    mpc.simulation()