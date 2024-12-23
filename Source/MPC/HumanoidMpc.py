import time
from abc import ABC

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from BaseMpc import MpcSkeleton

"""
    humanoid state: [p_x, v_x, p_y, v_y, theta]
    humanoid control: [f_x, f_y, omega]
"""

GRAVITY_CONST = 9.81
COM_HEIGHT = 1

class HumanoidMPC(MpcSkeleton, ABC):
    def __init__(self, state_dim=5, control_dim=3, N_horizon=40, N_simul=300, sampling_time=1e-3, goal=None):
        super().__init__(state_dim, control_dim, N_horizon, N_simul, sampling_time)
        self.goal = goal

        self.add_constraints()
        self.cost_function()

    def add_constraints(self):
        # initial position constraint
        self.optim_prob.subject_to(self.X_mpc[:, 0] == self.x0)

        # goal constraint (only in position)
        self.optim_prob.subject_to(self.X_mpc[0, self.N_horizon] == self.goal[0])
        self.optim_prob.subject_to(self.X_mpc[2, self.N_horizon] == self.goal[2])


        # horizon constraint (via dynamics)
        for k in range(self.N_horizon):
            self.optim_prob.subject_to(self.X_mpc[:, k+1] == self.lip3d_dynamics(self.X_mpc[:, k], self.U_mpc[:, k]))


        # walking velocities constraint
        # FIXME: leads to infeasible solution
        v_min = [-0.1, 0.1]
        v_max = [0.8, 0.4]
        for k in range(self.N_horizon):
            local_velocities = self.walking_velocities(self.X_mpc[:, k], k)
            self.optim_prob.subject_to(cs.le(local_velocities, v_max))
            self.optim_prob.subject_to(cs.ge(local_velocities, v_min))


        # leg reachability
        l_max = 0.17320508075 # = 0.1 * sqrt(3)
        l_min = -l_max
        for k in range(self.N_horizon):
            reachability = self.leg_reachability(self.X_mpc[:, k])
            self.optim_prob.subject_to(cs.le(reachability, l_max))
            self.optim_prob.subject_to(cs.ge(reachability, l_min))


        # maneuverability constraint
        # FIXME: leads to infeasible solution
        v_max = [0.8, 0.4]
        for k in range(self.N_horizon):
            velocity_term, turning_term = self.maneuverability(self.X_mpc[:, k], self.U_mpc[:, k])
            self.optim_prob.subject_to(cs.le(velocity_term, cs.minus(v_max, turning_term)))


        # TODO: control barrier functions constraint


    def cost_function(self):
        # control actions cost
        control_cost = cs.sumsqr(self.U_mpc)
        # (p_x - g_x)^2 + (p_y - g_y)^2
        distance_cost = cs.sumsqr(self.X_mpc[0] - self.goal[0]) + cs.sumsqr(self.X_mpc[2] - self.goal[1])
        self.optim_prob.minimize(distance_cost + control_cost)

    def integrate(self):
        raise NotImplemented()

    def plot(self):
        # plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")
        # if self.goal is not None:
        #     plt.plot(self.goal[0], self.goal[1], marker='o', color="darkorange", label="Goal")
        # else:
        #     plt.plot(ref[0, :], ref[1, :], color="yellowgreen", label="Reference Trajectory")
        # plt.plot(x_pred[0, :], x_pred[1, :], color="mediumpurple", label="Predicted Trajectory")
        # plt.legend()
        # plt.show()
        raise NotImplemented()

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

            # assign to X_mpc and U_mpc the relative values
            self.optim_prob.set_initial(self.X_mpc, kth_solution.value(self.X_mpc))
            self.optim_prob.set_initial(self.U_mpc, kth_solution.value(self.U_mpc))

            # compute x_k_next using x_k and u_k
            X_pred[:, k+1] = self.lip3d_dynamics(X_pred[:, k], U_pred[:, k]).full().squeeze(-1)

            computation_time[k] = time.time() - starting_iter_time  # CLOCK

        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")
        self.plot(X_pred)

    # ===== HUMANOID SPECIFIC CONSTRAINTS =====
    def lip3d_dynamics(self, x_k, u_k):
        beta = cs.sqrt(GRAVITY_CONST / COM_HEIGHT)

        # Ad = [
        #     [cs.cosh(beta * self.sampling_time), cs.sinh(beta * self.sampling_time) / beta],
        #     [beta * cs.sinh(beta * self.sampling_time), cs.cosh(beta * self.sampling_time)]
        # ]

        Ad11 = cs.cosh(beta * self.sampling_time)
        Ad12 = cs.sinh(beta * self.sampling_time) / beta
        Ad21 = beta * cs.sinh(beta * self.sampling_time)
        Ad22 = cs.cosh(beta * self.sampling_time)

        Al = np.array([
            [Ad11, Ad12, 0, 0, 0],
            [Ad21, Ad22, 0, 0, 0],
            [0, 0, Ad11, Ad12, 0],
            [0, 0, Ad21, Ad22, 0],
            [0, 0, 0, 0, 1]
        ])

        # Bd = [
        #     [1-cs.cosh(beta*self.sampling_time)],
        #     [-beta*cs.sinh(beta*self.sampling_time)]
        # ]

        Bd1 = 1 - cs.cosh(beta * self.sampling_time)
        Bd2 = -beta * cs.sinh(beta * self.sampling_time)

        Bl = np.array([
            [Bd1, 0, 0],
            [Bd2, 0, 0],
            [0, Bd1, 0],
            [0, Bd2, 0],
            [0, 0, self.sampling_time]
        ])

        first_term = cs.mtimes(Al, x_k)
        second_term = cs.mtimes(Bl, u_k)

        # print("##########")
        # print(first_term.shape, Al.shape, x_k.shape)
        # print(second_term.shape, Bl.shape, u_k.shape)

        return first_term + second_term

    def walking_velocities(self, x_k, k):
        theta = x_k[4]
        s_v = 1 if k%2==0 else -1

        local_velocities = cs.vertcat(
            cs.cos(theta)*x_k[1] + cs.sin(theta)*s_v*x_k[3],
            -cs.sin(theta)*x_k[1] + cs.cos(theta)*s_v*x_k[3]
        )

        return local_velocities

    def leg_reachability(self, x_k):
        theta = x_k[4]

        local_positions = cs.vertcat(
            cs.cos(theta)*x_k[0] + cs.sin(theta)*x_k[2],
            -cs.sin(theta)*x_k[0] + cs.cos(theta)*x_k[2]
        )

        return local_positions

    def maneuverability(self, x_k, u_k):
        alpha = 1.44 # or 3.6?
        omega = u_k[2]

        velocity_term = cs.vertcat(
            cs.cos(alpha)*x_k[1],
            cs.sin(alpha)*x_k[3]
        )

        safety_term = alpha/cs.pi * cs.fabs(omega)
        turning_term = cs.vertcat(safety_term, safety_term)

        return velocity_term, turning_term

if __name__ == "__main__":
    mpc = HumanoidMPC(
        state_dim=5, control_dim=3, N_horizon=10, N_simul=300, sampling_time=1e-3, goal=(4, 1, 4, 1, cs.pi)
    )
    mpc.simulation()