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
    # DONE
    def __init__(self, state_dim, control_dim, N_horizon, N_simul, sampling_time, goal):
        super().__init__(state_dim=5, control_dim=3, N_horizon=300, N_simul=10, sampling_time=1e-3)
        self.goal = goal

        self.add_constraints()
        self.cost_function()

    def add_constraints(self):
        # FIXME: ask to Euge
        self.optim_prob.subject_to(self.X_mpc[:, 0] == self.x0)
        # self.optim_prob.subject_to(self.X_mpc[:, 1] == self.x0)

        # goal constraint
        self.optim_prob.subject_to(self.X_mpc[:, self.N_horizon] == self.goal)

        # horizon constraint (via dynamics)
        for k in range(self.N_horizon):
            self.optim_prob.subject_to(self.X_mpc[:, k+1] == self.lip3d_dynamics(self.X_mpc[:, k], self.U_mpc[k]))
        # TODO: Walking velocities constraint
        # TODO: Leg reachability
        # TODO: Maneuverability constraint

    # DONE
    def lip3d_dynamics(self, x_k, u_k):
        beta = cs.sqrt(GRAVITY_CONST/COM_HEIGHT)

        # Ad = [
        #     [cs.cosh(beta * self.sampling_time), cs.sinh(beta * self.sampling_time) / beta],
        #     [beta * cs.sinh(beta * self.sampling_time), cs.cosh(beta * self.sampling_time)]
        # ]
        # print(Ad.shape)

        Al = np.array([
            [cs.cosh(beta * self.sampling_time), cs.sinh(beta * self.sampling_time) / beta, 0, 0, 0],
            [beta * cs.sinh(beta * self.sampling_time), cs.cosh(beta * self.sampling_time), 0, 0, 0],
            [0, 0, cs.cosh(beta * self.sampling_time), cs.sinh(beta * self.sampling_time) / beta, 0],
            [0, 0, beta * cs.sinh(beta * self.sampling_time), cs.cosh(beta * self.sampling_time), 0],
            [0, 0, 0, 0, 1]
        ])
        print(Al.shape)

        # Bd = [
        #     [1-cs.cosh(beta*self.sampling_time)],
        #     [-beta*cs.sinh(beta*self.sampling_time)]
        # ]

        Bl = np.array([
            [1-cs.cosh(beta*self.sampling_time), 0, 0],
            [-beta*cs.sinh(beta*self.sampling_time), 0, 0],
            [0, 1-cs.cosh(beta*self.sampling_time), 0],
            [0, -beta*cs.sinh(beta*self.sampling_time), 0],
            [0, 0, self.sampling_time]
        ])
        print(Bl.shape)




        first_term = cs.mtimes(Al, x_k)
        second_term = cs.mtimes(Bl, u_k)

        return first_term + second_term

    # DONE
    def cost_function(self):
        # (p_x - g_x)^2 + (p_y - g_y)^2
        cost = cs.sumsqr(self.X_mpc[0] - self.goal[0]) + cs.sumsqr(self.X_mpc[2] - self.goal[1])
        self.optim_prob.minimize(cost)

    def integrate(self):
        raise NotImplemented()

    def plot(self, x_pred):
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

            X_pred[:, k+1] = self.lip3d_dynamics(X_pred[:, k], U_pred[k]).full().squeeze(-1)

            computation_time[k] = time.time() - starting_iter_time  # CLOCK

        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")
        self.plot(X_pred)



if __name__ == "__main__":
    mpc = HumanoidMPC(
        state_dim=5, control_dim=3, N_horizon=300, N_simul=10, sampling_time=1e-3, goal=(4, 1, 4, 1, cs.pi)
    )
    mpc.simulation()