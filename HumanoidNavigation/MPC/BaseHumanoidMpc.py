import time

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from BaseMpc import MpcSkeleton

from HumanoidNavigation.Utils.obstacles_no_sympy import generate_random_convex_polygon, plot_polygon

"""
    This is the implementation related to a base Humanoid with the 3D Lip Dynamics (preview control)
    humanoid state: [Pc, dPc, ddPc] = [Pc_x, Pc_y, Pc_z, dPc_x, dPc_y, dPc_z, ddPc_x, ddPc_y, ddPc_z] (m=9)
    humanoid control: [dddPc] = [dddPc_x, dddPc_y, dddPc_z] (m=3)
"""

GRAVITY_CONST = 9.81
COM_HEIGHT = 1
ETA = np.sqrt(GRAVITY_CONST/COM_HEIGHT)
ALPHA = 100 # weight related to the tracking of the zmp trajectory

class BasicHumanoidMPC(MpcSkeleton):
    def __init__(self, state_dim=9, control_dim=3, N_horizon=5, N_simul=100, sampling_time=1e-3, goal=None, obstacles=None):
        super().__init__(state_dim, control_dim, N_horizon, N_simul, sampling_time)
        self.goal = goal
        self.obstacles = obstacles

        # to be sure they are defined
        assert(self.goal is not None and self.obstacles is not None)
        
        # define the parameter related to the zmp trajectory
        self.ZMP_ref = self.optim_prob.parameter(3, self.N_horizon + 1)

        self.add_constraints()
        self.cost_function()

    def add_constraints(self):
        # initial position constraint
        self.optim_prob.subject_to(self.X_mpc[:, 0] == self.x0)

        # horizon constraint (via dynamics)
        for k in range(self.N_horizon):
            self.optim_prob.subject_to(self.X_mpc[:, k+1] == self.integrate(self.X_mpc[:, k], self.U_mpc[:, k]))
            
    def get_ZMP_position(self, x_k):
        # x_k is the entire state and NOT the position of the COM 
        Cl = cs.horzcat(1,0,-1/(ETA**2))
        return cs.mtimes(Cl, x_k[0:3,:])


    def cost_function(self):
        # control actions cost
        control_cost = cs.sumsqr(self.U_mpc)
        zmp_tracking_cost = 0
        for k in range(self.N_horizon - 1):
            zmp_tracking_cost += cs.sumsqr(self.get_ZMP_position(self.X_mpc[:, k]) - self.ZMP_ref[:, k])
        zmp_terminal_cost = cs.sumsqr(self.get_ZMP_position(self.X_mpc[:, self.N_horizon]) - self.ZMP_ref[:, self.N_horizon])
        self.optim_prob.minimize(control_cost + ALPHA*zmp_tracking_cost + ALPHA*zmp_terminal_cost)

    def integrate(self, x_k, u_k):
        # these are the equation of the dynamics, but we need to discretize them
        # Al = cs.vertcat(cs.horzcat(0, 1, 0), cs.horzcat(0, 0, 1), cs.horzcat(0, 0, 0))
        # Bl = cs.vertcat(0,0,1)
        # Cl = cs.vertcat(1,0,-eta**-2)
        
        Ad = cs.vertcat(cs.horzcat(1, self.sampling_time, 0.5*self.sampling_time**2), cs.horzcat(0, 1, self.sampling_time), cs.horzcat(0, 0, 1))
        Bd = cs.vertcat((self.sampling_time**3)/6, 0.5*self.sampling_time**2, self.sampling_time)
        
        return cs.vertcat(
            cs.mtimes( Ad, x_k[0:3] + cs.mtimes(Bd, u_k[0]) ), 
            cs.mtimes( Ad, x_k[3:6] + cs.mtimes(Bd, u_k[1]) ), 
            cs.mtimes( Ad, x_k[6:9] + cs.mtimes(Bd, u_k[2]) ))

    def plot(self, X_pred, ZMP_pred, ZMP_ref):
        # X_pred: predicted state (Pc, dPc, ddPc)
        # ZMP_pred: predicted trajectory of the ZMP
        # ZMP_ref: reference trajectory of the ZMP
        plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")
        if self.goal is not None:
            plt.plot(self.goal[0], self.goal[1], marker='x', color="darkorange", label="Goal")
        if self.obstacles is not None:
            plot_polygon(obstacles)
        plt.plot(X_pred[0,:], X_pred[1,:], color="mediumpurple", label="COM Trajectory")
        plt.plot(ZMP_pred[0,:], ZMP_pred[1,:], color="forestgreen", label="ZMP Predicted Trajectory")
        plt.plot(ZMP_ref[0,:], ZMP_ref[1,:], color="coral", label="ZMP Reference Trajectory")
        print(X_pred)
        plt.legend()
        plt.show()


    def simulation(self, zmp_reference):
        # zmp_reference is the entire reference trajectory of the zmp while ZMP_ref is the container used by casadi which only has the 
        # trajectory during the horizon 
        X_pred = np.zeros(shape=(self.state_dim, self.N_simul + 1))
        U_pred = np.zeros(shape=(self.control_dim, self.N_simul))
        ZMP_pred = np.zeros(shape=(3, self.N_simul+1))
        computation_time = np.zeros(self.N_simul)

        for k in range(self.N_simul):
            starting_iter_time = time.time() # CLOCK

            # set x_0
            self.optim_prob.set_value(self.x0, X_pred[:, k])
            
            # set ZMP trajectory
            self.optim_prob.set_value(self.ZMP_ref, zmp_reference[:, k:k+self.N_horizon])

            # solve
            kth_solution = self.optim_prob.solve()

            # get u_0 for x_1
            U_pred[:, k] = kth_solution.value(self.U_mpc[:, 0])

            # assign to X_mpc and U_mpc the relative values
            self.optim_prob.set_initial(self.X_mpc, kth_solution.value(self.X_mpc))
            self.optim_prob.set_initial(self.U_mpc, kth_solution.value(self.U_mpc))

            # compute x_k_next using x_k and u_k
            X_pred[:, k+1] = self.integrate(X_pred[:, k], U_pred[:, k]).full().squeeze(-1)
            ZMP_pred[:, k+1] = self.get_ZMP_position(X_pred[:, k+1])

            computation_time[k] = time.time() - starting_iter_time  # CLOCK

        print(f"Average Computation time: {np.mean(computation_time) * 1000} ms")
        self.plot(X_pred)




if __name__ == "__main__":
    obstacles = generate_random_convex_polygon(5, (10, 11), (10, 11)) # only one
    
    zmp_reference = np.zeros() #TODO    

    mpc = BasicHumanoidMPC(
        state_dim=9,
        control_dim=3,
        N_horizon=5,
        N_simul=1000,
        sampling_time=1e-3,
        goal=(4, 2, 1, 1,1, 0, 0, 0, 0, 0), # position=(4, 0), velocity=(0, 0) theta=0
        obstacles=obstacles
    )

    mpc.simulation(zmp_reference)