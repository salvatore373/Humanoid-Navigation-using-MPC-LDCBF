import time 
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from BaseMpc import MpcSkeleton

class DifferentialDriveMPC(MpcSkeleton):
    def __init__(self, state_dim, control_dim, N_horizon, N_simul, sampling_time, goal=None):
        super().__init__(state_dim, control_dim, N_horizon, N_simul, sampling_time)
        # if goal remains None then it's assumed to be a trajectory tracking problem
        self.goal = goal
        if self.goal is None:
            self.reference = self.optim_prob.parameter(self.state_dim-1, self.N_horizon) # in this case we are only defining a cartesian trajectory

        self.add_constraints()
        self.cost_function()

    def integrate(self, X, U):
        # Euler Integration
        return cs.vertcat(U[0]*cs.cos(X[2]), U[0]*cs.sin(X[2]), U[1])

    def add_constraints(self):
        self.optim_prob.subject_to(self.X_mpc[:,0] == self.x0)
        if self.goal is not None:
            self.optim_prob.subject_to(self.X_mpc[:,self.N_horizon] == self.goal)
        for k in range(self.N_horizon):
            self.optim_prob.subject_to(self.X_mpc[:,k+1] == self.X_mpc[:,k] + self.sampling_time*self.integrate(self.X_mpc[:,k], self.U_mpc[:,k]))

    def cost_function(self):
        cost = cs.sumsqr(self.U_mpc) # input effort
        if self.goal is None: # hence we have trajectory tracking
            reference_cost = 0
            for k in range(self.N_horizon-1):
                reference_cost += cs.sumsqr(self.X_mpc[:2,k] - self.reference[:, k])
            terminal_cost = cs.sumsqr(self.X_mpc[:2, -1] - self.reference[:, -1]) # not sure if it's -1 or -2 (double check this)
            weight = 200
            cost += weight*reference_cost + weight*terminal_cost
        self.optim_prob.minimize(cost)

    def plot(self, x_pred, ref):
        # you need to pass x_pred because is not defined inside this class while ref it's only defined as a casadi parameter
        plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")
        if self.goal is not None:
            plt.plot(self.goal[0], self.goal[1], marker='o', color="darkorange", label="Goal")
        else:
            plt.plot(ref[0,:], ref[1,:], color="yellowgreen", label="Reference Trajectory")
        plt.plot(x_pred[0,:], x_pred[1,:], color="mediumpurple", label="Predicted Trajectory")
        plt.legend()
        plt.show()

    def simulation(self, ref=None):
        # you need to pass your reference which will be inserted in your casadi parameter
        # container for the final predicted trajectory
        X_pred = np.zeros(shape=(self.state_dim, self.N_simul+1))
        U_pred = np.zeros(shape=(self.control_dim, self.N_simul))
        computation_time = np.zeros(self.N_simul)
        for k in range(self.N_simul):
            starting_iter_time = time.time()
            self.optim_prob.set_value(self.x0, X_pred[:,k])

            if self.goal is None and ref is not None:
                self.optim_prob.set_value(self.reference, ref[:, k:k+self.N_horizon]) # set the actual reference equal to the reference from k to k+N_horizon
            
            kth_solution = self.optim_prob.solve()
            U_pred[:,k] = kth_solution.value(self.U_mpc[:,0])
            self.optim_prob.set_initial(self.X_mpc, kth_solution.value(self.X_mpc))
            self.optim_prob.set_initial(self.U_mpc, kth_solution.value(self.U_mpc))
            X_pred[:,k+1] = X_pred[:,k] + self.sampling_time*self.integrate(X_pred[:,k], U_pred[:,k]).full().squeeze(-1)
            computation_time[k] = time.time() - starting_iter_time
        
        print(f"Average Computation time: {np.mean(computation_time)*1000} ms")
        self.plot(X_pred, ref)

if __name__ == "__main__":
    problem = "tracking" #"posture" or "tracking"

    if problem != "tracking":
        diffmpc = DifferentialDriveMPC(
            state_dim=3, control_dim=2, N_horizon=40, N_simul=300, sampling_time=0.01, goal=(-3,1,cs.pi)
        )
        diffmpc.simulation()
    elif problem == "posture":
        goal = (-3,1,cs.pi)
        delta_t = 0.01
        N_simul = 1000
        N = 30
        k = 1.8
        alpha_x = k*np.cos(goal[2]) - 3*goal[0]
        beta_x = k*np.cos(goal[2])
        alpha_y = k*np.cos(goal[2]) - 3*goal[1]
        beta_y = k*np.cos(goal[2])
        s = np.linspace(0, 1, N_simul+N+1)
        x_traj = goal[0]*s**3 - alpha_x*(s-1)*s**2 + beta_x*s*(s-1)**2
        y_traj = goal[1]*s**3 - alpha_x*(s-1)*s**2 + beta_y*s*(s-1)**2
        reference = np.array([x_traj, y_traj])
        diffmpc = DifferentialDriveMPC(
            state_dim=3, control_dim=2, N_horizon=N, N_simul=N_simul, sampling_time=delta_t
        )
        diffmpc.simulation(ref=reference)