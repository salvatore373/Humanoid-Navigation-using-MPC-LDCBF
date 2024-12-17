import time
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

class Mpc():

    def __init__(self, n, m, N, N_simul, sampling_time, goal=None):
        self.state_dim = n
        self.control_dim = m
        self.N = N # horizon MPC 
        self.N_simul = N_simul # simulation steps
        self.delta_t = sampling_time
        self.goal = goal

        # trajectory obtained after the simulation (is not something inside the mpc)
        self.u = np.zeros((self.control_dim, self.N_simul))
        self.x = np.zeros((self.state_dim, self.N_simul+1))

        self.optim_prob = cs.Opti()
        p_opts, s_opts = {"ipopt.print_level": 0, "expand": True}, {}
        self.optim_prob.solver("ipopt", p_opts, s_opts) # define the solver as ipopt (NLP solver)

        # trajectory predicted by the mpc (from k to k+N)
        self.X_mpc = self.optim_prob.variable(self.state_dim, self.N+1)
        self.U_mpc = self.optim_prob.variable(self.control_dim, self.N) 

        # initial state (this value will be update at each step)
        self.x0 = self.optim_prob.parameter(self.state_dim)
        # reference trajectory (at each simulation step k we take the reference from k to k+N)
        if goal is None:
            self.reference = self.optim_prob.parameter(self.state_dim-1, self.N)
        else :
            self.reference = None

        self.add_constraints()
        self.cost_function()

    def integrate(self, state, input):
        # currently this is only suited for the Differential Drive (Euler Integration)
        x_k_next = input[0]*cs.cos(state[2])
        y_k_next = input[0]*cs.sin(state[2])
        theta_k_next = input[1]
        return cs.vertcat(x_k_next, y_k_next, theta_k_next)

    def add_constraints(self):
        self.optim_prob.subject_to(self.X_mpc[:,0] == self.x0)
        for k in range(self.N):
            self.optim_prob.subject_to(self.X_mpc[:,k+1] == self.X_mpc[:,k] + self.delta_t*self.integrate(self.X_mpc[:,k], self.U_mpc[:,k]))
        if self.goal is not None:
            self.optim_prob.subject_to(self.X_mpc[:,self.N] == self.goal)

    def simulation(self, ref=None):
        simulation_time = np.zeros(self.N_simul)
        for k in range(self.N_simul):
            iter_time = time.time()
            self.optim_prob.set_value(self.x0, self.x[:,k])
            if ref is not None:
                self.optim_prob.set_value(self.reference, ref[:,k:k+self.N])
            solution = self.optim_prob.solve()

            self.u[:,k] = solution.value(self.U_mpc[:,0]) # only take the initial control from the prediction in the current horizon
            # these are the trajectories that goes from the current k to k+N (what I predict will happen N steps in the future from now)
            u_horizon = solution.value(self.U_mpc)
            x_horizon = solution.value(self.X_mpc)

            self.optim_prob.set_initial(self.X_mpc, x_horizon)
            self.optim_prob.set_initial(self.U_mpc, u_horizon)

            self.x[:, k+1] = self.x[:,k] + self.delta_t*self.integrate(self.x[:,k], self.u[:,k]).full().squeeze(-1)
            simulation_time[k] = time.time() - iter_time

        print('The average computation time is: ', np.mean(simulation_time) * 1000, ' ms')
        self.plot(ref)

    def cost_function(self):
        cost_function = cs.sumsqr(self.U_mpc)
        if self.reference is not None: # change the condition later
            reference_cost = 0
            for k in range(self.N-1): # change this everytime look from k to k+N
                reference_cost += cs.sumsqr(self.X_mpc[:2, k] - self.reference[:, k]) # xy trajectory
            terminal_cost = cs.sumsqr(self.X_mpc[:2, -2] - self.reference[:, -1])
            weight = 500
            cost_function += weight*reference_cost + weight*terminal_cost
        self.optim_prob.minimize(cost_function) # cost function

    def plot(self, ref):
        plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")
        if self.goal is not None:
            plt.plot(self.goal[0], self.goal[1], marker='o', color="darkorange", label="Goal")
        if self.reference is not None: # completely useless change asap
            plt.plot(ref[0,:], ref[1,:], color="yellowgreen", label="Reference Trajectory")
        plt.plot(self.x[0,:], self.x[1,:], color="mediumpurple", label="Predicted Trajectory")
        plt.legend()
        plt.show()      

# uncomment this to do posture regulation
goal = (-3,1.5,cs.pi/2)
delta_t = 0.01
mpc = Mpc(n=3, m=2, N=10, N_simul=300, sampling_time=delta_t, goal=goal)
mpc.simulation()

# uncomment this to do trajectory tracking
# goal = (-3,1.5,cs.pi/2)
# delta_t = 0.01
# N_simul = 1000
# N = 30
# k = 1.2
# alpha_x = k*np.cos(goal[2]) - 3*goal[0]
# beta_x = k*np.cos(goal[2])
# alpha_y = k*np.cos(goal[2]) - 3*goal[1]
# beta_y = k*np.cos(goal[2])
# s = np.linspace(0, 1, N_simul+N+1)
# x_traj = goal[0]*s**3 - alpha_x*(s-1)*s**2 + beta_x*s*(s-1)**2
# y_traj = goal[1]*s**3 - alpha_x*(s-1)*s**2 + beta_y*s*(s-1)**2
# reference = np.array([x_traj, y_traj])
# mpc = Mpc(n=3, m=2, N=N, N_simul=N_simul, sampling_time=delta_t)
# mpc.simulation(ref=reference)