import time
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

class Mpc():

    def __init__(self, n, m, N, N_simul, sampling_time, goal=None, reference=None):
        self.state_dim = n
        self.control_dim = m
        self.N = N # horizon MPC 
        self.N_simul = N_simul # simulation
        self.delta_t = sampling_time
        self.goal = goal
        self.reference = reference

        self.optim_prob = cs.Opti()
        p_opts, s_opts = {"ipopt.print_level": 0, "expand": True}, {}
        self.optim_prob.solver("ipopt", p_opts, s_opts) # define the solver as ipopt (NLP solver)

        # predicted trajectory over the simulation (not something inside mpc)
        self.u = np.zeros((self.control_dim, self.N_simul))
        self.x = np.zeros((self.state_dim, self.N_simul+1))

        # trajectory predicted during each step from the actual state to the horizon
        self.X = self.optim_prob.variable(self.state_dim, self.N+1)
        self.U = self.optim_prob.variable(self.control_dim, self.N) 

        # initial state (this value will be update at each step)
        self.x0 = self.optim_prob.parameter(self.state_dim)

    def integrate(self, state, input):
        # currently this is only suited for the Differential Drive
        x_k_next = input[0]*cs.cos(state[2])
        y_k_next = input[0]*cs.sin(state[2])
        theta_k_next = input[1]
        return cs.vertcat(x_k_next, y_k_next, theta_k_next)

    def add_constraints(self):
        self.optim_prob.subject_to(self.X[:,0] == self.x0)
        for k in range(self.N):
            self.optim_prob.subject_to(self.X[:,k+1] == self.X[:,k] + self.delta_t*self.integrate(self.X[:,k], self.U[:,k]))
        if self.goal is not None:
            self.optim_prob.subject_to(self.X[:,self.N] == self.goal)

    def simulation(self):
        self.cost_function()
        simulation_time = np.zeros(self.N_simul)
        for k in range(self.N_simul):
            iter_time = time.time()
            self.optim_prob.set_value(self.x0, self.x[:,k])
            solution = self.optim_prob.solve()

            self.u[:,k] = solution.value(self.U[:,0]) # only take the initial control from the prediction in the current horizon
            # these are the trajectories that goes from the current k to k+N (what I predict will happen N steps in the future from now)
            u_horizon = solution.value(self.U)
            x_horizon = solution.value(self.X)

            self.optim_prob.set_initial(self.X, x_horizon)
            self.optim_prob.set_initial(self.U, u_horizon)

            self.x[:, k+1] = self.x[:,k] + self.delta_t*self.integrate(self.x[:,k], self.u[:,k]).full().squeeze(-1)
            simulation_time[k] = time.time() - iter_time

        print('The average computation time is: ', np.mean(simulation_time) * 1000, ' ms')

    def cost_function(self):
        cost_function = cs.sumsqr(self.U)

        if self.reference is not None:
            reference_cost = 0
            for k in range(self.N): # change this everytime look from k to k+N
                reference_cost += cs.sumsqr(self.X[:2, k] - self.reference[:, k]) # xy trajectory
            terminal_cost = self.X[:2, self.N] - self.reference[:, self.N]
            weight = 100
            cost_function += weight*reference_cost + weight*cs.sumsqr(terminal_cost)

        self.optim_prob.minimize(cost_function) # cost function

    def plot(self):
        plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")
        if self.goal is not None:
            plt.plot(self.goal[0], self.goal[1], marker='o', color="darkorange", label="Goal")
        if self.reference is not None:
            plt.plot(self.reference[0,:], self.reference[1,:], color="yellowgreen", label="Reference Trajectory")
        plt.plot(self.x[0,:], self.x[1,:], color="mediumpurple", label="Predicted Trajectory")
        print(self.x)
        plt.legend()
        plt.show()      


delta_t = 0.01
# mpc = Mpc(n=3, m=2, N=10, N_simul=300, sampling_time=delta_t, goal=(4,1.5,cs.pi/2))
# mpc.add_constraints()
# mpc.simulation()
# mpc.plot()

N_simul = 300
psi = np.linspace(0, 2*np.pi, N_simul)
reference = np.array([[-1+np.cos(psi)],[np.sin(psi)]]).squeeze(1)
mpc = Mpc(n=3, m=2, N=10, N_simul=N_simul, sampling_time=delta_t, reference=reference)
mpc.add_constraints()
mpc.simulation()
mpc.plot()