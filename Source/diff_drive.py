import time
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

class DifferentialDrive():

    def __init__(self, n, m, N, N_simul):
        self.n = n # states dimension
        self.m = m # control dimension
        self.N = N # horizon
        self.N_simul = N_simul # simulation horizon

        # define your configuration vector [x y theta]'
        self.x = cs.MX.sym('x')
        self.y = cs.MX.sym('y')
        self.theta = cs.MX.sym('theta')
        self.state = cs.vertcat(self.x, self.y, self.theta) # my state vector

        # define your control vector [v w]'
        self.v = cs.MX.sym("v")
        self.w = cs.MX.sym("w")
        self.controls = cs.vertcat(self.v, self.w) # my control vector

        # Kinematic Model
        self.xdot = self.v*cs.cos(self.theta)
        self.ydot = self.v*cs.sin(self.theta)
        self.thetadot = self.w
        self.statedot = cs.vertcat(self.xdot, self.ydot, self.thetadot)
        kinemodel = cs.Function("km", [self.state, self.controls], [self.statedot])

    def plot(self, state, h=0.1, w=0.05):

        x, y, theta = state

        vert = np.array([[h, 0], [0, w/2], [0, -w/2]]).T # place it in the origin 
        rotation_matrix = np.array([
            [cs.cos(theta), -cs.sin(theta)],
            [cs.sin(theta),  cs.cos(theta)]
        ])

        triangle = (rotation_matrix @ vert).T + np.array([[x, y]]) # rotate then translate
        plt.fill(triangle[:, 0], triangle[:, 1], color='seagreen')

    def integration(self, X, U, type="Euler"): # gives me the continuos dynamics
        if type=="Euler":
            Xk1 = U[0]*cs.cos(X[2]) # Vk*cos(Thetak)
            Yk1 = U[0]*cs.sin(X[2]) # Vk*sin(Thetak)
            Thetak1 = U[1] # Wk
            return cs.vertcat(Xk1, Yk1, Thetak1)
    
        # add later Runge Kutta

    def mpc(self, goal, sampling_time):
        # N : horizon for the MPC
        # N_simul : simulation time

        optim_prob = cs.Opti() 
        p_opts, s_opts = {"ipopt.print_level": 0, "expand": True}, {}
        optim_prob.solver("ipopt", p_opts, s_opts) # define the solver as ipopt (NLP solver)

        # trajectory
        u = np.zeros((self.m, self.N_simul))
        x = np.zeros((self.n, self.N_simul+1))

        # define the variables of the optimization problem
        X = optim_prob.variable(self.n, self.N+1) # containter for all the states
        U = optim_prob.variable(self.m, self.N) # containter for all the control inputs

        x0 = optim_prob.parameter(self.n) # this is the initial state parameter which will be specified later (useful for MPC) with optim_prob.set_value(x0, value)
        # definitions of constraints for our MPC problem
        optim_prob.subject_to(X[:,0] == x0) # initial condition constraint
        optim_prob.subject_to(X[:,self.N] == goal) # final condition constraint
        for k in range(self.N):
            optim_prob.subject_to(X[:,k+1] == X[:,k] + sampling_time*self.integration(X[:,k], U[:,k])) # related to integration: xk+1 = xk + sampling_time*f()
        optim_prob.minimize(cs.sumsqr(U)) # cost function

        iter_time = np.zeros(self.N_simul)
        for k in range(self.N_simul): # at each instant of simulation I am looking forward N steps
            init_time = time.time()
            optim_prob.set_value(x0, x[:, k]) # now the initial condition of the step is the current state
            print(x0)
            solution = optim_prob.solve()
            u[:,k] = solution.value(U[:,0]) # get the first control action from your predictions
            u_pred = solution.value(U)
            x_pred = solution.value(X)
            optim_prob.set_initial(X, x_pred)
            optim_prob.set_initial(U, u_pred)
            x[:,k+1] = x[:,k] + sampling_time * self.integration(x[:,k], u[:,k]).full().squeeze(-1) # integration procedure
            iter_time[k] = time.time() - init_time

        print('Average computation time: ', np.mean(iter_time) * 1000, ' ms')
        return x

diff = DifferentialDrive(n=3, m=2, N=100, N_simul=300)
goal = (2, 3, cs.pi)
x = diff.mpc(goal, sampling_time=0.01)
diff.plot(state=[1, 2, -cs.pi/2])
plt.plot(x[0,:], x[1,:])
plt.plot(goal[0], goal[1], marker='o')
plt.show()