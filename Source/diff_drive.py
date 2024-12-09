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

        # define your configuration vector [x y theta]' (in casadi you also need to add time t)
        self.x = cs.MX.sym('x')
        self.y = cs.MX.sym('y')
        self.theta = cs.MX.sym('theta')
        self.t = cs.MX.sym('t')
        self.state = cs.vertcat(self.x, self.y, self.theta, self.t) # my state vector

        # define your control vector [v w]'
        self.v = cs.MX.sym("v")
        self.w = cs.MX.sym("w")
        self.controls = cs.vertcat(self.v, self.w) # my control vector

        # Kinematic Model
        self.xdot = self.v*cs.cos(self.theta)
        self.ydot = self.v*cs.sin(self.theta)
        self.thetadot = self.w
        self.tdot = 1
        self.statedot = cs.vertcat(self.xdot, self.ydot, self.thetadot, self.tdot)
        kinemodel = cs.Function("km", [self.state, self.controls], [self.statedot])

    def plot(self, state, h=1, w=0.5):

        x, y, theta = state

        vert = np.array([[h, 0], [0, w/2], [0, -w/2]]).T # place it in the origin 
        rotation_matrix = np.array([
            [cs.cos(theta), -cs.sin(theta)],
            [cs.sin(theta),  cs.cos(theta)]
        ])

        triangle = (rotation_matrix @ vert).T + np.array([[x, y]]) # rotate then translate
        plt.fill(triangle[:, 0], triangle[:, 1], color='seagreen')

    # TODO def animate(self, frame):
            # x_new = state[0][i]
            # y_new = state[1][i]
            # theta_new = state[2][i]

    def integration(self, X, U, type="Euler"):
        if type=="Euler":
            Xk1 = U[0]*cs.cos(X[2]) # Vk*cos(Thetak)
            Yk1 = U[0]*cs.sin(X[2]) # Vk*sin(Thetak)
            Thetak1 = U[1] # Wk
            Tk1 = 1
            return cs.vertcat(Xk1, Yk1, Thetak1, Tk1)
    
        # add later Runge Kutta

    def mpc(self, sampling_time):
        # N : horizon for the MPC
        # N_simul : simulation

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
        optim_prob.subject_to(X[:,self.N] == (2, 3, cs.pi, 0)) # final condition constraint
        for k in range(self.N):
            optim_prob.subject_to(X[:,k+1] == X[:,k] + sampling_time*self.integration(X[:,k], U[:,k])) # related to integration: xk+1 = xk + sampling_time*f()
        # TODO cost function + trajectory
        term_cost = cs.sumsqr(X[:2, -1] - x[:2, -1])
        cost = cs.sumsqr(X[:2, :-1] - x[:2, :-1]) + term_cost
        optim_prob.minimize(cost) # cost function

        iter_time = np.zeros(self.N_simul)
        for k in range(self.N_simul):
            init_time = time.time()
            optim_prob.set_value(x0, x[:, k]) # now the initial condition of the step is the current state
            solution = optim_prob.solve()
            u[:,k] = solution.solve(U[:,0])
            u_pred = solution.value(U)
            x_pred = solution.value(X)
            optim_prob.set_initial(X, x_pred)
            optim_prob.set_initial(U, u_pred)
            x[:,k+1] = x[:,k] + sampling_time * self.integration(x[:,k], u[:,k]) # integration procedure
            iter_time[k] = time.time() - init_time

        return 0

diff = DifferentialDrive(n=4, m=2, N=100, N_simul=200)
print(diff.statedot)
# TODO diff.mpc(sampling_time=0.01)
diff.plot(state=[1, 2, -cs.pi/2])
plt.show()

