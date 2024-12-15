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
        self.kinemodel = cs.Function("km", [self.state, self.controls], [self.statedot])

    def plot(self, state, h=0.1, w=0.05):

        x, y, theta = state

        vert = np.array([[h, 0], [0, w/2], [0, -w/2]]).T # place it in the origin 
        rotation_matrix = np.array([
            [cs.cos(theta), -cs.sin(theta)],
            [cs.sin(theta),  cs.cos(theta)]
        ])

        triangle = (rotation_matrix @ vert).T + np.array([[x, y]]) # rotate then translate
        plt.fill(triangle[:, 0], triangle[:, 1], color='seagreen')