from abc import ABC, abstractmethod
import casadi as cs

# this class is used to define the skeleton of an MPC

class MpcSkeleton(ABC):
    def __init__(self, state_dim: int, control_dim: int, N_horizon: int, N_simul: int, sampling_time: float):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.N_horizon = N_horizon
        self.N_simul = N_simul
        self.sampling_time = sampling_time

        self.optim_prob = cs.Opti()
        p_opts, s_opts = {"ipopt.print_level": 0, "expand": True}, {}
        self.optim_prob.solver("ipopt", p_opts, s_opts) # (NLP solver)

        # definition of the containers used + the parameter related to the initial state
        self.X_mpc = self.optim_prob.variable(self.state_dim, self.N_horizon+1)
        self.U_mpc = self.optim_prob.variable(self.control_dim, self.N_horizon) 
        self.x0 = self.optim_prob.parameter(self.state_dim)

    @abstractmethod
    def integrate(self):
        pass
    
    @abstractmethod
    def add_constraints(self):
        pass

    @abstractmethod
    def cost_function(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def simulation(self):
        pass