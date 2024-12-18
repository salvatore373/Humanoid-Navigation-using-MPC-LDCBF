from abc import ABC, abstractmethod

# this base class is used to define the skeleton of an MPC

class MpcSkeleton(ABC):

    def __init__(self):
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