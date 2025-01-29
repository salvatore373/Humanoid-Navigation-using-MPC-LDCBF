from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC


class HumanoidMPCUnknownEnvironment(HumanoidMPC):
    """
    A subclass of HumanoidMPC, where the robot is not aware of the full map, but it can only perceive the environment
     through a LiDAR system.
    """
    pass