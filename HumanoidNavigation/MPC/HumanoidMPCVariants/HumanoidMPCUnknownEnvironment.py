from HumanoidNavigation.MPC.HumanoidMpc import HumanoidMPC


class HumanoidMPCUnknownEnvironment(HumanoidMPC):
    """
    A subclass of HumanoidMPC, where the robot is not aware of the full map, but it can only perceive the environment
     through a LiDAR system.
    """

    def _get_list_c_and_eta(self, loc_x_k: float, loc_y_k: float, glob_theta_k: float, glob_x_k: float,
                            glob_y_k: float):
        # TODO: make changes here
        return super()._get_list_c_and_eta(loc_x_k, loc_y_k, glob_theta_k, glob_x_k, glob_y_k)
