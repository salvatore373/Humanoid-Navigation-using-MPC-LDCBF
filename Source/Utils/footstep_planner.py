from enum import Enum

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# The duration of the single support phase of the humanoids' locomotion loop.
SINGLE_SUPPORT_PHASE_DURATION = 2

# The duration of the double support phase of the humanoids' locomotion loop.
DOUBLE_SUPPORT_PHASE_DURATION = 2

# The displacement to add to the unicycle position to generate the position of the foot
ABS_UNICYCLE_DISPLACEMENT = 0.1


class Foot(Enum):
    """
    A class to identify the feet of the humanoid.
    """
    RIGHT = 0
    LEFT = 1


class Step:
    """
    A class containing all the information about a step
    """

    def __init__(self, position: np.ndarray, orientation: np.ndarray,
                 ss_duration: float, ds_duration: float,
                 support_foot: Foot, timestep: int, ):
        """
        Create a new Step object.

        :param position: A vector in the form (x, y, z) representing the position of the foot's centroid in the space.
        :param orientation: A vector in the form (theta_x, theta_y, theta_z) representing the orientation of the foot
         relative to the X, Y, Z axis of the absolute reference frame.
        :param ss_duration: The duration of the single support phase in this step.
        :param ds_duration: The duration of the double support phase in this step.
        :param support_foot: The foot used as support during this step
        :param timestep: The timestep that this step is associated with, relative to the plan (i.e. the first step will
         have timestep=0).
        """
        self.position = position
        self.orientation = orientation
        self.ss_duration = ss_duration
        self.ds_duration = ds_duration
        self.support_foot = support_foot
        self.timestep = timestep


class FootstepPlanner:
    @staticmethod
    def compute_plan_from_velocities(ref_velocities: np.ndarray, initial_left_foot_pose: np.ndarray,
                                     initial_right_foot_pose: np.ndarray, initial_support_foot: Foot,
                                     sampling_time: float) -> list[Step]:
        """
        Computes the sequence of footsteps that the humanoid must take to travel at the velocity specified in
        ref_velocities, starting with the provided left and right foot poses.

        :param ref_velocities: A list of reference velocities (vx, vy, ω) describing linear and angular velocities at
        each timestep. The shape is (num_time_instants x 3).
        :param initial_left_foot_pose: Initial states of the left foot, containing positional and orientation data in
         the form (x, y, z=0, theta).
        :param initial_right_foot_pose: Initial states of the right foot, containing positional and orientation data in
         the form (x, y, z=0, theta).
        :param initial_support_foot: The foot to use as support for the first step.
        :param sampling_time: The duration of a timestep in the simulation, in seconds.
        :return: The sequence of footsteps that the humanoid must take to travel at the specified ref_velocities.
        """
        # Initialize the unicycle's position as the midpoint between the feet,
        # and its orientation (unicycle_theta) as the average of the feet's orientations.
        unicycle_pos = (initial_left_foot_pose[:2] + initial_right_foot_pose[:2]) / 2.
        unicycle_theta = (initial_left_foot_pose[3] + initial_right_foot_pose[3]) / 2.

        # Compute the pose of all the steps of the plan
        plan = []
        support_foot = initial_support_foot
        for timestep in range(len(ref_velocities)):
            # Set this step duration
            if timestep == 0:
                # In the first step there is no SS phase and the DS phase takes longer
                ss_duration = 0
                ds_duration = (SINGLE_SUPPORT_PHASE_DURATION + DOUBLE_SUPPORT_PHASE_DURATION) * 2
            else:
                ss_duration = SINGLE_SUPPORT_PHASE_DURATION
                ds_duration = DOUBLE_SUPPORT_PHASE_DURATION

            # Compute the motion of the unicycle during the complete step cycle, whom total time is the sum of the SS
            # and DS phases durations.
            for _ in range(ss_duration + ds_duration):
                # Update the unicycle's pose
                if timestep > 1:
                    # Update the unicycle's orientation with theta += ω⋅T_s
                    unicycle_theta += ref_velocities[timestep][2] * sampling_time
                    # Update the unicycle's position (with a rotation matrix) resulting from the application
                    # of (v_x, v_y).
                    rot_mat = np.array([[np.cos(unicycle_theta), - np.sin(unicycle_theta)],
                                        [np.sin(unicycle_theta), np.cos(unicycle_theta)]])
                    unicycle_pos += rot_mat @ ref_velocities[timestep][:2] * sampling_time

            # Compute the step position based on the unicycle position
            curr_foot_displ = ABS_UNICYCLE_DISPLACEMENT if support_foot == Foot.LEFT else - ABS_UNICYCLE_DISPLACEMENT
            displ_x = -np.sin(unicycle_theta) * curr_foot_displ
            displ_y = np.cos(unicycle_theta) * curr_foot_displ
            pos = np.array((  # Define the (x, y, z=0) coordinates of the footstep
                unicycle_pos[0] + displ_x,
                unicycle_pos[1] + displ_y,
                0.))
            # Compute the step orientation based on the unicycle orientation
            ang = np.array((0., 0., unicycle_theta))

            # Add this step to the plan
            plan.append(Step(
                position=pos, orientation=ang,
                ss_duration=ss_duration, ds_duration=ds_duration,
                support_foot=support_foot, timestep=timestep
            ))

            # Switch the support foot
            support_foot = Foot.RIGHT if support_foot == Foot.LEFT else Foot.LEFT

        return plan

    @staticmethod
    def _comp_unicycle_pos_vel_acc_component(s, init_coordinate, goal_coordinate, theta_i, theta_f,
                                             is_for_x: bool = True):
        """
        Given the initial and goal states of the unicycle, compute the x or y path (in terms of position, velocity and
         acceleration) as functions of a parameter time contained in [0, 1].

        :param s: The parameter of the path: a variable contained in [0, 1].
        :param init_coordinate: The initial X or Y coordinate of the unicycle.
        :param goal_coordinate: The final X or Y coordinate of the unicycle.
        :param theta_i: The initial orientation of the unicycle.
        :param theta_f: The final orientation of the unicycle.
        :param is_for_x: Whether it has to compute the path for X or for Y.
        """
        k = 1
        sin_or_cos = np.cos if is_for_x else np.sin
        # Compute the polynomial coefficients
        a = init_coordinate
        b = k * sin_or_cos(theta_i)
        c = 3 * goal_coordinate - 3 * init_coordinate - k * sin_or_cos(theta_f) - 2 * k * sin_or_cos(theta_i)
        d = 2 * init_coordinate - 2 * goal_coordinate + k * sin_or_cos(theta_f) + k * sin_or_cos(theta_i)

        # Plug the coefficients in the polynomials to get the path expression
        return (a + b * s + c * np.pow(s, 2) + d * np.pow(s, 3),  # position path
                b + 2 * c * s + 3 * d * np.pow(s, 2),  # velocity path
                2 * c + 6 * d * s)  # acceleration path

    @staticmethod
    def _compute_unicycle_params(s, initial_state, goal_state):
        """
        Given the initial and goal states of the unicycle, compute x, y, x_dot, y_dot, theta, v, omega as functions of
        a parameter contained in [0, 1].

        :param s: The parameter of the path: a variable contained in [0, 1].
        :param initial_state: The initial configuration of the unicycle, in the form (x, y, theta).
        :param goal_state: The final configuration of the unicycle, in the form (x, y, theta).
        :return:
        """
        # Get the start and goal configurations
        x_i, y_i, theta_i = initial_state
        x_f, y_f, theta_f = goal_state

        # Compute the unicycle parameters
        x, x_dot, x_ddot = FootstepPlanner._comp_unicycle_pos_vel_acc_component(s, x_i, x_f, theta_i, theta_f,
                                                                                is_for_x=True)
        y, y_dot, y_ddot = FootstepPlanner._comp_unicycle_pos_vel_acc_component(s, y_i, y_f, theta_i, theta_f,
                                                                                is_for_x=False)
        theta = np.atan2(y_dot, x_dot)
        v = np.sqrt(np.pow(x_dot, 2) + np.pow(y_dot, 2))
        omega = (y_ddot * x_dot - y_dot * x_ddot) / (np.pow(x_dot, 2) + np.pow(y_dot, 2))

        return (x, x_dot, x_ddot,
                y, y_dot, y_ddot,
                theta, v, omega)

    @staticmethod
    def compute_plan_from_position(start_position: np.ndarray, goal_position: np.ndarray,
                                   initial_left_foot_pose: np.ndarray, initial_right_foot_pose: np.ndarray,
                                   initial_support_foot: Foot, sampling_time: float) -> list[Step]:
        """
        Computes the sequence of footsteps that the humanoid must take to make the midpoint of the feet travel
        from the provided start_position to goal_position, starting with the provided left and right foot poses.

        :param start_position: The vector of the coordinates (x, y) where the midpoint of the feet starts the motion.
        :param goal_position: The vector of the coordinates (x, y) where the midpoint of the feet should be at the end
         of the motion.
        :param initial_left_foot_pose: Initial states of the left foot, containing positional and orientation data in
         the form (x, y, z=0, theta).
        :param initial_right_foot_pose: Initial states of the right foot, containing positional and orientation data in
         the form (x, y, z=0, theta).
        :param initial_support_foot: The foot to use as support for the first step.
        :param sampling_time: The duration of a timestep in the simulation, in seconds.
        :return: The sequence of footsteps that the humanoid must take to travel at the specified ref_velocities.
        """
        s = np.linspace(0, 1, 25)
        init_state = np.insert(start_position, 2, (initial_left_foot_pose[3] + initial_right_foot_pose[3]) / 2.)
        goal_state = np.insert(goal_position, 2, 0)
        (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega) = (
            FootstepPlanner._compute_unicycle_params(s, init_state, goal_state))

        # move_and_plot_unicycle(x, y, theta, path_to_gif='../Assets/Animations/unicycle.gif',
        #                       triangle_height=1, triangle_width=0.8)

        return FootstepPlanner.compute_plan_from_velocities(ref_velocities=np.vstack((x_dot, y_dot, omega)).T,
                                                            initial_left_foot_pose=initial_left_foot_pose,
                                                            initial_right_foot_pose=initial_right_foot_pose,
                                                            initial_support_foot=initial_support_foot,
                                                            sampling_time=sampling_time)


# Function to plot steps
def _plot_steps(steps):
    fig, ax = plt.subplots()
    for step in steps:
        x, y, _ = step.position
        _, _, theta = step.orientation

        # Rectangle dimensions
        rect_width = 0.5
        rect_height = 0.2

        # Create rectangle centered at the position
        rect = Rectangle((-rect_width / 2, -rect_height / 2), rect_width, rect_height,
                         color='blue' if step.support_foot == Foot.RIGHT else 'green', alpha=0.7)

        # Apply rotation
        t = matplotlib.transforms.Affine2D().rotate(theta) + matplotlib.transforms.Affine2D().translate(x,
                                                                                                        y) + ax.transData
        rect.set_transform(t)

        # Add rectangle to the plot
        ax.add_patch(rect)

    # Set aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.grid()
    plt.title("Steps Visualization")
    plt.show()


if __name__ == '__main__':
    plan = FootstepPlanner.compute_plan_from_position(start_position=np.array([0, 0]),
                                                      goal_position=np.array([10, 10]),
                                                      initial_left_foot_pose=np.array(
                                                          [-ABS_UNICYCLE_DISPLACEMENT, 0, 0, 0]),
                                                      initial_right_foot_pose=np.array(
                                                          [+ABS_UNICYCLE_DISPLACEMENT, 0, 0, 0]),
                                                      initial_support_foot=Foot.RIGHT, sampling_time=0.01,
                                                      )
    _plot_steps(plan)

    # Simulation parameters
    num_time_instants = 20  # Number of time instants
    time = np.linspace(0, 1, num_time_instants)

    # Circular trajectory
    radius = 1.0  # Radius of the circle
    omega_circular = 0.5  # Angular velocity for the circular trajectory (rad/s)
    vx_circular = -radius * omega_circular * np.sin(omega_circular * time)
    vy_circular = radius * omega_circular * np.cos(omega_circular * time)
    w_circular = np.full_like(time, omega_circular)  # Constant angular velocity
    # Stack the velocities vectors
    circular_trajectory_ref_vels = np.vstack((vx_circular, vy_circular, w_circular)).T
    # Compute the footsteps plan
    circular_plan = FootstepPlanner.compute_plan_from_velocities(circular_trajectory_ref_vels,
                                                                 np.array([0.9, 0, 0, np.pi / 2]),
                                                                 np.array([1.1, 0, 0, np.pi / 2]),
                                                                 Foot.LEFT, sampling_time=0.001,
                                                                 )
    _plot_steps(circular_plan)

    # Linear trajectory
    vx_linear = np.full_like(time, 1.0)  # Constant velocity in x-direction
    vy_linear = np.zeros_like(time)  # No velocity in y-direction
    w_linear = np.zeros_like(time)  # No angular velocity
    # Stack the velocities vectors
    linear_trajectory_ref_vels = np.vstack((vx_linear, vy_linear, w_linear)).T
