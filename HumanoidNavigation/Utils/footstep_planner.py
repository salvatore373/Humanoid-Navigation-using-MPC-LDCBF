from enum import Enum

import matplotlib
import numpy as np
import sympy as sym
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# The duration of the single support phase of the humanoids' locomotion loop.
SINGLE_SUPPORT_PHASE_DURATION = 2

# The duration of the double support phase of the humanoids' locomotion loop.
DOUBLE_SUPPORT_PHASE_DURATION = 2

# The displacement to add to the unicycle position to generate the position of the foot
ABS_UNICYCLE_DISPLACEMENT = 0.2


# DEBUG
def _move_and_plot_unicycle(x_trajectory: np.ndarray, y_trajectory: np.ndarray, theta_trajectory: np.ndarray,
                            path_to_gif: str, triangle_height: float = 0.1, triangle_width: float = 0.05, ):
    """
    Plots the animation of this differential drive moving in a 2D graph along the provided state trajectory.
    The robot is represented by a triangle.
    x_trajectory, y_trajectory and theta_trajectory must have the same dimension.

    :param triangle_height: The height of the triangle representing the robot in the plot.
    :param triangle_width: The width of the triangle representing the robot in the plot.
    :param x_trajectory: The value of the X variable of the state along the trajectory. The i-th component of this
     vector represents the value of X at the i-th time instant.
    :param y_trajectory: The value of the Y variable of the state along the trajectory. The i-th component of this
     vector represents the value of Y at the i-th time instant.
    :param theta_trajectory: The value of the Theta variable of the state along the trajectory. The i-th component of this
     vector represents the value of Theta at the i-th time instant.
    :param path_to_gif: The path to the GIF file where the animation will be saved.
    """
    assert len(x_trajectory) == len(y_trajectory) == len(theta_trajectory), ("Length of x_trajectory, y_trajectory"
                                                                             " and theta_trajectory must be the"
                                                                             " same.")

    # Define the vertices representing a triangle with the base laying on the X-axis and the other vertex on
    # the positive Y-axis.
    vert = np.array([[triangle_height, 0], [0, triangle_width / 2], [0, -triangle_width / 2]]).T
    # Compute the barycenter of the triangle and put it in the origin
    barycenter = np.mean(vert, axis=1)[..., np.newaxis]
    vert = vert - barycenter
    barycenter = np.zeros((2, 1))
    # Define the rotation matrix that gives the robot the appropriate orientation
    rotation_matrix = np.array([
        [np.cos(theta_trajectory), -np.sin(theta_trajectory)],
        [np.sin(theta_trajectory), np.cos(theta_trajectory)]
    ]).squeeze()
    # Define the expression that puts the robot in the appropriate position and orientation
    triangle_poses = rotation_matrix.transpose(2, 0, 1) @ vert + np.array([[x_trajectory, y_trajectory]]).T
    # Define the expression that puts the robot's barycenter in the appropriate position
    barycenter_traj = barycenter + np.array([[x_trajectory, y_trajectory]]).T

    # Set up the plot
    fig, ax = plt.subplots()
    # TODO change this accordingly
    ax.set_xlim(0, 15)  # Set x-axis limits
    ax.set_ylim(0, 15)  # Set y-axis limits
    ax.set_aspect('equal')  # Set equal aspect ratio for accurate proportions

    # Initialize the triangle
    from matplotlib import patches
    triangle_patch = patches.Polygon(triangle_poses[0].T, closed=True, facecolor='green')
    ax.add_patch(triangle_patch)

    # Initialize the plots of the barycenter and its trajectory
    barycenter_point, = ax.plot([], [], 'ro', label="Barycenter")
    # trajectory_line, = ax.plot([], [], 'r-', lw=1, label="Trajectory")
    trajectory_line, = ax.plot([], [], '--k', lw=1, label="Trajectory")

    def update(frame):
        """Update the triangle's vertices, barycenter and trajectory at each frame."""
        triangle_patch.set_xy(triangle_poses[frame].T)  # Update the vertices

        # Update the barycenter position
        barycenter_curr_pos = barycenter_traj[frame]
        barycenter_point.set_data(barycenter_curr_pos[0], barycenter_curr_pos[1])
        # Update the barycenter trajectory
        trajectory_line.set_data(barycenter_traj[:frame + 1, 0], barycenter_traj[:frame + 1, 1])

        return triangle_patch, barycenter_point, trajectory_line

    # Create the animation
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(triangle_poses), interval=200, blit=True)  # 1 frame per second
    ani.save(path_to_gif, writer='ffmpeg')
    # Display the animation
    plt.show()


# DEBUG

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
            if timestep > 1:
                for _ in range(ss_duration + ds_duration):
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
    # def compute_plan_from_uni_state(unicycle_state: np.ndarray, initial_left_foot_pose: np.ndarray,
    #                                 initial_right_foot_pose: np.ndarray, initial_support_foot: Foot,
    #                                 sampling_time: float) -> list[Step]:
    def compute_plan_from_uni_state(unicycle_state: np.ndarray, initial_support_foot: Foot) -> list[Step]:
        """
        Computes the sequence of footsteps that the humanoid must take to travel at the velocity specified in
        ref_velocities, starting with the provided left and right foot poses.

        :param initial_left_foot_pose: Initial states of the left foot, containing positional and orientation data in
         the form (x, y, z=0, theta).
        :param initial_right_foot_pose: Initial states of the right foot, containing positional and orientation data in
         the form (x, y, z=0, theta).
        :param sampling_time: The duration of a timestep in the simulation, in seconds.
        :param unicycle_state: A matrix of shape (num_time_instants x 3), where each element represents the state of
         the unicycle in the form (x, y, theta).
        :param initial_support_foot: The foot to use as support for the first step.
        :return: The sequence of footsteps that the humanoid must take to travel at the specified ref_velocities.
        """
        # Initialize the unicycle's position as the midpoint between the feet,
        # and its orientation (unicycle_theta) as the average of the feet's orientations.
        unicycle_pos = unicycle_state[0][:2]
        unicycle_theta = unicycle_state[0][2]

        # Compute the pose of all the steps of the plan
        plan = []
        support_foot = initial_support_foot
        for timestep in range(len(unicycle_state)):
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
            if timestep > 1:
                for _ in range(ss_duration + ds_duration):
                    # Update the unicycle's orientation with theta += ω⋅T_s
                    # unicycle_theta += ref_velocities[timestep][2] * sampling_time
                    unicycle_theta = unicycle_state[timestep][2]
                    # Update the unicycle's position (with a rotation matrix) resulting from the application
                    # of (v_x, v_y).
                    # rot_mat = np.array([[np.cos(unicycle_theta), - np.sin(unicycle_theta)],
                    #                     [np.sin(unicycle_theta), np.cos(unicycle_theta)]])
                    # unicycle_pos += rot_mat @ ref_velocities[timestep][:2] * sampling_time
                    unicycle_pos = unicycle_state[timestep][:2]

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
        k = 7
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
        theta = sym.atan2(y_dot, x_dot)
        v = sym.sqrt(sym.Pow(x_dot, 2) + sym.Pow(y_dot, 2))
        omega = (y_ddot * x_dot - y_dot * x_ddot) / (sym.Pow(x_dot, 2) + sym.Pow(y_dot, 2))

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
        # s = np.linspace(0, 1, 25)

        # s_aux = sym.symbols('s_aux', nonnegative=True)
        # s = s_aux / (1 + s_aux)

        s = sym.symbols('s', nonnegative=True)

        init_state = np.insert(start_position, 2, (initial_left_foot_pose[3] + initial_right_foot_pose[3]) / 2.)
        goal_state = np.insert(goal_position, 2, 0)
        (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega) = (
            FootstepPlanner._compute_unicycle_params(s, init_state, goal_state))

        # DEBUG from MPC.DifferentialMpc import move_and_plot_unicycle
        # DEBUG move_and_plot_unicycle(x, y, theta, path_to_gif='../Assets/Animations/unicycle.gif',
        # DEBUG                        triangle_height=1, triangle_width=0.8)

        # Use a linear timing law
        t = sym.symbols('t', nonnegative=True)
        alpha = 0.5
        tim_law = alpha * t
        time_interval = np.linspace(0, 1 / alpha, 25)
        # Turn the path into a trajectory
        (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega) = (
            sym.lambdify(t, f.subs(s, tim_law), 'numpy') for f in (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega))
        (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega) = (
            f(time_interval) for f in (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega))

        # DEBUG from MPC.DifferentialMpc import move_and_plot_unicycle
        # move_and_plot_unicycle(x, y, theta, path_to_gif='../Assets/Animations/unicycle.gif',
        #                        triangle_height=1, triangle_width=0.8)

        return FootstepPlanner.compute_plan_from_uni_state(unicycle_state=np.vstack((x, y, theta)).T,
                                                           initial_support_foot=initial_support_foot)

        # return FootstepPlanner.compute_plan_from_velocities(ref_velocities=np.vstack((x_dot, y_dot, omega)).T,
        #                                                     initial_left_foot_pose=initial_left_foot_pose,
        #                                                     initial_right_foot_pose=initial_right_foot_pose,
        #                                                     initial_support_foot=initial_support_foot,
        #                                                     sampling_time=sampling_time)

    @staticmethod
    def plot_steps(steps, custom_fig=None, custom_ax=None):
        if custom_fig is None and custom_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = custom_fig, custom_ax

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
        if custom_fig is None and custom_ax is None:
            ax.set_xlim(-2, 15)
            ax.set_ylim(-2, 15)
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
    FootstepPlanner.plot_steps(plan)
