from typing import Callable

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from HumanoidNavigation.Utils.BaseAnimationHelper import BaseAnimationHelper

STEP_SIZE = 6
# The maximum number of previous CoM position to plot
NUMBER_OF_SHADOWS = 10
# The maximum number of previous footsteps to plot
NUMBER_OF_FOOTSTEPS_SHADOWS = NUMBER_OF_SHADOWS // 2
# The distance between the right and the left feet
FOOT_DISTANCE = 3
NUMBER_OF_FOOTSTEPS = 20


class HumanoidAnimationHelper(BaseAnimationHelper):
    # Assuming that the plot is a square, the minimum and maximum value that x and y coordinated can have
    MIN_COORD = -50
    MAX_COORD = +50
    # The length of the tick in the line representing the motion of the CoM
    TICK_LENGTH = 1

    def __init__(self, start_conf, goal_conf, zmp_trajectory: np.ndarray, com_reference: np.ndarray,
                 zmp_reference: np.ndarray, following_com_position=None, following_left_foot=None,
                 following_right_foot=None):
        """
        Initializes a new instance of the utility to show the animation regarding the humanoid.

        :param start_conf: The start configuration of the humanoid, represented by a 3-components vector:
         X-coord, Y-coord, orientation.
        :param goal_conf: The goal configuration, represented by a 3-components vector: X-coord, Y-coord, orientation.
        :param following_com_position: The CoM positions starting from the configuration after the start configuration.
        :param following_left_foot: The left foot positions starting from the configuration after the start
         configuration.
        :param following_right_foot: The right foot positions starting from the configuration after the start
         configuration.
        :param zmp_trajectory: The trajectory of the ZMP. A numpy array of shape (num_animation_frame, 2).
        :param com_reference: The reference trajectory of the CoM. A numpy array of shape (num_animation_frame, 2).
        :param zmp_reference: The reference trajectory of the ZMP. A numpy array of shape (num_animation_frame, 2).
        """
        # Initialize the plots
        self.fig = plt.figure(figsize=(10, 6))
        # gs = GridSpec(2, 3, figure=self.fig, width_ratios=[3, 1, 1], height_ratios=[1, 1])
        gs = GridSpec(4, 2, figure=self.fig, width_ratios=[3, 1], height_ratios=[1, 1, 1, 1])
        # Animation plot (spans 2 rows in the first column)
        ax1 = self.fig.add_subplot(gs[:, 0])  # Full left side for animation
        ax1.set_aspect('equal')
        ax1.set_title("Animation of Humanoid")  # todo: change title
        self.anim_ax = ax1
        # Second plot (upper part of the second column)
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.set_title("CoM trajectory and reference")
        self.com_traj_ref_ax = ax2
        # Third plot (lower part of the second column)
        ax3 = self.fig.add_subplot(gs[1, 1])
        ax3.set_title("CoM error")
        self.com_err_ax = ax3
        # Fourth plot (upper part of the third column)
        # ax4 = self.fig.add_subplot(gs[0, 2])
        ax4 = self.fig.add_subplot(gs[2, 1])
        ax4.set_title("ZMP trajectory and reference")
        self.zmp_traj_ref_ax = ax4
        # Fifth plot (lower part of the third column)
        # ax5 = self.fig.add_subplot(gs[1, 2])
        ax5 = self.fig.add_subplot(gs[3, 1])
        ax5.set_title("ZMP errors")
        ax5.set_xlabel('Time')
        self.zmp_err_ax = ax5

        # Adjust the layout
        self.fig.tight_layout()

        # Set a time reference
        self.time = np.linspace(0, num_frames, num_frames)

        # Internally store the CoM and ZMP trajectories and references
        self.zmp_trajectory = zmp_trajectory
        self.com_reference = com_reference
        self.zmp_reference = zmp_reference
        # Initialize the lines to represent the CoM and ZMP trajectories, references and errors
        self.com_traj_line, = self.com_traj_ref_ax.plot([], [], '-', lw=1, label="CoM Trajectory")
        self.com_ref_line, = self.com_traj_ref_ax.plot([], [], '--', lw=1, label="CoM Reference")
        self.com_err_line, = self.com_err_ax.plot([], [], '-', lw=1, label="CoM Error")
        self.zmp_traj_line, = self.zmp_traj_ref_ax.plot([], [], '-', lw=1, label="ZMP Trajectory")
        self.zmp_ref_line, = self.zmp_traj_ref_ax.plot([], [], '--', lw=1, label="ZMP Reference")
        self.zmp_err_line, = self.zmp_err_ax.plot([], [], '-', lw=1, label="ZMP Error")

        # Put the start and goal configurations in a numpy vector (and put the orientation in the interval [0, 2pi] rad)
        self.start = np.array([start_conf[0], start_conf[1], np.deg2rad(np.rad2deg(start_conf[2]) % 360)])
        self.goal = np.array([goal_conf[0], goal_conf[1], np.deg2rad(np.rad2deg(goal_conf[2]) % 360)])

        # Get the initial left foot position
        left_foot = np.array([
            self.start[0] - FOOT_DISTANCE * np.sin(self.start[2]),
            self.start[1] + FOOT_DISTANCE * np.cos(self.start[2])
        ])
        # Get the initial right foot position
        right_foot = np.array([
            self.start[0] + FOOT_DISTANCE * np.sin(self.start[2]),
            self.start[1] - FOOT_DISTANCE * np.cos(self.start[2])
        ])

        # Initialize the lists containing the evolution of the CoM and feet position
        self.com_pose_history = np.empty((num_frames, 3))
        self.com_pose_history[0, :] = self.start[:, 0]
        self.left_foot_history = [left_foot]
        self.right_foot_history = [right_foot]

        # Compute the following CoM and feet position
        self.following_com_position = following_com_position
        self.following_left_foot = following_left_foot
        self.following_right_foot = following_right_foot
        # The number of the previous frame
        self.last_seen_frame = -1

    def _draw_circle(self, position, alpha=1.0, radius=2.0, color='tomato', fill=True, linewidth=2):
        robot = plt.Circle((position[0], position[1]), radius=radius, color=color,
                           fill=fill, linewidth=linewidth, alpha=alpha)
        self.anim_ax.add_patch(robot)

    def _draw_tick(self, position, alpha=1.0, color='black', linewidth=2):
        tick_x = [position[0], position[0] + HumanoidAnimationHelper.TICK_LENGTH * np.cos(position[2])]
        tick_y = [position[1], position[1] + HumanoidAnimationHelper.TICK_LENGTH * np.sin(position[2])]
        self.anim_ax.plot(tick_x, tick_y, color=color, linewidth=linewidth, alpha=alpha)

    def update(self, frame: int):
        print("### FRAME", frame)

        # explanation of such control:
        # https://stackoverflow.com/questions/74252467/why-when-doing-animations-with-matplotlib-frame-0-appears-several-times
        if self.last_seen_frame != frame:
            self.last_seen_frame = frame
        else:
            return -1

        # Clear the canvas
        self.anim_ax.cla()

        # Plot the goal position
        self.anim_ax.scatter(self.goal[0], self.goal[1], marker="o", color="royalblue", label='goal', s=300)

        # Draw the last CoMs
        if frame != 0:
            for idx, com in enumerate(self.com_pose_history[frame:max(0, frame - NUMBER_OF_SHADOWS):-1]):
                # Draw the point representing the CoM
                self._draw_circle(com, alpha=1.0 - idx / NUMBER_OF_SHADOWS)
                # Draw the line connecting this CoM to the previous one
                self._draw_tick(com, alpha=1.0 - idx / NUMBER_OF_SHADOWS)

        # Draw the last left footsteps
        for idx, left in enumerate(
                reversed(self.left_foot_history[-min(len(self.left_foot_history), NUMBER_OF_FOOTSTEPS_SHADOWS):])):
            # Plot the point corresponding to the left foot position (and give a label - to be used by the legend -
            # only to the first footstep).
            self.anim_ax.scatter(left[0], left[1], marker="o", color="green", label='left foot' if idx == 0 else None,
                                 s=20, alpha=1.0 - idx * 1 / NUMBER_OF_SHADOWS / 2)

        # Draw the last right footsteps
        for idx, right in enumerate(
                reversed(self.right_foot_history[-min(len(self.right_foot_history), NUMBER_OF_FOOTSTEPS_SHADOWS):])):
            # Plot the point corresponding to the right foot position (and give a label - to be used by the legend -
            # only to the first footstep).
            self.anim_ax.scatter(right[0], right[1], marker="o", color="lightgreen",
                                 label='right foot' if idx == 0 else None,
                                 s=20, alpha=1.0 - idx * 1 / NUMBER_OF_SHADOWS / 2)

        # Add a legend
        self.anim_ax.legend(loc='upper left', ncol=1, fancybox=True, shadow=False,  # bbox_to_anchor=(1.45, 0.5),
                            fontsize="13")

        return self.anim_ax

    def show_animation_with_autogeneration(self, path_to_gif: str, num_frames: int = 20, interval: int = 200):
        """
        Shows the animation regarding the humanoid and saves it at save_path. All the configurations
         following the start and goal configurations will be computed automatically after each frame.

        :param path_to_gif: The path to the GIF file where the animation will be saved.
        :param num_frames: The number of frames in the animation.
        :param interval: Delay between frames in milliseconds.
        """
        super().show_animation(path_to_gif, num_frames, interval, self.update_with_autogeneration)

    def show_animation_with_offline_trajectory(self, path_to_gif: str, num_frames: int = 20, interval: int = 200):
        """
        Shows the animation regarding the humanoid and saves it at save_path. The configurations
        following the start and goal configurations are the ones stored in self.following_com_position,
        self.following_left_foot and self.following_right_foot.

        :param path_to_gif: The path to the GIF file where the animation will be saved.
        :param num_frames: The number of frames in the animation.
        :param interval: Delay between frames in milliseconds.
        """
        super().show_animation(path_to_gif, num_frames, interval, self.update_with_offline_trajectory)

    def _update_plots(self, frame: int):
        """
        Performs the update regarding the CoM and ZMP reference, trajectory and error.

        :param frame: The number of the frame whom state has to be represented.
        """
        # Update the trajectory, reference and error graphs of the CoM (x-component)
        self.com_traj_line.set_data(self.time[:frame + 1], self.com_pose_history[:frame + 1, 0])
        self.com_ref_line.set_data(self.time[:frame + 1], self.com_reference[:frame + 1, 0])
        self.com_err_line.set_data(self.time[:frame + 1],
                                   self.com_reference[:frame + 1, 0] - self.com_pose_history[:frame + 1, 0])
        # Update the trajectory, reference and error graphs of the CoM (y-component)
        self.com_traj_line.set_data(self.time[:frame + 1], self.com_pose_history[:frame + 1, 1])
        self.com_ref_line.set_data(self.time[:frame + 1], self.com_reference[:frame + 1, 1])
        self.com_err_line.set_data(self.time[:frame + 1],
                                   self.com_reference[:frame + 1, 1] - self.com_pose_history[:frame + 1, 1])
        # Update the trajectory, reference and error graphs of the ZMP (x-component)
        self.zmp_traj_line.set_data(self.time[:frame + 1], self.zmp_trajectory[:frame + 1, 0])
        self.zmp_ref_line.set_data(self.time[:frame + 1], self.zmp_reference[:frame + 1, 0])
        self.zmp_err_line.set_data(self.time[:frame + 1],
                                   self.zmp_reference[:frame + 1, 0] - self.zmp_trajectory[:frame + 1, 0])
        # Update the trajectory, reference and error graphs of the ZMP (y-component)
        self.zmp_traj_line.set_data(self.time[:frame + 1], self.zmp_trajectory[:frame + 1, 1])
        self.zmp_ref_line.set_data(self.time[:frame + 1], self.zmp_reference[:frame + 1, 1])
        self.zmp_err_line.set_data(self.time[:frame + 1],
                                   self.zmp_reference[:frame + 1, 1] - self.zmp_trajectory[:frame + 1, 1])

        # Adjust the plots limits
        self.anim_ax.set_xlim(HumanoidAnimationHelper.MIN_COORD, HumanoidAnimationHelper.MAX_COORD)
        self.anim_ax.set_ylim(HumanoidAnimationHelper.MIN_COORD, HumanoidAnimationHelper.MAX_COORD)
        self.com_traj_ref_ax.set_xlim(0, max(frame, 1))
        self.com_traj_ref_ax.set_ylim(-1.5, +1.5)
        self.com_err_ax.set_xlim(0, max(frame, 1))
        self.com_err_ax.set_ylim(-1.5, +1.5)
        self.zmp_traj_ref_ax.set_xlim(0, max(frame, 1))
        self.zmp_traj_ref_ax.set_ylim(-1.5, +1.5)
        self.zmp_err_ax.set_xlim(0, max(frame, 1))
        self.zmp_err_ax.set_ylim(-1.5, +1.5)

        if frame == 0:
            # Display the legends for the plots
            self.zmp_traj_ref_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            self.zmp_err_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            self.com_traj_ref_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            self.com_err_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if frame == 1:
            self.fig.tight_layout()

    def show_animation(self, path_to_gif: str, num_frames: int = 20, interval: int = 200, update: Callable = None):
        self.show_animation_with_autogeneration(path_to_gif, num_frames, interval)

    def update_with_autogeneration(self, frame):
        """
        Updates the canvas to display the state corresponding to the given frame number. All the configurations
         following the start and goal configurations will be computed automatically after each frame.

        :param frame: The number of the frame whom state has to be represented.
        """
        res = self.update(frame)
        if res == -1:
            return

        # Update the ZMP/CoM plots
        self._update_plots(frame)

        # ===== COMPUTING NEXT VALUES =====
        # Set a lower step size for the first step
        step_size = STEP_SIZE / 2 if frame == 0 else STEP_SIZE
        # Compute the next CoM position
        last_conf = self.com_pose_history[frame]
        next_conf = np.array([
            last_conf[0] + step_size / 2 * np.cos(last_conf[2]),
            last_conf[1] + step_size / 2 * np.sin(last_conf[2]),
            last_conf[2]
        ])
        self.com_pose_history[frame + 1] = next_conf

        foot = 0 if frame % 2 == 0 else 1

        if foot == 0:
            print("LEFT")
            last_left_foot = self.left_foot_history[-1]
            left_foot = np.array([
                last_left_foot[0] + step_size * np.cos(next_conf[2]),
                last_left_foot[1] + step_size * np.sin(next_conf[2])
            ])
            self.left_foot_history.append(left_foot)
        else:
            print("RIGHT")
            last_right_foot = self.right_foot_history[-1]
            right_foot = np.array([
                last_right_foot[0] + step_size * np.cos(next_conf[2]),
                last_right_foot[1] + step_size * np.sin(next_conf[2])
            ])
            self.right_foot_history.append(right_foot)

        self.anim_ax.set_xlim(HumanoidAnimationHelper.MIN_COORD, HumanoidAnimationHelper.MAX_COORD)
        self.anim_ax.set_ylim(HumanoidAnimationHelper.MIN_COORD, HumanoidAnimationHelper.MAX_COORD)
        self.anim_ax.set_aspect('equal', adjustable='box')

    def update_with_offline_trajectory(self, frame):
        """
        Updates the canvas to display the state corresponding to the given frame number. The configurations
        following the start and goal configurations are the ones stored in self.following_left_foot,
        self.following_left_foot and self.following_right_foot.

        :param frame: The number of the frame whom state has to be represented.
        """
        res = self.update(frame)
        if res == -1:
            return

        # Update the ZMP/CoM plots
        self._update_plots(frame)

        # ===== COMPUTING NEXT VALUES =====
        self.com_pose_history[frame + 1] = self.following_com_position.pop(0)[:, 0]
        foot = 0 if frame % 2 == 0 else 1
        if foot == 0:
            print("LEFT")
            self.left_foot_history.append(self.following_left_foot.pop(0))
        else:
            print("RIGHT")
            self.right_foot_history.append(self.following_right_foot.pop(0))

        # Additional adjustments
        self.anim_ax.set_aspect('equal', adjustable='box')


def generate_com_and_feet_evolution(start_com, start_left_foot, start_right_foot, num_frames):
    """
    Generates all the CoM and feet positions of the animation (starting from the configuration after the start
     configuration).
    """
    com_pose_history = []
    left_foot_history = []
    right_foot_history = []

    last_conf = start_com
    last_left_foot = start_left_foot
    last_right_foot = start_right_foot
    for i in range(num_frames):
        # Set a lower step size for the first step
        step_size = STEP_SIZE / 2 if i == 0 else STEP_SIZE

        # Generate the new CoM configuration
        next_conf = np.array([
            last_conf[0] + step_size / 2 * np.cos(last_conf[2]),
            last_conf[1] + step_size / 2 * np.sin(last_conf[2]),
            last_conf[2]
        ])
        com_pose_history.append(next_conf)
        last_conf = next_conf

        foot = 0 if i % 2 == 0 else 1
        if foot == 0:
            # Generate the new left feet position
            new_left_foot = np.array([
                last_left_foot[0] + step_size * np.cos(next_conf[2]),
                last_left_foot[1] + step_size * np.sin(next_conf[2])
            ])
            left_foot_history.append(new_left_foot)
            last_left_foot = new_left_foot
        else:
            # Generate the new right feet position
            new_right_foot = np.array([
                last_right_foot[0] + step_size * np.cos(next_conf[2]),
                last_right_foot[1] + step_size * np.sin(next_conf[2])
            ])
            right_foot_history.append(new_right_foot)
            last_right_foot = new_right_foot

    return com_pose_history, left_foot_history, right_foot_history


if __name__ == "__main__":
    # TESTING HumanoidAnimationHelper WITH DUMMY DATA
    num_frames = 21

    # Set a random goal
    goal = np.random.randint(-10, 10, (3, 1))
    # Set a random initial position
    start = np.random.randint(-10, 10, (3, 1))

    # Set the CoM and ZMP trajectory and reference
    time = np.linspace(0, num_frames, num_frames)
    zmp_trajectory = np.array([np.sin(time + np.pi), np.cos(time + np.pi)]).T  # dim: (num_frames, 2)
    zmp_reference = np.array([np.sin(time), np.cos(time)]).T  # dim: (num_frames, 2)
    com_reference = np.array([np.cos(time + np.pi), np.sin(time + np.pi)]).T  # dim: (num_frames, 2)
    anim_helper = HumanoidAnimationHelper(goal_conf=goal, start_conf=start,
                                          zmp_reference=zmp_reference, zmp_trajectory=zmp_trajectory,
                                          com_reference=com_reference)
    # anim_helper.show_animation_with_autogeneration('./Assets/Animations/humanoid_2d_animation.gif')

    anim_helper.following_com_position, anim_helper.following_left_foot, anim_helper.following_right_foot = (
        generate_com_and_feet_evolution(
            anim_helper.start, anim_helper.left_foot_history[0],
            anim_helper.right_foot_history[0], num_frames=num_frames))
    anim_helper.show_animation_with_offline_trajectory('./Assets/Animations/humanoid_2d_animation.gif')
