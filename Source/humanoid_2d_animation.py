import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation

# ===== CONSTANTS =====
STEP_SIZE = 6
# The maximum number of previous CoM position to plot
NUMBER_OF_SHADOWS = 10
# The maximum number of previous footsteps to plot
NUMBER_OF_FOOTSTEPS_SHADOWS = NUMBER_OF_SHADOWS // 2
FOOT_DISTANCE = 3
NUMBER_OF_FOOTSTEPS = 20


# ===== UTILITY FUNCTIONS =====
class HumanoidAnimationHelper:
    # Assuming that the plot is a square, the minimum and maximum value that x and y coordinated can have
    MIN_COORD = -100
    MAX_COORD = +100
    # The length of the tick in the line representing the motion of the CoM
    TICK_LENGTH = 1

    def __init__(self, start_conf, goal_conf, fig=None, ax=None):
        """
        Initializes a new instance of the utility to show the animation regarding the humanoid.

        :param start_conf: The start configuration of the humanoid, represented by a 3-components vector:
         X-coord, Y-coord, orientation.
        :param goal_conf: The goal configuration, represented by a 3-components vector: X-coord, Y-coord, orientation.
        :param fig: The first element returned by plt.subplots(). If either fig or axis is not provided, a new one
        is created.
        :param ax: The second element returned by plt.subplots(). If either fig or axis is not provided, a new one
        is created.
        """
        if fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots()

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
        self.com_pose_history = [self.start]
        self.left_foot_history = [left_foot]
        self.right_foot_history = [right_foot]

        # The number of the previous frame
        self.last_seen_frame = -1

    def show_animation(self, path_to_gif: str, num_frames: int = 20, interval: int = 200):
        """
        Shows the animation regarding the humanoid and saves it at save_path.
        :param path_to_gif: The path to the GIF file where the animation will be saved.
        :param num_frames: The number of frames in the animation.
        :param interval: Delay between frames in milliseconds.
        """
        ani = FuncAnimation(self.fig, self._update_with_autogeneration, frames=num_frames, interval=interval)
        ani.save(path_to_gif, writer='ffmpeg')
        plt.show()

    def _draw_circle(self, position, alpha=1.0, radius=2.0, color='tomato', fill=True, linewidth=2):
        robot = plt.Circle((position[0], position[1]), radius=radius, color=color,
                           fill=fill, linewidth=linewidth, alpha=alpha)
        self.ax.add_patch(robot)

    def _draw_tick(self, position, alpha=1.0, color='black', linewidth=2):
        tick_x = [position[0], position[0] + HumanoidAnimationHelper.TICK_LENGTH * np.cos(position[2])]
        tick_y = [position[1], position[1] + HumanoidAnimationHelper.TICK_LENGTH * np.sin(position[2])]
        plt.plot(tick_x, tick_y, color=color, linewidth=linewidth, alpha=alpha)

    def _update_with_autogeneration(self, frame):
        """
        Updates the canvas to display the state corresponding to the given frame number. Only the start and goal
         configuration can be provided: the following configurations will be automatically computed.

        :param frame: The number of the frame whom state has to be represented.
        """
        print("### FRAME", frame)

        # explanation of such control:
        # https://stackoverflow.com/questions/74252467/why-when-doing-animations-with-matplotlib-frame-0-appears-several-times
        if self.last_seen_frame != frame:
            self.last_seen_frame = frame
        else:
            return

        # Clear the current axis
        plt.cla()

        # Set a lower step size for the first step
        step_size = STEP_SIZE / 2 if frame == 0 else STEP_SIZE

        # Plot the goal position
        plt.scatter(self.goal[0], self.goal[1], marker="o", color="royalblue", label='goal', s=300)

        # Draw the last CoMs
        for idx, com in enumerate(
                reversed(self.com_pose_history[-min(len(self.com_pose_history), NUMBER_OF_SHADOWS):])):
            # Draw the point representing the CoM
            self._draw_circle(com, alpha=1.0 - idx / NUMBER_OF_SHADOWS)
            # Draw the line connecting this CoM to the previous one
            self._draw_tick(com, alpha=1.0 - idx / NUMBER_OF_SHADOWS)

        # Draw the last left footsteps
        for idx, left in enumerate(
                reversed(self.left_foot_history[-min(len(self.left_foot_history), NUMBER_OF_FOOTSTEPS_SHADOWS):])):
            # Plot the point corresponding to the left foot position (and give a label - to be used by the legend -
            # only to the first footstep).
            plt.scatter(left[0], left[1], marker="o", color="green", label='left foot' if idx == 0 else None,
                        s=20, alpha=1.0 - idx * 1 / NUMBER_OF_SHADOWS / 2)

        # Draw the last right footsteps
        for idx, right in enumerate(
                reversed(self.right_foot_history[-min(len(self.right_foot_history), NUMBER_OF_FOOTSTEPS_SHADOWS):])):
            # Plot the point corresponding to the right foot position (and give a label - to be used by the legend -
            # only to the first footstep).
            plt.scatter(right[0], right[1], marker="o", color="lightgreen", label='right foot' if idx == 0 else None,
                        s=20, alpha=1.0 - idx * 1 / NUMBER_OF_SHADOWS / 2)

        # ===== COMPUTING NEXT VALUES =====
        last_conf = self.com_pose_history[-1]
        next_conf = np.array([
            last_conf[0] + step_size / 2 * np.cos(last_conf[2]),
            last_conf[1] + step_size / 2 * np.sin(last_conf[2]),
            last_conf[2]
        ])
        self.com_pose_history.append(next_conf)

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

        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.subplots_adjust(right=0.75)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=1, fancybox=True, shadow=False, fontsize="13")


if __name__ == "__main__":
    # Set a random goal
    goal = np.random.randint(-10, 10, (3, 1))
    # Set a random initial position
    start = np.random.randint(-10, 10, (3, 1))

    anim_helper = HumanoidAnimationHelper(goal_conf=goal, start_conf=start)
    anim_helper.show_animation('../Assets/Animations/humanoid_2d_animation.gif')
