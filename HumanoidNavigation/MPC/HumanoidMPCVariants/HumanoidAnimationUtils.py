import matplotlib
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull


class HumanoidAnimationUtils:
    """
    The class that handles the animation of the humanoid's motion.
    """
    # The width and height of the triangle representing the pose of the CoM in the animation
    TRIANGLE_HEIGHT = 0.4
    TRIANGLE_WIDTH = 0.35

    # The height and the width of the rectangle that represents the stance foot in the animation plot
    FOOT_RECTANGLE_WIDTH = 0.3
    FOOT_RECTANGLE_HEIGHT = 0.15

    # The constants used to represent the left and right feet
    RIGHT_FOOT = 1
    LEFT_FOOT = -1

    class _HumanoidAnimationFrame:
        """
        A class whose instance describes a single frame of the humanoid's animation.
        """

        def __init__(self, com_position: np.ndarray, humanoid_orientation: float, footstep_position: np.ndarray,
                     which_footstep: int):
            # The global position of the CoM in the map
            self.com_position = com_position
            # The global orientation of the humanoid in the map
            self.humanoid_orientation = humanoid_orientation
            # The global position of the stance foot in the map
            self.footstep_position = footstep_position
            # 1 if the stance foot is the right one, -1 otherwise
            self.which_footstep = which_footstep
            # TODO: add c, eta and half plane for each obstacle

    def __init__(self, goal_position: np.ndarray, obstacles: list[ConvexHull] = []):
        """
        Initializes the animation environment, and adds to the animation the elements that will be displayed in
         each frame.

        :param goal_position: The global position that the humanoid has to reach in the map.
        :param obstacles: The list of all the obstacles in the map, represented as polygons.
        """
        self._frames_data: list[HumanoidAnimationUtils._HumanoidAnimationFrame] = []
        self.obstacles: list[ConvexHull] = obstacles
        self.goal_position: np.ndarray = goal_position

    def add_frame_data(self, com_position: np.ndarray, humanoid_orientation: float, footstep_position: np.ndarray,
                       which_footstep: int) -> None:
        """
        Adds to the sequence of frame all the data referred to the current frame of the animation.

        :param com_position: The global position of the CoM in the map.
        :param humanoid_orientation: The global orientation of the humanoid in the map.
        :param footstep_position: The global position of the stance foot in the map.
        :param which_footstep: 1 if the stance foot is the right one, -1 otherwise.
        """
        self._frames_data.append(HumanoidAnimationUtils._HumanoidAnimationFrame(
            com_position, humanoid_orientation, footstep_position, which_footstep
        ))

    @staticmethod
    def _plot_polygon(ax: plt.Axes, polygon: ConvexHull, color='blue', label=None):
        polygon = polygon.points[polygon.vertices]
        # 'close' the polygon by appending the first vertex to the end of the vertex list
        polygon = np.append(polygon, [polygon[0]], axis=0)
        ax.plot(polygon[:, 0], polygon[:, 1], '-', color=color, label=label)
        ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.2, color=color)

    def plot_animation(self, path_to_gif: str = None):
        """
        Shows the animation of the humanoid's motion.

        :param path_to_gif: The path to the GIF image where the animation will be saved.
        """
        # Extract the x, y and theta trajectories of the triangle representing the CoM
        x_trajectory = np.zeros(len(self._frames_data))
        y_trajectory = np.zeros(len(self._frames_data))
        theta_trajectory = np.zeros(len(self._frames_data))
        for ind, frame in enumerate(self._frames_data):
            x_trajectory[ind] = frame.com_position[0]
            y_trajectory[ind] = frame.com_position[1]
            theta_trajectory[ind] = frame.humanoid_orientation

        # Extract the position of all the footsteps and which is the stance
        footsteps = np.zeros((3, len(self._frames_data)))
        for ind, frame in enumerate(self._frames_data):
            footsteps[0, ind] = self._frames_data[ind].footstep_position[0]
            footsteps[1, ind] = self._frames_data[ind].footstep_position[1]
            footsteps[2, ind] = self._frames_data[ind].which_footstep

        # Define the vertices representing a triangle with the base laying on the X-axis and the other vertex on
        # the positive Y-axis.
        vert = np.array([[self.TRIANGLE_HEIGHT, 0], [0, self.TRIANGLE_WIDTH / 2], [0, -self.TRIANGLE_WIDTH / 2]]).T
        # Compute the barycenter of the triangle and put it in the origin
        barycenter = np.mean(vert, axis=1)[..., np.newaxis]
        vert = vert - barycenter
        barycenter = np.zeros((2, 1))
        # Define the rotation matrix that gives the robot the appropriate orientation
        rotation_matrix = np.array([
            [np.cos(theta_trajectory), -np.sin(theta_trajectory)],
            [np.sin(theta_trajectory), np.cos(theta_trajectory)]
        ]).squeeze().transpose(2, 0, 1)
        # Define the expression that puts the robot in the appropriate position and orientation
        triangle_poses = rotation_matrix @ vert + np.array([[x_trajectory, y_trajectory]]).T
        # Define the expression that puts the robot's barycenter in the appropriate position
        barycenter_traj = barycenter + np.array([[x_trajectory, y_trajectory]]).T

        # Set up the plot
        fig, ax = plt.subplots()
        min_x, max_x = min(min(x_trajectory), min(footsteps[0, :])), max(max(x_trajectory), max(footsteps[0, :]))
        min_y, max_y = min(min(y_trajectory), min(footsteps[1, :])), max(max(y_trajectory), max(footsteps[1, :]))
        ax.set_xlim(min_x - 2, max_x + 2)  # Set x-axis limits
        ax.set_ylim(min_y - 2, max_y + 2)  # Set y-axis limits
        ax.set_aspect('equal')  # Set equal aspect ratio for accurate proportions

        # Compute the rectangle representing each footstep
        footsteps_rectangles: list[Rectangle] = []
        for ind, footstep in enumerate(footsteps.T):
            # Build the rectangle associated to this footstep
            x, y = footstep[0], footstep[1]
            # Create rectangle centered at the position with the humanoid's orientation
            rect = Rectangle((-self.FOOT_RECTANGLE_WIDTH / 2, -self.FOOT_RECTANGLE_HEIGHT / 2),
                             self.FOOT_RECTANGLE_WIDTH, self.FOOT_RECTANGLE_HEIGHT,
                             color='blue' if footstep[2] == self.RIGHT_FOOT else 'green', alpha=0.7)
            t = (matplotlib.transforms.Affine2D().rotate(theta_trajectory[ind]) +
                 matplotlib.transforms.Affine2D().translate(x, y) + ax.transData)
            rect.set_transform(t)
            # Store this rectangle in a list
            footsteps_rectangles.append(rect)
            # Add this rectangle to the plot without showing it
            rect.set_visible(False)
            ax.add_patch(rect)

        # Initialize the triangle
        triangle_patch = patches.Polygon(triangle_poses[0].T, closed=True, facecolor='green')
        ax.add_patch(triangle_patch)

        # Initialize the plots of the barycenter and its trajectory
        barycenter_point, = ax.plot([], [], 'ro', label="Barycenter")
        # trajectory_line, = ax.plot([], [], 'r-', lw=1, label="Trajectory")
        trajectory_line, = ax.plot([], [], '--k', lw=1, label="Trajectory")

        # Show all the obstacles
        for obs in self.obstacles:
            self._plot_polygon(ax, obs)

        # TODO: show the goal and c and eta for each obstacle

        def update(frame):
            """Update the triangle's vertices, barycenter and trajectory at each frame."""
            # Update the CoM triangle position
            triangle_patch.set_xy(triangle_poses[frame].T)

            # Update the barycenter position
            barycenter_curr_pos = barycenter_traj[frame]
            barycenter_point.set_data(barycenter_curr_pos[0], barycenter_curr_pos[1])
            # Update the barycenter trajectory
            trajectory_line.set_data(barycenter_traj[:frame + 1, 0], barycenter_traj[:frame + 1, 1])

            # Update the footsteps opacity
            for i in range(frame):
                footsteps_rectangles[i].set_alpha(footsteps_rectangles[i].get_alpha() * .85)
            # Display the rectangles
            footsteps_rectangles[frame].set_visible(True)

            return triangle_patch, barycenter_point, trajectory_line, footsteps_rectangles[:frame]

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(triangle_poses), )  # 1 frame per second
        if path_to_gif is not None:
            ani.save(path_to_gif, writer='ffmpeg')
        # Display the animation
        plt.show()


if __name__ == '__main__':
    animator = HumanoidAnimationUtils(
        goal_position=np.array([10, 10]),
        obstacles=[ConvexHull(np.array([[-0.5, 2], [-0.5, 4], [2, 2], [2, 4]]))],
    )

    for i in range(10):
        animator.add_frame_data(
            com_position=np.array([i, i]),
            humanoid_orientation=(np.pi / 2) * (i / 9),
            footstep_position=np.array([-i, i]),
            which_footstep=i % 2,
        )

    animator.plot_animation(path_to_gif='/Users/salvatore/Downloads/res.gif')
