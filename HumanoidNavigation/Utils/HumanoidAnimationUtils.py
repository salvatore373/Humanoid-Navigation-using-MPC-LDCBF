import os
import tempfile

import matplotlib
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from yaml import safe_load

this_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.dirname(this_dir)
with open(config_dir + '/config.yml', 'r') as file:
    conf = safe_load(file)


class HumanoidAnimationUtils:
    """
    The class that handles the animation of the humanoid's motion.
    """

    class _HumanoidAnimationFrame:
        """
        A class whose instance describes a single frame of the humanoid's animation.
        """

        def __init__(self, com_position: np.ndarray, humanoid_orientation: float, footstep_position: np.ndarray,
                     which_footstep: int, list_point_c: list[np.ndarray], inferred_obstacles=[], lidar_readings=[]):
            # The global position of the CoM in the map
            self.com_position = com_position
            # The global orientation of the humanoid in the map
            self.humanoid_orientation = humanoid_orientation
            # The global position of the stance foot in the map
            self.footstep_position = footstep_position
            # 1 if the stance foot is the right one, -1 otherwise
            self.which_footstep = which_footstep
            # The list of the points c on the edge of the obstacles in the map.
            self.list_point_c = list_point_c
            # The list of the inferred obstacles.
            self.inferred_obstacles = inferred_obstacles
            # The list of the LiDAR readings.
            self.lidar_readings = lidar_readings

    def __init__(self, goal_position: np.ndarray, obstacles: list[ConvexHull] = [], delta: float = 0):
        """
        Initializes the animation environment, and adds to the animation the elements that will be displayed in
         each frame.

        :param goal_position: The global position that the humanoid has to reach in the map.
        :param obstacles: The list of all the obstacles in the map, represented as polygons.
        :param delta: The minimum value that the LDCBF can take.
        """
        self._frames_data: list[HumanoidAnimationUtils._HumanoidAnimationFrame] = []
        self.obstacles: list[ConvexHull] = obstacles
        self.goal_position: np.ndarray = np.array(goal_position).reshape((-1, 2))
        self.delta = delta

    def add_goal(self, new_goal):
        """
        Adds new_goal to the list of goals to plot
        """
        self.goal_position = np.vstack((self.goal_position, new_goal))

    def add_frame_data(self, com_position: np.ndarray, humanoid_orientation: float, footstep_position: np.ndarray,
                       which_footstep: int, list_point_c: list[np.ndarray], inferred_obstacles=[],
                       lidar_readings=[]) -> None:
        """
        Adds to the sequence of frame all the data referred to the current frame of the animation.

        :param com_position: The global position of the CoM in the map.
        :param humanoid_orientation: The global orientation of the humanoid in the map.
        :param footstep_position: The global position of the stance foot in the map.
        :param which_footstep: 1 if the stance foot is the right one, -1 otherwise.
        :param list_point_c: The list of the points c on the edge of the obstacles in the map.
        """
        self._frames_data.append(HumanoidAnimationUtils._HumanoidAnimationFrame(
            com_position, humanoid_orientation, footstep_position, which_footstep,
            list_point_c, inferred_obstacles, lidar_readings
        ))

    @staticmethod
    def _plot_polygon(ax: plt.Axes, polygon: ConvexHull, color='blue', label=None):
        if isinstance(polygon, ConvexHull):
            polygon = polygon.points[polygon.vertices]
        else:
            polygon = np.array(polygon)
        # 'close' the polygon by appending the first vertex to the end of the vertex list
        polygon = np.append(polygon, [polygon[0]], axis=0)
        ax.plot(polygon[:, 0], polygon[:, 1], '-', color=color, label=label)
        ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.2, color=color)

    def plot_animation(self, path_to_gif: str = None, path_to_frames_folder: str = None, num_sampled_frames: int = 10):
        """
        Shows the animation of the humanoid's motion.

        :param path_to_gif: The path to the GIF image where the animation will be saved.
        :param path_to_frames_folder: The path to the folder where some frames sampled from the animation will be saved
        as PDF images. If this is provided, the legend will not be shown.
        :param num_sampled_frames: The number of frames to sample from the animation and put in path_to_frames_folder.
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
        vert = np.array(
            [[conf["TRIANGLE_HEIGHT"], 0], [0, conf["TRIANGLE_WIDTH"] / 2], [0, -conf["TRIANGLE_WIDTH"] / 2]]).T
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
        fig, ax = plt.subplots(dpi=100)
        # Compute the min and max coordinates of the obstacles
        min_obs, max_obs = float('inf'), float('-inf')
        for o in self.obstacles:
            vertices = o.points[o.vertices]
            min_obs, max_obs = min(min_obs, vertices.min()), max(max_obs, vertices.max())
        # Compute the min and max coordinates of the trajectory and goal
        min_rob_goal = min(min(x_trajectory), min(y_trajectory), footsteps.min(), self.goal_position.min())
        max_rob_goal = max(max(x_trajectory), max(y_trajectory), footsteps.max(), self.goal_position.max())
        # Compute the overall min and max coordinates
        min_coord, max_coord = min(min_rob_goal, min_obs), max(max_rob_goal, max_obs)
        ax.set_xlim(min_coord - 2, max_coord + 2)  # Set x-axis limits
        ax.set_ylim(min_coord - 2, max_coord + 2)  # Set y-axis limits
        ax.set_aspect('equal')  # Set equal aspect ratio for accurate proportions

        # Compute the rectangle representing each footstep
        footsteps_rectangles: list[Rectangle] = []
        for ind, footstep in enumerate(footsteps.T):
            # Build the rectangle associated to this footstep
            x, y = footstep[0], footstep[1]
            # In the last frame, the input is not computed and no footstep is available
            if x is None and y is None:
                continue
            # Create rectangle centered at the position with the humanoid's orientation
            rect = Rectangle((-conf["FOOT_RECTANGLE_WIDTH"] / 2, -conf["FOOT_RECTANGLE_HEIGHT"] / 2),
                             conf["FOOT_RECTANGLE_WIDTH"], conf["FOOT_RECTANGLE_HEIGHT"],
                             color='blue' if footstep[2] == conf["RIGHT_FOOT"] else 'green', alpha=0.7, zorder=3)
            t = (matplotlib.transforms.Affine2D().rotate(theta_trajectory[ind]) +
                 matplotlib.transforms.Affine2D().translate(x, y) + ax.transData)
            rect.set_transform(t)
            # Store this rectangle in a list
            footsteps_rectangles.append(rect)
            # Add this rectangle to the plot without showing it
            rect.set_visible(False)
            ax.add_patch(rect)

        # Initialize the triangle
        triangle_patch = patches.Polygon(triangle_poses[0].T, closed=True, facecolor='cornflowerblue', zorder=4)
        ax.add_patch(triangle_patch)

        # lidar_range = patches.Circle((float(barycenter_traj[0][0]), float(barycenter_traj[0][1])),
        #                    radius=3.0, color='tomato',
        #                    label='LiDAR range', fill=False, linewidth=1, alpha=1.0, zorder=6)
        # ax.add_patch(lidar_range)

        # Initialize the plots of the barycenter and its trajectory
        barycenter_point, = ax.plot([], [], 'o', label="CoM", color='cornflowerblue', zorder=5)
        trajectory_line, = ax.plot([], [], '--k', lw=1, label="CoM Trajectory", zorder=4)

        # Plot the goal point
        ax.scatter(self.goal_position[:, 0], self.goal_position[:, 1], color='darkorange', label='Goal Position',
                   zorder=3)

        # Show all the obstacles
        for obs in self.obstacles:
            self._plot_polygon(ax, obs, color='orange')

        point_c_per_frame = []
        for frame_num, frame_data in enumerate(self._frames_data):
            list_point_c_curr_frame = []
            for obs_num, c in enumerate(frame_data.list_point_c):
                list_point_c_curr_frame.append(c)
            point_c_per_frame.append(list_point_c_curr_frame)

        inferred_obstacle_per_frame = [frame_data.inferred_obstacles for frame_data in self._frames_data]

        # inferred_obstacles_outline = []
        # inferred_obstacles_fill = []
        # inferred_obstacle_per_frame = []
        # for frame_num, frame_data in enumerate(self._frames_data):
        #     inferred_obstacle_per_frame.append(frame_data.inferred_obstacles)
        #
        #     inferred_obstacle_outline = [ # , label='Inferred Obstacle' if frame_num == 0 else None
        #         ax.plot([], [], '-', color='blue')[0]
        #         for i in range(len(frame_data.inferred_obstacles))
        #     ]
        #     inferred_obstacles_outline.append(inferred_obstacle_outline)
        #
        #     inferred_obstacle_fill = [
        #         ax.fill([], [], alpha=0.2, color='blue')[0]
        #         for i in range(len(frame_data.inferred_obstacles))
        #     ]
        #     inferred_obstacles_fill.append(inferred_obstacle_fill)

        # inferred_obstacle_outline, = ax.plot([], [], '-', color='blue', label='Inferred Obstacle')
        # inferred_obstacle_fill, = ax.fill([], [], alpha=0.2, color='blue')

        lidar_readings_per_frame = []
        for frame_num, frame_data in enumerate(self._frames_data):
            lidar_readings_x = []
            lidar_readings_y = []
            for point in frame_data.lidar_readings:
                if point:  # ignore None points (i.e. no obstacles)
                    lidar_readings_x.append(point[0])
                    lidar_readings_y.append(point[1])
            lidar_readings_per_frame.append(list(zip(lidar_readings_x, lidar_readings_y)))

        lidar_readings = ax.scatter([], [], s=1.5, color='green', label="LiDAR readings", zorder=4)

        # Plot a circle around the robot, representing the LiDAR's range. Plot only if the LiDAR readings were provided
        # lidar_range = None
        display_lidar_range = False
        if any(r != [] for r in lidar_readings_per_frame):
            lidar_range = patches.Circle((float(barycenter_traj[0][0]), float(barycenter_traj[0][1])),
                                         radius=1.5, color='tomato',
                                         label='LiDAR range', fill=False, linewidth=1,
                                         alpha=1.0, zorder=6)
            ax.add_patch(lidar_range)
            display_lidar_range = True

        # For each obstacle, initialize a vector and a point to display at each frame at the appropriate position
        # points_c = ax.scatter(np.zeros(len(self.obstacles)), np.zeros(len(self.obstacles)),
        #                       color='red', label="Points c", zorder=3)
        points_c = ax.scatter([], [], color='red', label="Points c", zorder=3)
        segments_eta = [
            ax.plot([], [], 'r--', label="Vectors $\eta$" if i == 0 else None, zorder=3)[0]
            for i in range(len(self.obstacles))
        ]

        # For each obstacle, initialize the half plane representing the safe area of its CBF
        x_linspace = np.linspace(*ax.get_xlim(), 300)
        y_linspace = np.linspace(*ax.get_ylim(), 300)
        X_meshgrid, Y_meshgrid = np.meshgrid(x_linspace, y_linspace)
        half_planes = [None for _ in range(len(self.obstacles))]

        # Show the legend (at the most appropriate location)
        if path_to_frames_folder is None:  # In the grid frames there will be no legend
            plt.legend()

        sampled_frames_ind = []
        if path_to_frames_folder is not None:
            # Create the file names that will store the SVGs to put in the frames grid
            os.makedirs(path_to_frames_folder, exist_ok=True)
            pdf_frames = [f'{path_to_frames_folder}/frame_{i}.pdf' for i in range(num_sampled_frames)]
            # Compute which frames should be sampled
            sampled_frames_ind = np.linspace(0, len(triangle_poses) - 1, num=num_sampled_frames, dtype=int)

        def update(frame):
            # Update the CoM triangle position
            triangle_patch.set_xy(triangle_poses[frame].T)

            # Update the barycenter position
            barycenter_curr_pos = barycenter_traj[frame]
            barycenter_point.set_data(barycenter_curr_pos[0], barycenter_curr_pos[1])
            # Update the barycenter trajectory
            trajectory_line.set_data(barycenter_traj[:frame + 1, 0], barycenter_traj[:frame + 1, 1])

            # Update the LiDAR range circle center
            if display_lidar_range:
                lidar_range.set_center(barycenter_curr_pos.squeeze())

            # # inferred polygons for current frame
            # curr_inferred_obstacles = inferred_obstacle_per_frame[frame]
            # if len(curr_inferred_obstacles) > 0:
            #     glob_curr_inferred_obstacles = [
            #         np.array(rotation_matrix[frame] @ np.array(curr_inferred_obstacles[k].points).T + np.array(
            #             [[x_trajectory[frame], y_trajectory[frame]]]).T).T
            #         for k in range(len(curr_inferred_obstacles))
            #     ]
            #
            #     obs_curr_frame = inferred_obstacles_outline[frame]
            #     for obs_ind, o in enumerate(obs_curr_frame):
            #         o.set_data(glob_curr_inferred_obstacles[obs_ind][:, 0], glob_curr_inferred_obstacles[obs_ind][:, 1])
            #         inferred_obstacles_fill[frame][obs_ind].set_xy(glob_curr_inferred_obstacles[obs_ind])

            # Remove previously drawn inferred-obstacle artists (if any)
            if hasattr(update, "current_inferred_outlines"):
                for artist in update.current_inferred_outlines:
                    artist.remove()
            if hasattr(update, "current_inferred_fills"):
                for artist in update.current_inferred_fills:
                    artist.remove()
            # Initialize lists to store the current frame’s inferred obstacle artists
            update.current_inferred_outlines = []
            update.current_inferred_fills = []

            # Get inferred obstacles for the current frame
            curr_inferred_obstacles = inferred_obstacle_per_frame[frame]
            for obs in curr_inferred_obstacles:
                # Transform the obstacle’s points from its local frame to global coordinates.
                local_points = np.array(obs.points)  # shape: (N,2)
                global_points = (rotation_matrix[frame] @ local_points.T).T + np.array(
                    [x_trajectory[frame], y_trajectory[frame]])
                # Plot the outline
                outline, = ax.plot(global_points[:, 0], global_points[:, 1], '-', color='blue')
                # Plot the filled polygon (ax.fill returns a list; take its first element)
                fill = ax.fill(global_points[:, 0], global_points[:, 1], alpha=0.2, color='blue')[0]
                update.current_inferred_outlines.append(outline)
                update.current_inferred_fills.append(fill)

            curr_lidar_reading = np.array(lidar_readings_per_frame[frame]).T
            if len(curr_lidar_reading) > 0:
                corrected_lidar_reading = rotation_matrix[frame] @ curr_lidar_reading
                corrected_lidar_reading = corrected_lidar_reading + np.array(
                    [[x_trajectory[frame], y_trajectory[frame]]]).T
                # corrected_lidar_reading = rotation_matrix[frame].T @ curr_lidar_reading
                # corrected_lidar_reading = np.array(rotation_matrix[frame] @ curr_lidar_reading).T
                # corrected_lidar_reading = np.array(rotation_matrix[frame].T @ curr_lidar_reading).T
                # corrected_lidar_reading = rotation_matrix[frame] @ curr_lidar_reading + np.array([[x_trajectory[frame], y_trajectory[frame]]]).T
                # corrected_lidar_reading = lidar_readings_per_frame[frame]
                # corrected_lidar_reading = curr_lidar_reading + np.array([[x_trajectory[frame], y_trajectory[frame]]]).T
                lidar_readings.set_offsets(corrected_lidar_reading.T)

            # Update the position of points c on the obstacles' edges
            # points_c.set_offsets(point_c_per_frame[frame, :])
            c_per_frame = point_c_per_frame[frame]
            if len(c_per_frame) != 0:
                points_c.set_offsets(c_per_frame)

            # Update the position of vectors eta from the obstacles' edges
            for obs_ind, s in enumerate(segments_eta):
                if obs_ind >= len(c_per_frame):
                    s.set_data([], [])
                else:
                    c_x, c_y = c_per_frame[obs_ind]
                    com_x, com_y = barycenter_curr_pos.squeeze()
                    s.set_data([c_x, com_x], [c_y, com_y])

                    # Update the position of half-planes representing safe areas
                    hp = half_planes[obs_ind]
                    if hp is not None:
                        for coll in hp.collections:
                            coll.remove()
                    eta = np.array([com_x - c_x, com_y - c_y])
                    eta /= np.linalg.norm(eta)
                    eta_x, eta_y = eta
                    condition = eta_x * (X_meshgrid - c_x) + eta_y * (Y_meshgrid - c_y) - self.delta >= 0
                    half_planes[obs_ind] = ax.contourf(X_meshgrid, Y_meshgrid, condition, levels=[0.5, 1],
                                                       colors='gray', alpha=0.5)

            # Update the footsteps opacity
            for i in range(frame):
                footsteps_rectangles[i].set_alpha(footsteps_rectangles[i].get_alpha() * .95)
            # Display the rectangles
            footsteps_rectangles[frame].set_visible(True)

            sampling_ind = np.where(frame == sampled_frames_ind)[0]
            if path_to_frames_folder is not None and len(sampling_ind) > 0:
                # This frame has to be put in the frames grid, then save it
                fig.savefig(pdf_frames[sampling_ind[0]], format="pdf")

            return (triangle_patch, barycenter_point, trajectory_line, footsteps_rectangles[:frame],
                    points_c, segments_eta, half_planes, *update.current_inferred_outlines,
                    *update.current_inferred_fills, lidar_readings)

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(triangle_poses))  # 1 frame per second
        if path_to_gif is not None:
            os.makedirs(os.path.dirname(path_to_gif), exist_ok=True)
            ani.save(path_to_gif, writer='ffmpeg')
        else:
            # Call save() with a temporary file in order to generate all the frames
            with tempfile.NamedTemporaryFile(suffix='.gif', mode='w') as tmp_gif:
                ani.save(tmp_gif.name, writer='ffmpeg')

        # Don't save the grid frames again with plt.show()
        sampled_frames_ind = []

        # Display the animation or the frames grid
        plt.show()

    @staticmethod
    def plot_fast_static(state_glob, input_glob, goal_position, obstacles: list[ConvexHull], s_v: list):
        """
        It makes a static (though fast) plot of the trajectory of the CoM contained in state_glob and the
        footsteps prints contained in input_glob.

        :param state_glob: A matrix of shape (5xN_simul+1) that represents the state of the humanoid dynamic system at
        any instant of the simulation. The coordinates system is the one of the inertial RF.
        :param input_glob: A matrix of shape (3xN_simul) that represents the input of the humanoid dynamic system at
        any instant of the simulation. The coordinates system is the one of the inertial RF.
        :param goal_position: The position of the goal in global coordinates
        :param obstacles: The ConvexHulls representing the obstacles in the map, in global coordinates
        :param s_v: The evolution of the s_v parameter of the humanoid.
        """
        fix, ax = plt.subplots()

        # Plot the start position
        plt.plot(state_glob[0, 0], state_glob[2, 0], marker='o', color="cornflowerblue", label="Start")

        # Plot the goal position
        plt.plot(goal_position[0], goal_position[1], marker='o', color="darkorange", label="Goal")

        # Plot the obstacles
        for obstacle in obstacles:
            HumanoidAnimationUtils._plot_polygon(ax, obstacle, color='orange')

        # Plot the trajectory of the CoM computed by the MPC
        plt.plot(state_glob[0, :], state_glob[2, :], color="mediumpurple", label="Predicted Trajectory")

        # Plot the footsteps plan computed by the MPC
        for time_instant, (step_x, step_y, _) in enumerate(input_glob.T):
            foot_orient = state_glob[4, time_instant]

            # Create rectangle centered at the position
            rect = Rectangle((-conf["FOOT_RECTANGLE_WIDTH"] / 2,
                              -conf["FOOT_RECTANGLE_HEIGHT"] / 2),
                             conf["FOOT_RECTANGLE_WIDTH"],
                             conf["FOOT_RECTANGLE_HEIGHT"],
                             color='blue' if s_v[time_instant] == conf["RIGHT_FOOT"] else 'green',
                             alpha=0.7)
            # Apply rotation
            t = (matplotlib.transforms.Affine2D().rotate(foot_orient) +
                 matplotlib.transforms.Affine2D().translate(step_x, step_y) + ax.transData)
            rect.set_transform(t)

            # Add rectangle to the plot
            ax.add_patch(rect)

        plt.legend()
        plt.xlim(-5, 12)
        plt.ylim(-5, 12)
        plt.show()
