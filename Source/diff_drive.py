import time
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from casadi import MX, vertcat
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from sympy import symbols


class DifferentialDrive:
    def __init__(self, n, m, N, N_simul):
        """
        Initialize an object representing a differential drive vehicle.

        :param n: The number of state variables.
        :param m: The number of control inputs.
        :param N:
        :param N_simul:
        """
        # TODO: In this kinematic model, shouldn't be always N=3 and M=2? Then n and m should not be constructor's
        #  parameters
        self.n = n  # states dimension
        self.m = m  # control dimension
        self.N = N  # horizon
        self.N_simul = N_simul  # simulation horizon

        # Initialize the X, Y and Theta variables (regarding position and orientation of the robot),
        # and put them in the state vector (x y theta).
        self.x = cs.MX.sym('x')
        self.y = cs.MX.sym('y')
        self.theta = cs.MX.sym('theta')
        self.state = cs.vertcat(self.x, self.y, self.theta)  # my state vector

        # Initialize the V and Omega variables (regarding translational and angular velocity of the robot),
        # and put them in the controls vector (v omega).
        self.v = cs.MX.sym("v")
        self.w = cs.MX.sym("w")
        self.controls = cs.vertcat(self.v, self.w)  # my control vector

        # Define the kinematic model of the differential drive
        self.xdot = self.v * cs.cos(self.theta)
        self.ydot = self.v * cs.sin(self.theta)
        self.thetadot = self.w
        self.statedot = cs.vertcat(self.xdot, self.ydot, self.thetadot)
        self.kinemodel = cs.Function("km", [self.state, self.controls], [self.statedot])

    @staticmethod
    def move_and_plot(x_trajectory: np.ndarray, y_trajectory: np.ndarray, theta_trajectory: np.ndarray,
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
            [cs.cos(theta_trajectory), -cs.sin(theta_trajectory)],
            [cs.sin(theta_trajectory), cs.cos(theta_trajectory)]
        ]).squeeze()
        # Define the expression that puts the robot in the appropriate position and orientation
        triangle_poses = rotation_matrix.transpose(2, 0, 1) @ vert + np.array([[x_trajectory, y_trajectory]]).T
        # Define the expression that puts the robot's barycenter in the appropriate position
        barycenter_traj = barycenter + np.array([[x_trajectory, y_trajectory]]).T

        # Set up the plot
        fig, ax = plt.subplots()
        ax.set_xlim(0, 6)  # Set x-axis limits
        ax.set_ylim(0, 6)  # Set y-axis limits
        ax.set_aspect('equal')  # Set equal aspect ratio for accurate proportions

        # Initialize the triangle
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
        ani = FuncAnimation(fig, update, frames=len(triangle_poses), interval=200, blit=True)  # 1 frame per second
        ani.save(path_to_gif, writer='ffmpeg')
        # Display the animation
        plt.show()


if __name__ == "__main__":
    model = DifferentialDrive(1, 1, 1, 1)

    T = 25
    x_traj = np.concat([np.linspace(0, 3, T), np.linspace(3, 5, 10)])
    y_traj = np.concat([np.linspace(0, 3, T), np.linspace(3, 3, 10)])
    theta_traj = np.concat([np.linspace(0, cs.pi / 2, T), np.linspace(cs.pi / 2, cs.pi / 2, 10)])
    model.move_and_plot(x_traj, y_traj, theta_traj, path_to_gif='./Assets/Animations/diff_drive.gif',
                        triangle_height=1, triangle_width=0.8)
