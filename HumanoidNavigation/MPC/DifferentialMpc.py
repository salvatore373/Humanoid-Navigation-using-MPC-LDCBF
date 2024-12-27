import time 
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from BaseMpc import MpcSkeleton
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from sympy import symbols

class DifferentialDriveMPC(MpcSkeleton):
    def __init__(self, state_dim, control_dim, N_horizon, N_simul, sampling_time, goal=None, cbf=None):
        super().__init__(state_dim, control_dim, N_horizon, N_simul, sampling_time)
        # if goal remains None then it's assumed to be a trajectory tracking problem
        self.goal = goal
        if self.goal is None:
            self.reference = self.optim_prob.parameter(self.state_dim-1, self.N_horizon) # in this case we are only defining a cartesian trajectory

        self.add_constraints()
        self.cost_function()

    def integrate(self, X, U):
        # Euler Integration
        return cs.vertcat(U[0]*cs.cos(X[2]), U[0]*cs.sin(X[2]), U[1])

    def add_constraints(self):
        self.optim_prob.subject_to(self.X_mpc[:,0] == self.x0)
        if self.goal is not None:
            self.optim_prob.subject_to(self.X_mpc[:,self.N_horizon] == self.goal)
        for k in range(self.N_horizon):
            self.optim_prob.subject_to(self.X_mpc[:,k+1] == self.X_mpc[:,k] + self.sampling_time*self.integrate(self.X_mpc[:,k], self.U_mpc[:,k]))

    def cost_function(self):
        cost = cs.sumsqr(self.U_mpc) # input effort
        if self.goal is None: # hence we have trajectory tracking
            reference_cost = 0
            for k in range(self.N_horizon-1):
                reference_cost += cs.sumsqr(self.X_mpc[:2,k] - self.reference[:, k])
            terminal_cost = cs.sumsqr(self.X_mpc[:2, -1] - self.reference[:, -1]) # not sure if it's -1 or -2 (double check this)
            weight = 200
            cost += weight*reference_cost + weight*terminal_cost
        self.optim_prob.minimize(cost)

    def plot(self, x_pred, ref):
        # you need to pass x_pred because is not defined inside this class while ref it's only defined as a casadi parameter
        plt.plot(0, 0, marker='o', color="cornflowerblue", label="Start")
        if self.goal is not None:
            plt.plot(self.goal[0], self.goal[1], marker='o', color="darkorange", label="Goal")
        else:
            plt.plot(ref[0,:], ref[1,:], color="yellowgreen", label="Reference Trajectory")
        plt.plot(x_pred[0,:], x_pred[1,:], color="mediumpurple", label="Predicted Trajectory")
        plt.legend()
        plt.show()

    def simulation(self, ref=None):
        # you need to pass your reference which will be inserted in your casadi parameter
        # container for the final predicted trajectory
        X_pred = np.zeros(shape=(self.state_dim, self.N_simul+1))
        U_pred = np.zeros(shape=(self.control_dim, self.N_simul))
        computation_time = np.zeros(self.N_simul)
        for k in range(self.N_simul):
            starting_iter_time = time.time()
            self.optim_prob.set_value(self.x0, X_pred[:,k])

            if self.goal is None and ref is not None:
                self.optim_prob.set_value(self.reference, ref[:, k:k+self.N_horizon]) # set the actual reference equal to the reference from k to k+N_horizon
            
            kth_solution = self.optim_prob.solve()
            U_pred[:,k] = kth_solution.value(self.U_mpc[:,0])
            self.optim_prob.set_initial(self.X_mpc, kth_solution.value(self.X_mpc))
            self.optim_prob.set_initial(self.U_mpc, kth_solution.value(self.U_mpc))
            X_pred[:,k+1] = X_pred[:,k] + self.sampling_time*self.integrate(X_pred[:,k], U_pred[:,k]).full().squeeze(-1)
            computation_time[k] = time.time() - starting_iter_time
        
        print(f"Average Computation time: {np.mean(computation_time)*1000} ms")
        self.plot(X_pred, ref)
        return X_pred
    
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
            [np.cos(theta_trajectory),-np.sin(theta_trajectory)],
            [np.sin(theta_trajectory), np.cos(theta_trajectory)]
        ]).squeeze()
        # Define the expression that puts the robot in the appropriate position and orientation
        triangle_poses = rotation_matrix.transpose(2, 0, 1) @ vert + np.array([[x_trajectory, y_trajectory]]).T
        # Define the expression that puts the robot's barycenter in the appropriate position
        barycenter_traj = barycenter + np.array([[x_trajectory, y_trajectory]]).T

        # Set up the plot
        fig, ax = plt.subplots()
        # TODO change this accordingly
        ax.set_xlim(-4, 1)  # Set x-axis limits
        ax.set_ylim(-1, 1)  # Set y-axis limits
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
    problem = "tracking" #"posture" or "tracking"

    if problem != "tracking":
        diffmpc = DifferentialDriveMPC(
            state_dim=3, control_dim=2, N_horizon=40, N_simul=300, sampling_time=0.01, goal=(-3,1,cs.pi)
        )
        diffmpc.simulation()
    elif problem == "tracking":
        goal = (-3,1,cs.pi)
        delta_t = 0.01
        N_simul = 1000
        N = 30
        k = 1.8
        alpha_x = k*np.cos(goal[2]) - 3*goal[0]
        beta_x = k*np.cos(goal[2])
        alpha_y = k*np.cos(goal[2]) - 3*goal[1]
        beta_y = k*np.cos(goal[2])
        s = np.linspace(0, 1, N_simul+N+1)
        x_traj = goal[0]*s**3 - alpha_x*(s-1)*s**2 + beta_x*s*(s-1)**2
        y_traj = goal[1]*s**3 - alpha_x*(s-1)*s**2 + beta_y*s*(s-1)**2
        reference = np.array([x_traj, y_traj])
        diffmpc = DifferentialDriveMPC(
            state_dim=3, control_dim=2, N_horizon=N, N_simul=N_simul, sampling_time=delta_t
        )
        pred_traj = diffmpc.simulation(ref=reference)
        print(pred_traj.shape)
        theta_traj = np.atan2(pred_traj[1,:], pred_traj[0,:])
        move_and_plot(pred_traj[0,:], pred_traj[1,:], theta_traj, path_to_gif='../Assets/Animations/diff_drive.gif',
                        triangle_height=0.1, triangle_width=0.08)