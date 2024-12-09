import numpy as np
import matplotlib.pyplot as plt
import math

# ===== CONSTANTS =====
g = 9.81 # gravity (m/s^2)
dt = 0.01 # infinitesimal time increment (s)


# ===== Linear Inverted Pendulum =====
class HumanoidLip:
    com_height = 1.5 # CoM constant height (m)
    foot_offset = 0.2 # foot left and right offset w.r.t. CoM (m)
    turning_rate = 0.156 * math.pi # turning rate of the robot (rad/s)


    def __init__(self, initial_state, goal_position):
        self.timestep = 0
        self.goal = np.array(goal_position, dtype=float)

        # robot state = [px, vx, py, vy, theta]
        self.state = np.array(initial_state, dtype=float)
        self.com_position = np.array([initial_state[0], initial_state[2]], dtype=float)
        self.com_velocity = np.array([initial_state[1], initial_state[3]], dtype=float)
        self.heading_angle = np.array(initial_state[4], dtype=float)

        # # computing initial stances position [fx, fy]
        # r_stance_x = self.com_position[0] + self.com_position[0] * math.sin(self.heading_angle)
        # r_stance_y = self.com_position[1] - self.com_position[1] * math.cos(self.heading_angle)
        # self.r_stance_state = np.array([r_stance_x, r_stance_y], dtype=float)
        #
        # l_stance_x = self.com_position[0] - self.com_position[0] * math.sin(self.heading_angle)
        # l_stance_y = self.com_position[1] + self.com_position[1] * math.cos(self.heading_angle)
        # self.l_stance_state = np.array([l_stance_x, l_stance_y], dtype=float)

        # supposing to use always the right foot to start
        r_stance_x = self.com_position[0] + self.com_position[0] * math.sin(self.heading_angle)
        r_stance_y = self.com_position[1] - self.com_position[1] * math.cos(self.heading_angle)
        self.stance_state = np.array([r_stance_x, r_stance_y], dtype=float)

        # python list since we don't need to perform computations on it
        self.state_history = [self.state]

    def position_history(self):
        positions = [[h[0], h[2]] for h in self.state_history]
        return positions

    def dynamics(self):
        # computing acceleration
        com_acceleration = g/self.com_height * (self.com_position - self.stance_state)
        # velocity bi deriving acceleration
        self.com_velocity = com_acceleration * dt
        # position by deriving velocity
        self.com_position = self.com_velocity * dt
        # heading_angle by deriving CONSTANT turning rate
        self.heading_angle += self.turning_rate * dt

        state = [
            self.com_position[0],
            self.com_velocity[0],
            self.com_position[1],
            self.com_velocity[1],
            self.heading_angle
        ]
        self.state = np.array(state, dtype=float)

        # appending state
        self.state_history.append(self.state)


# ===== CONTROLLER =====
# TODO


# ===== SIMULATION =====
initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]  # at origin with 0 velocity and 0 angle
goal_position = [10.0, 10.0]

# FIXME: maybe not needed with a while(True)
#  but not sure if is the right way
time_horizon = 100

# Initialize the robot
robot = HumanoidLip(initial_state, goal_position)

# Simulation loop
for _ in range(time_horizon):
    # TODO: apply control
    robot.dynamics()

# Extract robot trajectory
trajectory = np.array(robot.position_history())

# Plot the results
plt.figure(figsize=(10, 10))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Robot Trajectory", marker='o')
plt.scatter(*goal_position, color='green', label="Goal Position", s=100)
plt.scatter(initial_state[0], initial_state[1], color='blue', label="Start Position", s=100)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Trajectory toward goal")
plt.legend()
plt.grid()
plt.show()
