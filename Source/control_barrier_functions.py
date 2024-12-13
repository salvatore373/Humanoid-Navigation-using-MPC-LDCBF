import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# np.random.seed(45)
obstacles = np.random.randint(-100, 100, (2, 5))
# origin = [-30, 30]
origin_positions = [[-30 + i, 30 - i] for i in range(-20, 21)]

# to plot the current frame
def update(frame):
    origin = origin_positions[frame]
    plt.cla()

    for obs in obstacles.transpose():
        # origin-to-obstacle line: ax+by=c
        plt.plot([origin[0], obs[0]], [origin[1], obs[1]], 'black')

        # normal line -> a(-y) + b(x) = c
        robot_to_obstacle_vector = (obs[0]-origin[0], obs[1]-origin[1])
        plt.axline(
            (obs[0], obs[1]),
            (-(obs[1]+origin[0])+obs[0], obs[0]+origin[1]+obs[1]),
            color="black", linestyle=(0, (5, 5)))

        # filling gray area
        """
            y = m*x + b
            m = (y_2-y_1)/(x_2-x_1)
            b = y
        """
        if obs[0] - origin[0] != 0:
            robot_obstacle_slope = (obs[1] - origin[1]) / (obs[0] - origin[0])
            normal_slope = -1/robot_obstacle_slope
            # obs[0] & obs[1] are whatever points lying in normal line
            normal_intercept = obs[1] - normal_slope * obs[0]
        elif obs[1] - origin[1] == 0:
            robot_obstacle_slope = 0
            normal_slope = np.inf
        else:
            normal_slope = None
            robot_obstacle_slope = np.inf


        x = np.linspace(-200, 200, 4000)
        # horizontal case
        if robot_obstacle_slope == 0:
            if obs[0] > origin[0]:
                plt.fill_betweenx(x, obs[0], 200, color='gray', alpha=0.5)
            else:
                plt.fill_betweenx(x, -200, obs[0], color='gray', alpha=0.5)
        # normal slope
        elif normal_slope is not None:
            y = normal_slope * x + normal_intercept

            if obs[0] > origin[0] and obs[1] > origin[1]:
                plt.fill_between(x, y, 200, color='gray', alpha=0.5)
            elif obs[0] < origin[0] and obs[1] > origin[1]:
                plt.fill_between(x, y, 200, color='gray', alpha=0.5)
            elif obs[0] < origin[0] and obs[1] < origin[1]:
                plt.fill_between(x, y, -200, color='gray', alpha=0.5)
            else:  # Region to the left of the origin
                plt.fill_between(x, y, -200, color='gray', alpha=0.5)
        # vertical slope
        else:
            if obs[1] > origin[1]:
                plt.fill_between(x, obs[1], 200, color='gray', alpha=0.5)
            else:
                plt.fill_between(x, -200, obs[1], color='gray', alpha=0.5)

    # plot all obstacles
    plt.scatter(obstacles[0], obstacles[1], color='cyan')
    # plot origin
    plt.scatter(origin[0], origin[1], color='red', s=500)
    # set boundaries and equal aspect
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=len(origin_positions), interval=200)
ani.save('cbf_animation.gif', writer='ffmpeg')
plt.show()
