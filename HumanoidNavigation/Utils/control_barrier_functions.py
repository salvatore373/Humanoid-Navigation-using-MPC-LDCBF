import matplotlib.pyplot as plt
import numpy as np

from HumanoidNavigation.Utils.BaseAnimationHelper import BaseAnimationHelper


class CBFAnimationHelper(BaseAnimationHelper):
    def __init__(self, obstacles, origin_positions, fig=None, ax=None):
        """
        Initializes a new instance of the utility to show the animation regarding control barrier functions.

        :param obstacles: The position of the cells of the grid containing obstacles.
        :param origin_positions: The positions of the point to move in each frame of the animation
        :param fig: The first element returned by plt.subplots(). If either fig or axis is not provided, a new one
        is created.
        :param ax: The second element returned by plt.subplots(). If either fig or axis is not provided, a new one
        is created.
        """
        super(CBFAnimationHelper, self).__init__(fig, ax)
        self.obstacles = obstacles
        self.origin_positions = origin_positions

        # The list containing some of the positions of the origin in the frames that have already been displayed
        self.origin_history = []
        # The total number of frames in the animation
        self.num_frames = len(origin_positions)

    def update(self, frame: int):
        # Get the position of the origin (red point) in this frame
        origin = self.origin_positions[frame]
        # origin_history = origin_positions[max(1, frame-10):frame]
        # Periodically store the current position of the origin
        if frame % 10 == 0:
            self.origin_history.append(origin)

        # Clear the canvas
        plt.cla()

        for obs in self.obstacles.transpose():
            # Plot the line from the origin this obstacle
            plt.plot([origin[0], obs[0]], [origin[1], obs[1]], 'black')  # origin-to-obstacle line: ax+by=c

            # Plot the line normal to the line from the origin this obstacle
            robot_to_obstacle_vector = (obs[0] - origin[0], obs[1] - origin[1])  # normal line: a(-y) + b(x) = c
            plt.axline(
                (obs[0], obs[1]),
                (-(obs[1] + origin[0]) + obs[0], obs[0] + origin[1] + obs[1]),
                color="black", linestyle=(0, (5, 5)))

            # Find the parameters of the line defining the area to paint
            # filling gray area
            #       y = m*x + b
            #       m = (y_2-y_1)/(x_2-x_1)
            #       b = y
            if obs[0] - origin[0] != 0:
                robot_obstacle_slope = (obs[1] - origin[1]) / (obs[0] - origin[0])
                if robot_obstacle_slope == 0:
                    normal_slope = -np.inf
                else:
                    normal_slope = -1 / robot_obstacle_slope
                # obs[0] & obs[1] are whatever points lying in normal line
                normal_intercept = obs[1] - normal_slope * obs[0]
            else:
                normal_slope = None
                robot_obstacle_slope = np.inf

            # Paint the area of the plot that starts from this obstacle and do not contain the origin
            x = np.linspace(-200, 200, 4000)
            if robot_obstacle_slope == 0:  # horizontal case
                if obs[0] > origin[0]:
                    plt.fill_betweenx(x, obs[0], 200, color='gray', alpha=0.5)
                else:
                    plt.fill_betweenx(x, -200, obs[0], color='gray', alpha=0.5)
            elif normal_slope is not None:  # normal slope case
                y = normal_slope * x + normal_intercept

                if obs[0] > origin[0] and obs[1] > origin[1]:
                    plt.fill_between(x, y, 200, color='gray', alpha=0.5)
                elif obs[0] < origin[0] and obs[1] > origin[1]:
                    plt.fill_between(x, y, 200, color='gray', alpha=0.5)
                elif obs[0] < origin[0] and obs[1] < origin[1]:
                    plt.fill_between(x, y, -200, color='gray', alpha=0.5)
                else:  # Region to the left of the origin
                    plt.fill_between(x, y, -200, color='gray', alpha=0.5)
            else:  # vertical slope case
                if obs[1] > origin[1]:
                    plt.fill_between(x, obs[1], 200, color='gray', alpha=0.5)
                else:
                    plt.fill_between(x, -200, obs[1], color='gray', alpha=0.5)

        # Plot all the obstacles as cyan points
        plt.scatter(self.obstacles[0], self.obstacles[1], color='cyan')

        # Plot the origin as a red point
        plt.scatter(origin[0], origin[1], color='red', s=500)

        # Plot the previous positions of the origin as blurred red points
        for i, h in enumerate(self.origin_history):
            plt.scatter(h[0], h[1], color='red', s=500, alpha=i * 0.1)


if __name__ == '__main__':
    # The number of obstacles to put in the plot
    num_obstacles = 5
    # The position of the cells of the grid containing obstacles
    obstacles = np.random.randint(CBFAnimationHelper.MIN_COORD, CBFAnimationHelper.MAX_COORD, (2, num_obstacles))
    # The positions of the point to move in each frame of the animation
    origin_positions = [[-30 + i, 30 - i] for i in range(-20, 21)]

    animation_helper = CBFAnimationHelper(obstacles=obstacles,
                                          origin_positions=origin_positions)
    animation_helper.show_animation(path_to_gif='./Assets/Animations/cbf_animation.gif')
