import abc
from typing import Callable

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class BaseAnimationHelper:
    __metaclass__ = abc.ABCMeta

    # Assuming that the plot is a square, the minimum and maximum value that x and y coordinated can have
    MIN_COORD = -100
    MAX_COORD = +100

    def __init__(self, fig=None, ax=None):
        """
        Initializes a new instance of the utility to show the animation.

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

    def show_animation(self, path_to_gif: str, num_frames: int = 20, interval: int = 200, update: Callable = None):
        """
        Shows the animation painted by self.update() and saves it at save_path.
        :param path_to_gif: The path to the GIF file where the animation will be saved.
        :param num_frames: The number of frames in the animation.
        :param interval: Delay between frames in milliseconds.
        :param update: The function to generate the frames of the animation. The default is self._full_update()
        """
        if update is None:
            update = self._full_update

        ani = FuncAnimation(self.fig, update, frames=num_frames, interval=interval)
        ani.save(path_to_gif, writer='ffmpeg')
        plt.show()

    @abc.abstractmethod
    def update(self, frame: int):
        """
        Updates the canvas to display the state corresponding to the given frame number.

        :param frame: The number of the frame whom state has to be represented.
        """
        raise NotImplementedError("The update() method must be overridden by a subclass.")

    def _full_update(self, frame: int):
        """
        Executes self.update() with some additional boilerplate after the canvas update.
        :param frame: The number of the frame whom state has to be represented.
        """
        # Update the canvas
        self.update(frame)

        # Set the boundaries of the plot and equal aspect ratio
        plt.xlim(BaseAnimationHelper.MIN_COORD, BaseAnimationHelper.MAX_COORD)
        plt.ylim(BaseAnimationHelper.MIN_COORD, BaseAnimationHelper.MAX_COORD)
        self.ax = plt.gca()
        self.ax.set_aspect('equal', adjustable='box')
