import os

import matplotlib.pyplot as plt
import numpy as np


class PlotUtils:
    """
    A class of utility functions to generate the plots that will be used in the report.
    """

    @staticmethod
    def compute_local_velocities(theta_evol, global_velocities):
        rotation_matrix = np.array([
            [np.cos(theta_evol), np.sin(theta_evol)],
            [-np.sin(theta_evol), np.cos(theta_evol)]
        ]).squeeze().transpose(2, 0, 1)
        return np.einsum('ijk,ik->ji', rotation_matrix, global_velocities.T)

    @staticmethod
    def plot_signals(data_tuples: list[tuple], path_to_pdf: str, samples_per_second: float = None):
        """
        Prints a vertical stack of plots. The i-th plot contains the data at position i in data_tuples.

        :params data_tuples: A list of tuples: the first element is a 2D matrix containing the data to plot;
         the second one is the label to put on the y-axis; the third one is optional, and is the label to assign to the
          data in each row of the matrix. The content of each tuple is plotted separately.
        :params path_to_pdf: The path to the folder where the plots will be stored as PDF files.
        :params samples_per_second: The number of elements belonging to the same second in the array to show on
         the Y-axis.
        """
        os.makedirs(path_to_pdf, exist_ok=True)

        for i, tpl in enumerate(data_tuples):
            # Create one plot per tuple
            fig, ax = plt.subplots(figsize=(8, 4))
            matrix, ylabel_str = tpl[0], tpl[1]
            steps = np.arange(matrix.shape[1]) if samples_per_second is None else \
                np.arange(matrix.shape[1]) / samples_per_second
            # Plot each row as a separate signal.
            for j in range(matrix.shape[0]):
                ax.plot(steps, matrix[j, :], label=None if len(tpl) == 2 else tpl[2][j])
            # Plot only an interval of the whole simulation, if that interval is provided
            if len(tpl) == 5:
                ax.set_xlim(tpl[3][0], tpl[3][1])
                ax.set_ylim(tpl[4][0], tpl[4][1])
            ax.set_ylabel(ylabel_str)
            ax.set_xlabel("Simulation Step k" if samples_per_second is None else "Time (s)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f'{path_to_pdf}/evolution_{i}.pdf')
            plt.show()

    @staticmethod
    def plot_com_and_zmp(path_to_pdf, com_x, com_y, zmp_x, zmp_y):
        # plt.figure()
        fig, ax = plt.subplots(figsize=(8, 4))

        # Create a line graph
        ax.plot(com_x, com_y, label='CoM')
        ax.plot(zmp_x, zmp_y, label='ZMP')

        # Adding titles and labels
        ax.set_title('CoM and ZMP (foot stance)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.legend()

        # Show the plot
        plt.tight_layout()
        plt.savefig(f'{path_to_pdf}/evolution_5.pdf')
        plt.show()
