import os

import matplotlib.pyplot as plt
import numpy as np


class PlotUtils:
    """
    A class of utility functions to generate the plots that will be used in the report.
    """

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
            ax.set_ylabel(ylabel_str)
            ax.set_xlabel("Simulation Step k" if samples_per_second is None else "Time (s)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f'{path_to_pdf}/evolution_{i}.pdf')
            plt.show()
