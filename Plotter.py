from matplotlib import pyplot as plt
from typing import List
import numpy as np
from IPython.display import clear_output

class Plotter:
    @staticmethod
    def normalize(to_normalize: List):
        max_value = np.amax(to_normalize)
        if max_value == 0:
            max_value = 1

        return [[value / max_value * 255 for value in row] for row in to_normalize]

    @staticmethod
    def plotSinogram(sinogram: List):
        clear_output(wait=True)
        plt.imshow(Plotter.normalize(sinogram), cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()
