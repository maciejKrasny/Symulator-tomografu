import numpy as np
import math
from scipy.fftpack import fft, ifft, fftfreq
from Plotter import Plotter
from Bresenham import Bresenham

class Scanner:
    def __init__(self, steps, detectors_count, spread):
        self.alfa = 360
        self.steps = steps
        self.step_angle = self.alfa / steps
        self.detectors_count = detectors_count
        self.spread = spread

    def create_sinogram(self, img):
        self.space = img
        step_angle = math.radians(self.step_angle)
        center = int(self.space.shape[0] / 2)
        radius = center - 5
        actual_angle = 0
        sinogram = []

        for step in range(self.steps):
            emitter = (
                center + int(radius * np.cos(actual_angle)),
                center + int(radius * np.sin(actual_angle)),
            )
            selected_detector_angle = math.radians(180 - self.spread / 2) + actual_angle
            detector_step = math.radians(self.spread / (self.detectors_count - 1))
            measurements = []

            for j in range(self.detectors_count):
                detector = (
                    center + int(radius * np.cos(selected_detector_angle)),
                    center + int(radius * np.sin(selected_detector_angle)),
                )
                selected_detector_angle += detector_step
                path = Bresenham.bresenham_line(emitter, detector)
                measurements.append(self.space[path[:, 0], path[:, 1]].mean())

            sinogram.append(measurements)
            actual_angle += step_angle
            Plotter.plotSinogram(sinogram)

        return np.array(sinogram)

    def deconstruct_sinogram(self, sinogram):
        height, width = self.space.shape
        result_img = np.zeros((width, height))
        step_angle = math.radians(self.step_angle)
        actual_angle = 0

        center = int(width / 2)
        radius = center - 5

        for i in range(self.steps):
            w, h = height - 5, width - 5
            emitter = (
                center + int(radius * np.cos(actual_angle)),
                center + int(radius * np.sin(actual_angle)),
            )
            selected_detector_angle = math.radians(180 - self.spread / 2) + actual_angle
            detector_step = math.radians(self.spread / (self.detectors_count - 1))

            for j in range(self.detectors_count):
                detector = (
                    center + int(radius * np.cos(selected_detector_angle)),
                    center + int(radius * np.sin(selected_detector_angle))
                )
                selected_detector_angle += detector_step
                path = Bresenham.bresenham_line(emitter, detector)
                for p in path:
                    result_img[p[0]][p[1]] += sinogram[i][j]

            actual_angle += step_angle
            Plotter.plotSinogram(np.array(result_img)[50:-50, 50:-50])

        return np.array(result_img)[50:-50, 50:-50]

    def filter(self, sinogram):
        f = fftfreq(sinogram.shape[0]).reshape(-1, 1)
        fourier_filter = 2 * np.abs(f)
        projection = fft(sinogram, axis=0) * fourier_filter
        return np.real(ifft(projection, axis=0))

    @staticmethod
    def bresenham_line_inverse(img, emitter, detector, brightness):
        # Setup initial conditions
        x1, y1 = emitter
        x2, y2 = detector

        x = x1
        y = y1

        if x1 < x2:
            xi = 1
            dx = x2 - x1
        else:
            xi = -1
            dx = x1 - x2

        if y1 < y2:
            yi = 1
            dy = y2 - y1
        else:
            yi = -1
            dy = y1 - y2

        img[x][y] += brightness

        if dx > dy:
            ai = (dy - dx) * 2
            bi = dy * 2
            d = bi - dx
            while x != x2:
                if d >= 0:
                    x += xi
                    y += yi
                    d += ai
                else:
                    d += bi
                    x += xi

                img[x][y] += brightness
        else:
            ai = (dx - dy) * 2
            bi = dx * 2
            d = bi - dy
            while y != y2:
                if d >= 0:
                    x += xi
                    y += yi
                    d += ai
                else:
                    d += bi
                    y += yi

                img[x][y] += brightness

        return img