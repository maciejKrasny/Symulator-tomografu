import numpy as np

class Bresenham:

    @staticmethod
    def bresenham_line(emitter, detector):
        """
        Runs bresenham algorithm

        :param emmiter: Tuple of coordinates (x, y)
        :param detector: Tuple of coordinates (x, y)
        :return array of bresenham line pixels
        """
        # Setup initial
        x1, y1 = emitter
        x2, y2 = detector
        dx, dy = (x2 - x1), (y2 - y1)

        if abs(dy) < abs(dx):
            if x1 > x2:
                path = Bresenham.plot_line_low(detector, emitter)
            else:
                path = Bresenham.plot_line_low(emitter, detector)
        else:
            if y1 > y2:
                path = Bresenham.plot_line_high(detector, emitter)
            else:
                path = Bresenham.plot_line_high(emitter, detector)
        return np.array(path)

    @staticmethod
    def plot_line_low(emitter, detector):
        # Setup initial
        x1, y1 = emitter
        x2, y2 = detector
        dx, dy = (x2 - x1), (y2 - y1)

        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy

        D = 2 * dy - dx
        y = y1

        path = []
        for x in range(x1, x2 + 1):
            path.append((x, y))
            if D > 0:
                y = y + yi
                D = D - 2 * dx

            D = D + 2 * dy

        return path

    @staticmethod
    def plot_line_high(emitter, detector):
        # Setup initial
        x1, y1 = emitter
        x2, y2 = detector
        dx, dy = (x2 - x1), (y2 - y1)

        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = 2 * dx - dy
        x = x1

        path = []
        for y in range(y1, y2 + 1):
            path.append((x, y))
            if D > 0:
                x = x + xi
                D = D - 2 * dy
            D = D + 2 * dx
        return path

