import numpy as np
from matplotlib import pyplot as plt


class InteractiveBBoxBlobDetector:
    def __init__(self):
        self.bbox = []
        self.spots = None

    def fit(self, image, fx, fy):
        plt.close()

        if len(self.bbox) == 0:
            raise RuntimeError("run plot_clickable(image) first")

        bbox = np.asarray(self.bbox).reshape(-1, 2)
        tl, bl, tr = bbox

        # width and height of bbox
        w = tr[0] - tl[0]
        h = bl[1] - tl[1]

        # slope of bbox
        dy = tr[1] - tl[1]
        dx = bl[0] - tl[0]

        # rescale x and y coordinates of the array
        x = w * (fx - fx.min()) / (fx.max() - fx.min()) + tl[0]
        y = h * (fy - fy.min()) / (fy.max() - fy.min()) + tl[1]

        # combine x and y and adjust for slope
        xy = np.stack(np.meshgrid(x, y), axis=-1)
        xy[:, :, 1] += dy * np.linspace(0, 1, xy.shape[1])
        xy[:, :, 0] += dx * np.linspace(0, 1, xy.shape[0])[:, np.newaxis]

        self.spots = xy

    def plot_clickable(self, image):
        fig, ax = plt.subplots(figsize=(16, 3))

        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(
            "Click (1) top left atom, (2) bottom left atom, (3) top right atom"
        )

        def onclick(event):
            # only react to right clicks
            if event.button != 3:
                return

            x, y = event.xdata, event.ydata

            ax.scatter(x, y, s=20, color="red")
            self.bbox.append((x, y))

            fig.canvas.draw()

        def onkey(event):
            if event.key == "delete" and len(ax.lines) > 0:
                ax.lines[0].remove()

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkey)

        plt.tight_layout()
        plt.show()

    def plot_result(self, image):
        if self.spots is None:
            raise RuntimeError("run fit first")
        xy = self.spots

        fig, ax = plt.subplots(figsize=(15, 3))

        ax.imshow(image)
        ax.set_axis_off()

        for idx, (x, y) in enumerate(xy.reshape(-1, 2)):
            plt.text(x + 2, y + 2, str(idx), color="yellow", fontsize=9)
        ax.scatter(xy.reshape(-1, 2)[:, 0], xy.reshape(-1, 2)[:, 1], s=30, color="red")

        plt.show()
