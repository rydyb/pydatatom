import numpy as np
from matplotlib import pyplot as plt


class ClickableImage:
    def __init__(self):
        self._fig = None
        self._axes = None
        self._scatter = None
        self._points = []
        self._conn_ids = {}

    @property
    def figure(self):
        if self._fig is None:
            raise ValueError("no figure available, call plot() first")
        return self._fig

    @property
    def axes(self):
        if self._axes is None:
            raise ValueError("no axes available, call plot() first")
        return self._axes

    @property
    def points(self):
        return np.asarray(self._points)

    def _disconnect(self):
        if self._fig is None:
            return
        for cid in self._conn_ids.values():
            try:
                self._fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._conn_ids = {}

    def _update_scatter(self):
        if self._scatter is None:
            self._scatter = self._axes.scatter([], [], s=36, c="red", marker="o")
        if self._points:
            arr = np.asarray(self._points)
            self._scatter.set_offsets(arr[:, :2])
        else:
            self._scatter.set_offsets(np.empty((0, 2)))

    def plot(self, image):
        if self._fig is not None:
            self._disconnect()
            try:
                plt.close(self._fig)
            except Exception:
                pass

        self._fig, self._axes = plt.subplots(figsize=(16, 3))
        self._axes.imshow(image)
        self._axes.set_axis_off()
        self._axes.set_title(
            "Right-click to select a point, Delete/Backspace to remove the last point"
        )
        self._points = []
        self._scatter = None
        self._update_scatter()

        def onclick(event):
            if event.inaxes is not self._axes:
                return
            if event.xdata is None or event.ydata is None:
                return
            if event.button != 3:
                return
            self._points.append((float(event.xdata), float(event.ydata)))
            self._update_scatter()
            self._fig.canvas.draw_idle()

        def onkey(event):
            if event.key not in ("delete", "backspace"):
                return
            if not self._points:
                return
            self._points.pop()
            self._update_scatter()
            self._fig.canvas.draw_idle()

        def onclose(event):
            if self._fig is None or event.canvas is not self._fig.canvas:
                return
            self._disconnect()
            self._fig = None
            self._axes = None
            self._scatter = None

        self._conn_ids["button"] = self._fig.canvas.mpl_connect(
            "button_press_event", onclick
        )
        self._conn_ids["key"] = self._fig.canvas.mpl_connect("key_press_event", onkey)
        self._conn_ids["close"] = self._fig.canvas.mpl_connect("close_event", onclose)

        plt.tight_layout()
        plt.show()
