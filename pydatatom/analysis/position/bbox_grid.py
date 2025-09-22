import numpy as np
from scipy.spatial.distance import pdist, squareform


def bbox_grid(bbox: np.array, fx: np.array, fy: np.array):
    pts = np.asarray(bbox).reshape(-1, 2)
    dst = squareform(pdist(pts))

    # find opposing points
    i, j = np.unravel_index(np.argmax(dst), dst.shape)
    left, right = pts[i], pts[j]
    if left[0] > right[0]:
        # left has a larger x coordinate than right
        left, right = pts[j], pts[i]

    # find reference point
    ref = pts[[k for k in range(3) if k not in (i, j)]][0]

    if ref[0] < left[0]:
        # reference is left
        w = np.linalg.vector_norm(right - ref)
        h = np.linalg.vector_norm(left - ref)
    else:
        # reference is right
        w = np.linalg.vector_norm(left - ref)
        h = np.linalg.vector_norm(right - ref)

    x = ref[0] - w * (fx - fx.min()) / (fx.max() - fx.min())
    y = ref[1] - h * (fy - fy.min()) / (fy.max() - fy.min())

    xy = np.stack(np.meshgrid(x, y), axis=-1)

    # rotate
    theta = np.arccos(-(left - ref)[0] / np.linalg.vector_norm(left - ref))

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    xy = (xy - ref) @ R.T + ref

    return xy.reshape(-1, 2)
