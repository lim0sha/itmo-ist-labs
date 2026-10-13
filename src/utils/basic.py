import numpy as np


def count_events(labels, positive=1):
    arr = np.asarray(labels) == positive
    padded = np.pad(arr, (1, 1), constant_values=False)
    return len(np.where(np.diff(padded))[0]) // 2


def blocks(mask, times):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((times[start], times[i - 1]))
            start = None
    if start is not None:
        out.append((times[start], times[len(mask) - 1]))
    return out
