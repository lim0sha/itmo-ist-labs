import numpy as np


def postprocess(proba1, meta, threshold=0.5, min_len=1, max_gap=0):
    label = np.zeros(len(meta), dtype="int64")
    for _, sub in meta.groupby("vehicle"):
        pos = sub.index.to_numpy()
        p = _fill_gaps(proba1[pos] >= threshold, max_gap)
        p = _drop_short(p, min_len)
        label[pos[p]] = 1
    return label


def _fill_gaps(p, max_gap):
    if max_gap <= 0 or not p.any():
        return p
    idx = np.where(p)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            p[a:b] = True
    return p


def _drop_short(p, min_len):
    if min_len <= 1 or not p.any():
        return p
    start = None
    for i in range(len(p) + 1):
        cur = p[i] if i < len(p) else False
        if cur and start is None:
            start = i
        elif not cur and start is not None:
            if i - start < min_len:
                p[start:i] = False
            start = None
    return p
