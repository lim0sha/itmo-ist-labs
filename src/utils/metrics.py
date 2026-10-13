import numpy as np

from .basic import blocks

TOLERANCE = np.timedelta64(5, "m")


def _overlaps(ps, pe, intervals):
    return [k for k, (a, b) in enumerate(intervals)
            if ps <= b + TOLERANCE and pe >= a - TOLERANCE]


def event_metrics(y_true, y_pred, meta):
    n_true = n_true_hit = n_pred = n_pred_hit = 0
    for _, sub in meta.groupby("vehicle"):
        pos = sub.index.to_numpy()
        times = sub["ts"].values
        t_true = blocks(y_true[pos] == 1, times)
        t_pred = blocks(y_pred[pos] == 1, times)
        t_amb = blocks(y_true[pos] == 2, times)
        hit_true = set()
        for ps, pe in t_pred:
            hit = _overlaps(ps, pe, t_true)
            if hit:
                hit_true.update(hit)
                n_pred += 1
                n_pred_hit += 1
            elif _overlaps(ps, pe, t_amb):
                continue
            else:
                n_pred += 1
        n_true += len(t_true)
        n_true_hit += len(hit_true)
    return _summarise(n_true, n_true_hit, n_pred, n_pred_hit)


def _summarise(n_true, n_true_hit, n_pred, n_pred_hit):
    tp, fn = n_true_hit, n_true - n_true_hit
    fp = n_pred - n_pred_hit
    precision = n_pred_hit / n_pred if n_pred else 0.0
    recall = n_true_hit / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}
