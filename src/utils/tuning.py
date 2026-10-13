import numpy as np

from .metrics import event_metrics
from .postprocess import postprocess


def tune_postprocess(proba1, y, meta):
    best = None
    for thr in np.arange(0.05, 0.96, 0.05):
        for ml in (1, 2, 3, 4):
            for mg in (0, 1, 2):
                pred = postprocess(proba1, meta, thr, ml, mg)
                f1 = event_metrics(y, pred, meta)["f1"]
                key = (f1, thr, -mg, ml)
                if best is None or key > best[0]:
                    best = (key, thr, ml, mg)
    return best[1], best[2], best[3], best[0][0]
