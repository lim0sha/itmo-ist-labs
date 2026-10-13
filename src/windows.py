import numpy as np
import pandas as pd


def make_windows(frame, features, seq_len):
    center = seq_len // 2
    X, y, veh, ts = [], [], [], []
    for v, g in frame.groupby("vehicle"):
        vals = g[features].values.astype("float32")
        lab = g["label"].values.astype("int64")
        times = g["ts"].values
        for i in range(len(g) - seq_len + 1):
            X.append(vals[i:i + seq_len])
            y.append(lab[i + center])
            veh.append(v)
            ts.append(times[i + center])
    meta = pd.DataFrame({"vehicle": veh, "ts": pd.to_datetime(ts)})
    return np.asarray(X), np.asarray(y), meta
