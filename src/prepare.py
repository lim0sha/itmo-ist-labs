from .config import features_for
from .windows import make_windows


def prepare(df, feature_set, seq_len, norm, val_veh):
    d = df.copy()
    features = features_for(feature_set)
    if norm:
        fit = d[d.split.eq("train") & ~d.vehicle.isin(val_veh)]
        mean = fit[features].mean()
        std = fit[features].std().replace(0, 1.0)
        d[features] = (d[features] - mean) / std
    train = d[d.split.eq("train") & ~d.vehicle.isin(val_veh)]
    val = d[d.vehicle.isin(val_veh)]
    test = d[d.split == "test"]
    return (make_windows(train, features, seq_len)
            + make_windows(val, features, seq_len)
            + make_windows(test, features, seq_len))
