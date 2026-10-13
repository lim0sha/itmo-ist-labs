import numpy as np
import pandas as pd

from .utils.basic import count_events


def add_features(g):
    g = g.sort_values("ts").copy()
    f = g["fuel"]
    for k in (1, 2, 3, 4, 6, 12):
        g[f"d{k}"] = f.diff(k).fillna(0.0)
    g["ign_on"] = (g["ign"] > 0.5).astype(float)
    g["speed_low"] = (g["speed"] < 3).astype(float)
    hour = g["ts"].dt.hour + g["ts"].dt.minute / 60.0
    g["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    g["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    g["is_day"] = ((hour >= 6) & (hour <= 22)).astype(float)
    g["n_msg_c"] = g["n_msg"].clip(upper=60) / 60.0
    g["fuel_roll3"] = f.rolling(3, center=True, min_periods=1).mean()
    g["fuel_roll7"] = f.rolling(7, center=True, min_periods=1).mean()
    g["fuel_resid"] = f - g["fuel_roll7"]
    g["fuel_roll_std5"] = f.rolling(5, center=True, min_periods=1).std().fillna(0.0)
    g["d1_max3"] = g["d1"].rolling(3, center=True, min_periods=1).max()
    g["d1_min3"] = g["d1"].rolling(3, center=True, min_periods=1).min()
    return g


def enrich(df):
    return pd.concat([add_features(g) for _, g in df.groupby("vehicle")],
                     ignore_index=True)


def val_vehicles(df):
    train = sorted(df[df.split == "train"].vehicle.unique())
    counts = {v: count_events(df[df.vehicle == v].label) for v in train}
    return set(sorted(train, key=lambda v: -counts[v])[1::5][:9])
