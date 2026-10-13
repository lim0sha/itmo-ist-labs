import argparse
import os
import urllib.request

import pandas as pd

from src.config import DATA_URL, DEFAULT_CFG, LOCAL_CSV
from src.data import enrich, val_vehicles
from src.pipeline import run_experiment
from src.utils.seed import set_seed

STAGES = [
    ("0. baseline", "base", 7, False, False, False),
    ("1. + normalization", "base", 7, True, False, False),
    ("2. + features/BiLSTM/attention", "rich", 9, True, True, True),
]


def load_csv(path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path, parse_dates=["ts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=LOCAL_CSV)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    df = enrich(load_csv(args.csv))
    val_veh = val_vehicles(df)

    for title, feature_set, seq, norm, bidir, attention in STAGES:
        run_experiment(df, title, feature_set, seq, norm, bidir, attention,
                       val_veh, DEFAULT_CFG, seed=args.seed)


if __name__ == "__main__":
    main()
