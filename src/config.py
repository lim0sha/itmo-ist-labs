import torch

DATA_URL = ("https://raw.githubusercontent.com/hehehelp123/"
            "fuel-refuel-lab/main/data/fuel_5min.csv")
LOCAL_CSV = "fuel_5min.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_FEATURES = ["fuel", "d1", "d2", "d3", "d4"]
ALL_FEATURES = [
    "fuel", "d1", "d2", "d3", "d4", "d6", "d12",
    "ign", "ign_on", "speed", "speed_low", "n_msg_c",
    "hour_sin", "hour_cos", "is_day",
    "fuel_roll3", "fuel_roll7", "fuel_resid", "fuel_roll_std5",
    "d1_max3", "d1_min3",
]

DEFAULT_CFG = dict(
    hidden=64, layers=2, dropout=0.2, bidir=True, attention=True,
    lr=1e-3, batch=512, epochs=30, cweight_power=0.5, max_weight=30.0,
)


def features_for(feature_set):
    return BASE_FEATURES if feature_set == "base" else ALL_FEATURES
