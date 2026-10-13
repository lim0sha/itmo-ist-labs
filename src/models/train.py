import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from ..config import DEVICE
from .net import FuelNet, class_weights


def predict_proba(model, X, batch=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(DEVICE)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_model(Xtr, ytr, Xva, yva, cfg, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FuelNet(Xtr.shape[2], cfg["hidden"], cfg["layers"], 3,
                    cfg["dropout"], cfg["bidir"], cfg["attention"]).to(DEVICE)
    weights = class_weights(ytr, power=cfg["cweight_power"],
                            max_weight=cfg["max_weight"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max",
                                                       factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                      torch.from_numpy(ytr)),
                        batch_size=cfg["batch"], shuffle=True)
    best_f1, best_state = -1.0, None
    for _ in range(cfg["epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        vp = predict_proba(model, Xva).argmax(1)
        f1 = f1_score(yva, vp, labels=[1], average="macro", zero_division=0)
        sched.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model
