import numpy as np
import torch
import torch.nn as nn


class FuelNet(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, n_classes=3,
                 dropout=0.2, bidir=True, attention=True):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True,
                           dropout=dropout, bidirectional=bidir)
        d = hidden * (2 if bidir else 1)
        self.use_attention = attention
        if attention:
            self.att = nn.Linear(d, 1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.use_attention:
            w = torch.softmax(self.att(out), dim=1)
            feat = (out * w).sum(dim=1)
        else:
            feat = out[:, -1, :]
        return self.fc(feat)


def class_weights(y, n_classes=3, power=0.5, max_weight=30.0):
    counts = np.bincount(y, minlength=n_classes)
    w = [min((counts.max() / c) ** power, max_weight) if c else 0.0
         for c in counts]
    return torch.tensor(w, dtype=torch.float32)
