from sklearn.metrics import f1_score, precision_score, recall_score

from .metrics import event_metrics


def report(y_true, y_pred, meta, title="model"):
    p = precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    r = recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    f = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ev = event_metrics(y_true, y_pred, meta)
    print(f"=== {title} ===")
    print(f"point (class 1): precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    print(f"macro F1: {macro:.3f}")
    print(f"events: TP={ev['tp']} FP={ev['fp']} FN={ev['fn']} "
          f"precision={ev['precision']:.3f} recall={ev['recall']:.3f} "
          f"F1={ev['f1']:.3f}")
    return {"point_f1": f, "event_f1": ev["f1"]}
