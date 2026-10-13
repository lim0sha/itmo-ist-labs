from .models.train import predict_proba, train_model
from .prepare import prepare
from .utils.metrics import event_metrics
from .utils.postprocess import postprocess
from .utils.report import report
from .utils.tuning import tune_postprocess


def run_experiment(df, title, feature_set, seq_len, norm, bidir, attention,
                   val_veh, cfg, seed=42):
    cfg = {**cfg, "bidir": bidir, "attention": attention}
    (Xtr, ytr, _, Xva, yva, mva, Xte, yte, mte) = prepare(
        df, feature_set, seq_len, norm, val_veh)
    model = train_model(Xtr, ytr, Xva, yva, cfg, seed=seed)

    raw = event_metrics(yte, predict_proba(model, Xte).argmax(1), mte)
    va_prob = predict_proba(model, Xva)[:, 1]
    te_prob = predict_proba(model, Xte)[:, 1]
    thr, ml, mg, val_f1 = tune_postprocess(va_prob, yva, mva)
    y_pred = postprocess(te_prob, mte, thr, ml, mg)

    res = report(yte, y_pred, mte, title)
    print(f"chosen on val: thr={thr:.2f} min_len={ml} max_gap={mg} "
          f"val_event_f1={val_f1:.3f} | raw_argmax_test_event_f1={raw['f1']:.3f}")
    res.update(raw_event_f1=raw["f1"], val_event_f1=val_f1,
               threshold=thr, min_len=ml, max_gap=mg)
    return res
