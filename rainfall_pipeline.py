#!/usr/bin/env python3
"""
Next-day rainfall (mm) prediction for Indian weather stations.

Run:  python rainfall_pipeline.py --train train_cleaned.csv --test test_cleaned.csv \
                                  --sample sample_submission.csv --outdir out

Validation: expanding-window, 4 yearly folds (train on years < Y, validate on Y,
Y = 2020..2023); the reported RMSE is the average over the 4 folds.

Only same-day information is used as features (no lags, no next-day rows), because the
test file contains neither previous-day nor next-day rows.
"""
import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score

from features import load, build_features

warnings.filterwarnings("ignore")
FOLDS = [2020, 2021, 2022, 2023]
N_SEEDS = 5
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def make_weights(d, half_life_years, monsoon_w):
    w = np.ones(len(d))
    if half_life_years:
        age = (d.date.max() - d.date).dt.days.values
        w = 0.5 ** (age / (365.25 * half_life_years))
    w = w * np.where(d.is_monsoon.values == 1, monsoon_w, 1.0)
    return w / w.mean()


# ----------------------------------------------------------------------------- models
def fit_predict_gbm(trn, tst, feats, w, seed, target="raw"):
    """Histogram gradient boosting (XGBoost-style). `seed` changes the column subsample."""
    y = trn.rainfall_next_day_mm.values
    yt = np.log1p(y) if target == "log" else y
    m = HGBR(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
             l2_regularization=1.0, max_features=0.8, random_state=seed)
    m.fit(trn[feats], yt, sample_weight=w)
    p = m.predict(tst[feats])
    if target == "log":
        # smearing correction for back-transformation bias (estimated on training residuals)
        resid = yt - m.predict(trn[feats])
        p = np.expm1(p + 0.5 * np.var(resid))
    return np.clip(p, 0, None)


def fit_predict_clf50(trn, tst, feats, w, seed=0):
    y = (trn.rainfall_next_day_mm.values > 50).astype(int)
    m = HGBC(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
             l2_regularization=1.0, max_features=0.8, random_state=seed)
    m.fit(trn[feats], y, sample_weight=w)
    return m.predict_proba(tst[feats])[:, 1]


def fit_predict_nn(trn, tst, feats, w, seed):
    """Small MLP (64-32) on quantile-normalised features."""
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal",
                             subsample=100_000, random_state=seed)
    Xtr = np.nan_to_num(qt.fit_transform(trn[feats].values.astype(np.float64)), nan=0.0)
    Xte = np.nan_to_num(qt.transform(tst[feats].values.astype(np.float64)), nan=0.0)
    scale = 15.0
    m = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                     batch_size=512, learning_rate_init=1e-3, max_iter=30,
                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=4,
                     random_state=seed)
    m.fit(Xtr, trn.rainfall_next_day_mm.values / scale, sample_weight=w)
    return np.clip(m.predict(Xte) * scale, 0, None)


# ----------------------------------------------------------------------------- helpers
def cached(path, fn):
    if os.path.exists(path):
        return np.load(path)
    out = fn()
    np.save(path, out)
    return out


def heavy_rmse(y, p, thr=50):
    h = y > thr
    return rmse(y[h], p[h]) if h.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_cleaned.csv")
    ap.add_argument("--test", default="test_cleaned.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--half_life", type=float, default=None, help="recency half-life in years (0=none)")
    ap.add_argument("--monsoon_w", type=float, default=None, help="weight multiplier for Jun-Sep")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cdir = os.path.join(args.outdir, "cache")
    os.makedirs(cdir, exist_ok=True)

    bw_path = os.path.join(args.outdir, "best_weights.json")
    bw = json.load(open(bw_path)) if os.path.exists(bw_path) else {"half": 0, "mw": 1.0}
    half = bw["half"] if args.half_life is None else args.half_life
    mw = bw["mw"] if args.monsoon_w is None else args.monsoon_w
    log(f"sample weights: recency half-life={half} yr, monsoon multiplier={mw}")

    log("building features (neighbours at 50/100/250 km, sector + upwind proxy)")
    df = load(args.train, args.test)
    df, feats = build_features(df)
    lab = df[df.is_test == 0].reset_index(drop=True)
    test = df[df.is_test == 1].reset_index(drop=True)
    log(f"labelled rows {len(lab):,}  test rows {len(test):,}  features {len(feats)}")

    model_names = ["gbm"] + ["nn"]
    # ------------------------------------------------------------------ 4-year validation
    oof = {k: [] for k in ["gbm", "nn", "log", "p50"]}
    ys = []
    seed_scores = {"gbm": [], "nn": []}
    for yr in FOLDS:
        trn, val = lab[lab.year < yr], lab[lab.year == yr]
        w = make_weights(trn, half, mw)
        y = val.rainfall_next_day_mm.values
        ys.append(y)
        log(f"=== fold {yr}: train {len(trn):,} rows, validate {len(val):,} rows")
        for name, fn in [("gbm", fit_predict_gbm), ("nn", fit_predict_nn)]:
            preds = []
            for s in range(args.seeds):
                p = cached(f"{cdir}/{name}_{yr}_s{s}.npy", lambda: fn(trn, val, feats, w, s))
                preds.append(p)
            oof[name].append(np.mean(preds, axis=0))
            log(f"   {name:4s} 5-seed avg RMSE {rmse(y, oof[name][-1]):.3f}  (single seeds: "
                + ", ".join(f"{rmse(y, q):.2f}" for q in preds) + ")")
        oof["log"].append(cached(f"{cdir}/log_{yr}.npy",
                                 lambda: fit_predict_gbm(trn, val, feats, w, 0, target="log")))
        log(f"   log  RMSE {rmse(y, oof['log'][-1]):.3f}")
        oof["p50"].append(cached(f"{cdir}/p50_{yr}.npy",
                                 lambda: fit_predict_clf50(trn, val, feats, w)))
        yb = (y > 50).astype(int)
        auc = roc_auc_score(yb, oof["p50"][-1]) if yb.min() != yb.max() else float("nan")
        log(f"   P(>50mm) classifier AUC {auc:.3f}")

    # ------------------------------------------------------------------ blending
    cols = ["gbm", "nn", "log", "p50"]

    def design(i):
        return np.column_stack([oof[c][i] for c in cols])

    def fit_blend(idx):
        A = np.vstack([design(i) for i in idx])
        A = np.column_stack([A, np.ones(len(A))])          # intercept (clipped at >=0 afterwards)
        b = np.concatenate([ys[i] for i in idx])
        # non-negative weights on models, free-sign intercept via +/- columns
        A2 = np.column_stack([A[:, :-1], A[:, -1], -A[:, -1]])
        coef, _ = nnls(A2, b)
        return np.append(coef[:len(cols)], coef[len(cols)] - coef[len(cols) + 1])

    def apply_blend(coef, X):
        return np.clip(X @ coef[:-1] + coef[-1], 0, None)

    blend_oof = []
    for i in range(len(FOLDS)):
        # blend weights come from the DENSE years (2022, 2023) only, like the 2024-25 test set;
        # the fold being scored is always excluded (honest out-of-fold blend)
        others = [j for j in (2, 3) if j != i]
        blend_oof.append(apply_blend(fit_blend(others), design(i)))
    coef_final = fit_blend([2, 3])
    log("blend weights (gbm, nn, log-gbm, P(>50)->mm, intercept): " + np.array2string(coef_final, precision=3))

    # ------------------------------------------------------------------ validation report
    rows = []
    for name, series in [("GBM (5 seeds)", oof["gbm"]), ("Neural net (5 seeds)", oof["nn"]),
                         ("GBM on log1p(rain)", oof["log"]), ("BLEND (out-of-fold weights)", blend_oof)]:
        per = [rmse(ys[i], series[i]) for i in range(4)]
        hv = [heavy_rmse(ys[i], series[i]) for i in range(4)]
        rows.append((name, per, float(np.mean(per)), float(np.nanmean(hv))))
    base_per = [rmse(ys[i], lab[lab.year < FOLDS[i]].rainfall_next_day_mm.mean()) for i in range(4)]
    rows.insert(0, ("Mean-only baseline", base_per, float(np.mean(base_per)), float("nan")))

    yall = np.concatenate(ys)
    pall = np.concatenate(blend_oof)
    pall23 = blend_oof[-1]
    y23 = ys[-1]

    print("\n" + "=" * 86)
    print("VALIDATION (expanding window, validate on 2020, 2021, 2022, 2023)")
    print("=" * 86)
    print(f"{'model':32s}" + "".join(f"{y:>9d}" for y in FOLDS) + f"{'AVG RMSE':>11s}{'heavy>50mm':>12s}")
    for name, per, avg, hv in rows:
        print(f"{name:32s}" + "".join(f"{v:9.3f}" for v in per) + f"{avg:11.3f}" + (f"{hv:12.2f}" if hv == hv else f"{'-':>12s}"))
    print("-" * 86)
    r2 = 1 - np.sum((yall - pall) ** 2) / np.sum((yall - yall.mean()) ** 2)
    r2_23 = 1 - np.sum((y23 - pall23) ** 2) / np.sum((y23 - y23.mean()) ** 2)
    mae = np.mean(np.abs(yall - pall))
    rain_acc = np.mean((pall >= 1) == (yall >= 1))
    rain_acc23 = np.mean((pall23 >= 1) == (y23 >= 1))
    heavy_acc = np.mean((pall > 50) == (yall > 50))
    print(f"Blend, all 4 folds : RMSE {rmse(yall, pall):.3f} | MAE {mae:.3f} | R2 {r2:.3f}")
    print(f"Blend, 2023 fold   : RMSE {rmse(y23, pall23):.3f} | R2 {r2_23:.3f}")
    print(f"Rain / no-rain accuracy (>=1 mm), 4 folds: {100 * rain_acc:.1f}%   (2023 only: {100 * rain_acc23:.1f}%)")
    print(f"Heavy-rain (>50 mm) yes/no accuracy: {100 * heavy_acc:.1f}%   |  RMSE on heavy days (blend): {heavy_rmse(yall, pall):.2f} mm")
    print("NOTE: rainfall in mm is a regression target, so 'accuracy' is not defined for it; RMSE / R2 are the")
    print("      honest measures. R2 is the fraction of variance explained (the mean-only baseline has R2 = 0).")
    print("=" * 86 + "\n")

    # ------------------------------------------------------------------ final fit + test predictions
    log("final fit on all labelled data (2015-2023) -> test predictions")
    w = make_weights(lab, half, mw)
    tp = {}
    for name, fn in [("gbm", fit_predict_gbm), ("nn", fit_predict_nn)]:
        ps = [cached(f"{cdir}/final_{name}_s{s}.npy", lambda: fn(lab, test, feats, w, s)) for s in range(args.seeds)]
        tp[name] = np.mean(ps, axis=0)
        log(f"   final {name} done")
    tp["log"] = cached(f"{cdir}/final_log.npy", lambda: fit_predict_gbm(lab, test, feats, w, 0, target="log"))
    tp["p50"] = cached(f"{cdir}/final_p50.npy", lambda: fit_predict_clf50(lab, test, feats, w))
    X = np.column_stack([tp[c] for c in cols])
    pred = np.round(apply_blend(coef_final, X), 2)

    sample = pd.read_csv(args.sample)
    assert len(sample) == len(test) and (sample["id"].values == test["id"].astype(int).values).all(), "id mismatch"
    sub = pd.DataFrame({"id": test["id"].astype(int).values, "rainfall_next_day_mm": pred})
    sub.to_csv(os.path.join(args.outdir, "submission.csv"), index=False)

    orig_test = pd.read_csv(args.test)
    full = orig_test.copy()
    full["rainfall_next_day_mm_predicted"] = pred
    full.to_csv(os.path.join(args.outdir, "predictions_with_data.csv"), index=False)
    log(f"saved {args.outdir}/submission.csv (sample_submission format) and predictions_with_data.csv")
    print("Predicted test rainfall: mean %.2f mm, max %.1f mm, share >=1 mm: %.1f%%" %
          (pred.mean(), pred.max(), 100 * np.mean(pred >= 1)))


if __name__ == "__main__":
    main()
