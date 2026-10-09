"""Diabetes risk screening on CDC BRFSS 2015 survey data.

Task: predict whether a survey respondent reports diabetes (vs. no diabetes or
pre-diabetes) from self-reported health and lifestyle indicators.

Run:  python train.py [path/to/csv]
Outputs: results/metrics.json, figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET_RECALL = 0.80  # screening goal: catch 80% of true diabetes cases
ROOT = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"

# Easy-to-collect subset, to test whether a short questionnaire is enough.
SHORT_FEATURES = ["Age", "BMI", "HighBP", "HighChol", "GenHlth"]

AGE_LABELS = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
              6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
              11: "70-74", 12: "75-79", 13: "80+"}

# Chart styling: one hue per job, thin marks, quiet grid.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#cfcec8", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10, "savefig.dpi": 160,
})


def load():
    df = pd.read_csv(DATA)
    n_raw = len(df)
    # Identical rows would otherwise land in both train and test and inflate scores.
    df = df.drop_duplicates().reset_index(drop=True)
    df["diabetes"] = (df["Diabetes_012"] == 2).astype(int)
    X = df.drop(columns=["Diabetes_012", "diabetes"])
    return X, df["diabetes"], n_raw, len(df)


def threshold_for_recall(y, p, target):
    """Highest threshold that still reaches the target recall on (y, p)."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]])


def operating_point(y, p, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": round(thr, 4),
        "recall": round(tp / (tp + fn), 4),
        "precision": round(tp / (tp + fp), 4),
        "specificity": round(tn / (tn + fp), 4),
        "flagged_share": round(pred.mean(), 4),
    }


def main():
    X, y, n_raw, n_clean = load()
    print(f"rows: {n_raw} raw -> {n_clean} after removing duplicates")
    print(f"diabetes prevalence: {y.mean():.3%}")

    # 60 / 20 / 20 stratified split; threshold is tuned on validation only.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    models = {
        "Logistic regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            list(X.columns)),
        "Gradient boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=24, l2_regularization=1.0,
                                           early_stopping=True, random_state=SEED),
            list(X.columns)),
        "Gradient boosting (5 features)": (
            HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=24, l2_regularization=1.0,
                                           early_stopping=True, random_state=SEED),
            SHORT_FEATURES),
    }

    results, curves, fitted = {}, {}, {}
    for name, (model, cols) in models.items():
        model.fit(X_tr[cols], y_tr)
        p_va = model.predict_proba(X_va[cols])[:, 1]
        p_te = model.predict_proba(X_te[cols])[:, 1]
        thr = threshold_for_recall(y_va, p_va, TARGET_RECALL)
        results[name] = {
            "roc_auc": round(roc_auc_score(y_te, p_te), 4),
            "pr_auc": round(average_precision_score(y_te, p_te), 4),
            "at_80pct_recall": operating_point(y_te.values, p_te, thr),
        }
        curves[name] = p_te
        fitted[name] = (model, cols)
        r = results[name]
        print(f"{name:32s} ROC-AUC {r['roc_auc']:.3f} | PR-AUC {r['pr_auc']:.3f} | "
              f"precision@recall~0.8 {r['at_80pct_recall']['precision']:.3f}")

    base_rate = float(y_te.mean())
    results["_meta"] = {
        "rows_raw": n_raw, "rows_after_dedup": n_clean,
        "prevalence": round(float(y.mean()), 4), "test_rows": int(len(y_te)),
        "test_prevalence": round(base_rate, 4), "seed": SEED,
        "target_recall": TARGET_RECALL, "short_features": SHORT_FEATURES,
    }

    # Permutation importance on the full-feature boosting model (test subsample for speed).
    model, cols = fitted["Gradient boosting"]
    idx = X_te.sample(n=min(25000, len(X_te)), random_state=SEED).index
    pi = permutation_importance(model, X_te.loc[idx, cols], y_te.loc[idx], scoring="roc_auc",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    results["_meta"]["top_features_auc_drop"] = {k: round(float(v), 4) for k, v in imp.head(8).items()}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2))

    # ---- Figure 1: prevalence by age group ----
    prev = pd.concat([X["Age"], y], axis=1).groupby("Age")["diabetes"].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar([AGE_LABELS[a] for a in prev.index], prev.values, color=BLUE, width=0.6)
    ax.set_ylabel("Respondents reporting diabetes (%)")
    ax.set_xlabel("Age group")
    ax.set_title("Diabetes prevalence climbs with age and peaks at 70-74", loc="left", fontsize=12)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "prevalence_by_age.png")
    plt.close(fig)

    # ---- Figure 2: ROC and precision-recall (full-feature vs 5-feature vs logistic) ----
    colors = {"Logistic regression": GRAY, "Gradient boosting": BLUE, "Gradient boosting (5 features)": ORANGE}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        a1.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
        a2.plot(rc[:-1], pr[:-1], color=colors[name], lw=2, label=f"{name} (AP {results[name]['pr_auc']:.3f})")
    a1.plot([0, 1], [0, 1], color="#bdbcb5", lw=1, ls="--")
    a2.axhline(base_rate, color="#bdbcb5", lw=1, ls="--")
    a1.set(xlabel="False positive rate", ylabel="True positive rate")
    a2.set(xlabel="Recall", ylabel="Precision")
    a1.set_title("ROC curve", loc="left")
    a2.set_title("Precision-recall curve", loc="left")
    a2.text(0.98, base_rate + 0.015, f"base rate {base_rate:.1%}", ha="right", color=INK2, fontsize=8)
    for a in (a1, a2):
        a.legend(frameon=False, fontsize=8, loc="lower right" if a is a1 else "upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "roc_pr_curves.png")
    plt.close(fig)

    # ---- Figure 3: permutation importance ----
    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top.index, top.values, color=BLUE, height=0.6)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("What the model relies on most", loc="left", fontsize=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "feature_importance.png")
    plt.close(fig)

    print("done: results/metrics.json and figures/ written")


if __name__ == "__main__":
    main()
