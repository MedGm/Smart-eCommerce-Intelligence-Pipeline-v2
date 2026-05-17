"""
XGBoost classifier: comparison model for high-potential product prediction.
Same label as RandomForest (top 20% by heuristic score).
Dossier: xgboost for predicting product success.

NOTE: The 'score' and 'popularity_proxy' columns are excluded from features
to avoid circular data leakage.
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from src.config import analytics_dir, get_logger, models_dir
from src.ml.utils import (
    build_high_potential_target,
    get_feature_columns,
    honesty_gate,
    label_integrity_diagnostics,
    load_features,
    optimize_f1_threshold,
)

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def run():
    if not HAS_XGBOOST:
        logger.warning("XGBoost not installed. pip install xgboost to enable.")
        return

    out_dir = analytics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features()
    if df.empty or len(df) < 20:
        logger.warning("Not enough data for XGBoost training (%d rows).", len(df))
        return

    target_origin = "external_observed"
    df["high_potential"], target_meta = build_high_potential_target(df)
    if df["high_potential"].nunique() < 2:
        logger.warning("Target has a single class after construction. Skipping XGBoost training.")
        return

    # Exclude score-derived and target-defining columns to reduce circularity.
    features = get_feature_columns(
        df,
        exclude_score=True,
        exclude_columns=target_meta.get("label_driver_features", []),
    )
    if not features:
        logger.warning("No numeric features found.")
        return

    X = df[features].fillna(0)
    y = df["high_potential"]
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    scale_pos_weight = float(neg / pos)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    cv = StratifiedKFold(
        n_splits=min(5, max(2, y.value_counts().min())),
        shuffle=True,
        random_state=42,
    )
    y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_threshold = optimize_f1_threshold(y, y_proba)
    y_pred = (y_proba >= cv_threshold["best"]["threshold"]).astype(int)

    metrics = {
        "model": "XGBoost",
        "method": "cross_validation",
        "n_samples": len(df),
        "n_features": len(features),
        "features": features,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "calibration": cv_threshold,
        "scale_pos_weight": scale_pos_weight,
        "grouped_cv_status": "not_run",
    }

    if "shop_name" in df.columns:
        shop_target = (
            df.assign(_y=y.astype(int))
            .groupby("shop_name")["_y"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "positives", "count": "rows"})
            .reset_index()
        )
        total_pos = int(shop_target["positives"].sum())
        shops_with_pos = int((shop_target["positives"] > 0).sum())
        max_share = float(shop_target["positives"].max() / total_pos) if total_pos > 0 else 0.0
        metrics["grouped_cv_feasibility"] = {
            "total_shops": int(shop_target["shop_name"].nunique()),
            "shops_with_positive_labels": shops_with_pos,
            "max_positive_share_single_shop": max_share,
            "high_risk": bool(shops_with_pos < 3 or max_share > 0.8),
            "notes": (
                "Positive labels are concentrated in few shops; grouped CV may understate generalization."
                if (shops_with_pos < 3 or max_share > 0.8)
                else "Positive labels are sufficiently distributed across shops."
            ),
            "by_shop": shop_target.sort_values("positives", ascending=False).to_dict("records"),
        }

    # Leakage-resistant evaluation: group folds by shop.
    if "shop_name" in df.columns and df["shop_name"].nunique() >= 2:
        groups = df["shop_name"].astype(str)
        group_cv = GroupKFold(n_splits=min(5, int(groups.nunique())))
        try:
            y_proba_group = cross_val_predict(
                clf,
                X,
                y,
                cv=group_cv,
                groups=groups,
                method="predict_proba",
            )[:, 1]
            grouped_threshold = optimize_f1_threshold(y, y_proba_group)
            y_pred_group = (y_proba_group >= grouped_threshold["best"]["threshold"]).astype(int)
            metrics["grouped_cv"] = {
                "n_splits": int(min(5, int(groups.nunique()))),
                "accuracy": float(accuracy_score(y, y_pred_group)),
                "precision": float(precision_score(y, y_pred_group, zero_division=0)),
                "recall": float(recall_score(y, y_pred_group, zero_division=0)),
                "f1": float(f1_score(y, y_pred_group, zero_division=0)),
                "confusion_matrix": confusion_matrix(y, y_pred_group).tolist(),
                "group_key": "shop_name",
                "calibration": grouped_threshold,
            }
            metrics["method"] = "cross_validation+grouped_cv"
            metrics["grouped_cv_status"] = "ok"
        except Exception as e:
            metrics["grouped_cv"] = {"error": str(e), "group_key": "shop_name"}
            metrics["grouped_cv_status"] = "error"
    else:
        metrics["grouped_cv_status"] = "skipped"

    diagnostics = label_integrity_diagnostics(X, y, clf, cv=cv, random_state=42)
    honesty = honesty_gate(
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        majority_baseline=diagnostics["majority_baseline_accuracy"],
        shuffled_accuracy=diagnostics["shuffled_label_accuracy"],
        target_origin=target_origin,
    )

    metrics["target_origin"] = target_origin
    metrics["target_definition"] = target_meta
    metrics["label_integrity"] = diagnostics
    metrics["honesty_gate"] = honesty

    if diagnostics["direct_leakage_check"] == "fail":
        logger.warning(
            "Potential leakage risk: shuffled-label accuracy %.3f exceeds majority baseline %.3f",
            diagnostics["shuffled_label_accuracy"],
            diagnostics["majority_baseline_accuracy"],
        )

    # Fit on full data, then persist model and predictions
    clf.fit(X, y)

    # Persist model
    m_dir = models_dir()
    m_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, m_dir / "xgboost.joblib")

    # Write per-product predictions
    proba_full = clf.predict_proba(X)[:, 1]
    preds_df = (
        df[["product_id"]].copy() if "product_id" in df.columns else pd.DataFrame(index=df.index)
    )
    preds_df["xgb_proba"] = proba_full
    preds_df.to_csv(out_dir / "xgb_predictions.csv", index=False)

    with open(out_dir / "model_metrics_xgboost.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        "XGBoost trained. accuracy=%.3f f1=%.3f direct_leakage=%s honesty=%s trust=%d -> analytics/model_metrics_xgboost.json",
        metrics["accuracy"],
        metrics["f1"],
        metrics["label_integrity"]["direct_leakage_check"],
        metrics["honesty_gate"]["status"],
        metrics["honesty_gate"]["trust_score"],
    )
    return clf, metrics


if __name__ == "__main__":
    run()
