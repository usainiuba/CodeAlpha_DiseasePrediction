"""Metrics and feature importance."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED

SCORING = ["roc_auc", "accuracy", "precision", "recall", "f1"]


def cross_validate_model(model, X, y, folds=5, seed=SEED):
    """Mean and standard deviation of each metric across the folds."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
        }
        for metric in SCORING
    }


def evaluate_on_test(model, X_test, y_test, threshold=0.5):
    """Score a fitted model on data it has never seen."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": threshold,
    }


def top_features(model, X_test, y_test, seed=SEED, repeats=20):
    """How much ROC-AUC drops when one measurement is shuffled (bigger = more used)."""
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=repeats, random_state=seed, n_jobs=1,
    )
    ranked = sorted(
        zip(X_test.columns, result.importances_mean), key=lambda item: item[1], reverse=True
    )
    return [{"feature": name, "importance": float(value)} for name, value in ranked]
