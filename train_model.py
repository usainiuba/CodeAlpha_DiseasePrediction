"""Compare the models, test the best one once, and save everything.

Run with:  python train_model.py
"""
import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import classification_report

from predict import (
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    OPTIONS_PATH,
    REPORT_PATH,
)
from src.data import SEED, feature_options, load_data, split_data
from src.evaluate import cross_validate_model, evaluate_on_test, top_features
from src.models import build_models

THRESHOLD = 0.5


def main(models_dir=MODELS_DIR, folds=5, quiet=False):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    say = (lambda *a, **k: None) if quiet else print

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    say(f"Data: {len(X)} cases, {int(y.sum())} malignant. "
        f"Train {len(X_train)} / test {len(X_test)}.")

    cv_results, skipped = {}, {}
    for name, model in build_models().items():
        say(f"Cross-validating {name} ...")
        try:
            cv_results[name] = cross_validate_model(model, X_train, y_train, folds=folds)
        except Exception as error:  # one broken model must not stop the others
            skipped[name] = str(error)[:200]
            say(f"  skipped {name}: {skipped[name]}")

    selected = max(
        cv_results,
        key=lambda n: (cv_results[n]["roc_auc"]["mean"], cv_results[n]["recall"]["mean"]),
    )
    say(f"\nSelected model: {selected}")

    best = build_models()[selected]
    best.fit(X_train, y_train)
    test = evaluate_on_test(best, X_test, y_test, THRESHOLD)
    features = top_features(best, X_test, y_test)

    metrics = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "rows": int(len(X)),
            "malignant": int(y.sum()),
            "benign": int(len(y) - y.sum()),
            "features": len(X.columns),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "seed": SEED,
        "threshold": THRESHOLD,
        "folds": folds,
        "cv": cv_results,
        "skipped": skipped,
        "selected_model": selected,
        "test": test,
        "top_features": features,
        "versions": {"scikit_learn": sklearn.__version__, "python": platform.python_version()},
    }

    joblib.dump(best, models_dir / MODEL_PATH.name)
    (models_dir / METRICS_PATH.name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (models_dir / OPTIONS_PATH.name).write_text(
        json.dumps(feature_options(X, y), indent=2), encoding="utf-8"
    )
    pred = (best.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    report = classification_report(y_test, pred, target_names=["benign", "malignant"])
    (models_dir / REPORT_PATH.name).write_text(report, encoding="utf-8")

    say("\nCross-validation (ROC-AUC, mean):")
    for name, scores in cv_results.items():
        say(f"  {name:<20} {scores['roc_auc']['mean']:.3f} +/- {scores['roc_auc']['std']:.3f}")
    say(f"\nFinal test ({len(X_test)} unseen cases):")
    for key in ("roc_auc", "accuracy", "precision", "recall", "specificity", "f1"):
        say(f"  {key:<12} {test[key]:.3f}")
    say(f"  confusion matrix {test['confusion_matrix']}")
    say(f"\nSaved to {models_dir}")
    return metrics


if __name__ == "__main__":
    main()
