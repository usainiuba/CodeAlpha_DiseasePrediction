"""The four models we compare. Each one is a full pipeline."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import SEED

try:
    from xgboost import XGBClassifier
except Exception:  # xgboost missing or not loadable on this computer
    XGBClassifier = None


def build_models(seed=SEED):
    """Return {name: pipeline}. XGBoost is included only if it is installed."""
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1
            )
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = make_pipeline(
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
            )
        )
    return models
