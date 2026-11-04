"""Loading, cleaning and splitting the breast cancer data.

The data ships inside scikit-learn, so nothing is downloaded and no account
is needed.
"""
import math

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42

# Only the ten "mean" measurements are used, so that a person can type them in.
FEATURE_MAP = {
    "mean radius": "radius",
    "mean texture": "texture",
    "mean perimeter": "perimeter",
    "mean area": "area",
    "mean smoothness": "smoothness",
    "mean compactness": "compactness",
    "mean concavity": "concavity",
    "mean concave points": "concave_points",
    "mean symmetry": "symmetry",
    "mean fractal dimension": "fractal_dimension",
}
FEATURES = list(FEATURE_MAP.values())

# Slider step and number of decimals shown for each measurement.
STEPS = {
    "radius": 0.1,
    "texture": 0.1,
    "perimeter": 0.5,
    "area": 5.0,
    "smoothness": 0.001,
    "compactness": 0.001,
    "concavity": 0.001,
    "concave_points": 0.001,
    "symmetry": 0.001,
    "fractal_dimension": 0.0005,
}
DECIMALS = {
    "radius": 1,
    "texture": 1,
    "perimeter": 1,
    "area": 0,
    "smoothness": 3,
    "compactness": 3,
    "concavity": 3,
    "concave_points": 3,
    "symmetry": 3,
    "fractal_dimension": 4,
}


def load_data():
    """Return X (ten measurements) and y (1 = malignant, 0 = benign)."""
    raw = load_breast_cancer(as_frame=True)
    X = raw.frame[list(FEATURE_MAP)].rename(columns=FEATURE_MAP)
    # In scikit-learn 0 means malignant and 1 means benign. We flip it so that
    # 1 = malignant, because malignant is the case we want to catch.
    y = (1 - raw.target).astype(int).rename("malignant")
    return X, y


def split_data(X, y, test_size=0.2, seed=SEED):
    """Stratified split, so train and test keep the same share of malignant cases."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _snap(value, step):
    """Round a value to the slider grid."""
    return round(round(float(value) / step) * step, 6)


def feature_options(X, y):
    """Slider ranges and typical values for the app, taken from the data."""
    options = {}
    for name in FEATURES:
        step = STEPS[name]
        column = X[name]
        options[name] = {
            "min": round(math.floor(float(column.min()) / step) * step, 6),
            "max": round(math.ceil(float(column.max()) / step) * step, 6),
            "step": step,
            "decimals": DECIMALS[name],
            "median": _snap(column.median(), step),
            "benign_median": _snap(column[y == 0].median(), step),
            "malignant_median": _snap(column[y == 1].median(), step),
        }
    return options
