import unittest

from predict import position_rows, predict_one, risk_band
from src.data import FEATURES, feature_options, load_data, split_data
from src.models import build_models


class PredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = load_data()
        X_train, _, y_train, _ = split_data(X, y)
        cls.model = build_models()["Logistic Regression"].fit(X_train, y_train)
        cls.options = feature_options(X, y)

    def test_bands(self):
        self.assertEqual(risk_band(0.05)["tone"], "low")
        self.assertEqual(risk_band(0.50)["tone"], "mid")
        self.assertEqual(risk_band(0.95)["tone"], "high")

    def test_typical_profiles_point_the_right_way(self):
        benign = {f: self.options[f]["benign_median"] for f in FEATURES}
        malignant = {f: self.options[f]["malignant_median"] for f in FEATURES}
        self.assertLess(predict_one(self.model, benign)["malignant"], 0.3)
        self.assertGreater(predict_one(self.model, malignant)["malignant"], 0.7)

    def test_probabilities_add_up(self):
        values = {f: self.options[f]["median"] for f in FEATURES}
        result = predict_one(self.model, values)
        self.assertAlmostEqual(result["benign"] + result["malignant"], 1.0)

    def test_position_rows(self):
        values = {f: self.options[f]["malignant_median"] for f in FEATURES}
        top = [{"feature": f, "importance": 1.0} for f in FEATURES]
        rows = position_rows(values, self.options, top, count=3)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertAlmostEqual(row["percent"], 75.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
