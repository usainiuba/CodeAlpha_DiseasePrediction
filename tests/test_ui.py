import unittest

from ui import components as c

METRICS = {
    "selected_model": "<b>Evil</b>",
    "dataset": {"rows": 569, "malignant": 212},
    "test": {
        "roc_auc": 0.99, "accuracy": 0.96, "precision": 0.95, "recall": 0.93,
        "specificity": 0.98, "f1": 0.94, "confusion_matrix": [[70, 2], [3, 39]],
    },
    "cv": {"SVM": {k: {"mean": 0.99, "std": 0.01} for k in ("roc_auc", "recall", "precision", "f1")}},
    "top_features": [{"feature": "area", "importance": 0.05}, {"feature": "concave_points", "importance": 0.02}],
}
PREDICTION = {"malignant": 0.8, "benign": 0.2,
              "band": {"tone": "high", "label": "Leans malignant", "message": "msg"}}


class UiTests(unittest.TestCase):
    def all_html(self):
        return [
            c.hero_html(METRICS), c.disclaimer_html(), c.steps_html(), c.empty_state_html(),
            c.result_html(PREDICTION),
            c.position_html([{"feature": "area", "value_text": "900", "percent": 60.0}]),
            c.models_html(METRICS["cv"], "SVM", {"XGBoost": "not installed"}),
            c.tiles_html(METRICS["test"]), c.confusion_html(METRICS["test"]),
            c.factor_bars_html(METRICS["top_features"]), c.about_html(METRICS), c.footer_html(),
        ]

    def test_no_line_breaks_or_indentation(self):
        for html in self.all_html():
            self.assertNotIn("\n", html)

    def test_text_is_escaped(self):
        self.assertNotIn("<b>Evil</b>", c.hero_html(METRICS))
        self.assertIn("&lt;b&gt;Evil", c.hero_html(METRICS))

    def test_result_shows_percent_and_tone(self):
        html = c.result_html(PREDICTION)
        self.assertIn("80.0%", html)
        self.assertIn("tone-high", html)

    def test_empty_position_rows(self):
        self.assertEqual(c.position_html([]), "")


if __name__ == "__main__":
    unittest.main()
