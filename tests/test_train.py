import json
import tempfile
import unittest
from pathlib import Path

import train_model


class TrainSmokeTest(unittest.TestCase):
    def test_full_run_saves_everything(self):
        with tempfile.TemporaryDirectory() as folder:
            metrics = train_model.main(models_dir=folder, folds=3, quiet=True)
            for name in ("disease_model.joblib", "metrics.json",
                         "feature_options.json", "classification_report.txt"):
                self.assertTrue((Path(folder) / name).exists(), name)
            saved = json.loads((Path(folder) / "metrics.json").read_text())
            self.assertEqual(saved["selected_model"], metrics["selected_model"])
            self.assertGreater(metrics["test"]["accuracy"], 0.9)
            self.assertGreaterEqual(len(metrics["cv"]), 3)


if __name__ == "__main__":
    unittest.main()
