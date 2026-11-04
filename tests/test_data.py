import unittest

from src.data import FEATURES, feature_options, load_data, split_data


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_data()

    def test_shape_and_columns(self):
        self.assertEqual(self.X.shape, (569, 10))
        self.assertEqual(list(self.X.columns), FEATURES)

    def test_malignant_is_the_positive_class(self):
        self.assertEqual(int(self.y.sum()), 212)

    def test_split_keeps_class_balance(self):
        X_train, X_test, y_train, y_test = split_data(self.X, self.y)
        self.assertEqual(len(X_train) + len(X_test), 569)
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), delta=0.02)

    def test_options_are_consistent(self):
        options = feature_options(self.X, self.y)
        for name in FEATURES:
            spec = options[name]
            self.assertLessEqual(spec["min"], spec["median"])
            self.assertLessEqual(spec["median"], spec["max"])
            self.assertLessEqual(spec["min"], spec["benign_median"])
            self.assertLessEqual(spec["malignant_median"], spec["max"])


if __name__ == "__main__":
    unittest.main()
