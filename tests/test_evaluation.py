import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import generate_student_dataset
from src.features import get_feature_matrix
from src.train import find_optimal_threshold
from src.explain import explain_features
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class EvaluationTests(unittest.TestCase):
    def test_test_labels_do_not_choose_model_or_threshold(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from src.train import train_and_compare
        data = generate_student_dataset(300)

        def run(invert_test):
            calls = 0
            def split(*args, **kwargs):
                nonlocal calls
                calls += 1
                values = list(train_test_split(*args, **kwargs))
                if calls == 1 and invert_test:
                    values[3] = 1 - values[3]
                return values
            with tempfile.TemporaryDirectory() as directory:
                with patch('src.train.load_or_create_data', return_value=data), patch('src.train.train_test_split', side_effect=split), contextlib.redirect_stdout(io.StringIO()):
                    train_and_compare(directory)
                return json.loads((Path(directory) / 'models/model_metadata.json').read_text())

        original, changed = run(False), run(True)
        self.assertEqual(original['best_model_name'], changed['best_model_name'])
        self.assertEqual(original['optimal_threshold'], changed['optimal_threshold'])
        self.assertEqual(original['validation_metrics'], changed['validation_metrics'])
        self.assertNotEqual(original['tuned_metrics']['roc_auc'], changed['tuned_metrics']['roc_auc'])

    def test_threshold_meets_validation_recall(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([.1, .4, .3, .8])
        threshold = find_optimal_threshold(y, prob, min_recall=1)
        self.assertEqual(int((prob[y == 1] >= threshold).sum()), 2)

    def test_shap_reconstructs_selected_model_log_odds(self):
        X, y = get_feature_matrix(generate_student_dataset(250))
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
        explanation, space = explain_features(model, X, X.iloc[:5])
        self.assertEqual(space, 'log_odds')
        np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), model.decision_function(X.iloc[:5]), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
