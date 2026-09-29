import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import torch

from scripts.aubin_model import ALPHABET, encode_sequences
from scripts.baseline_models import MLPModel, load_baseline, sparse_one_hot
from scripts.train_baselines import fit_ridge, fit_knn, neighbor_predictions, metrics


class BaselineTests(unittest.TestCase):
    def test_sparse_encoding_matches_neural_features(self):
        encoded = encode_sequences(['AC', 'YA'], 2)
        sparse = sparse_one_hot(encoded.numpy())
        dense = torch.nn.functional.one_hot(encoded, len(ALPHABET)).float()
        dense = torch.nn.functional.pad(dense, (0, 0, 0, 1)).flatten(start_dim=1)
        np.testing.assert_array_equal(sparse.toarray(), dense.numpy())
        with self.assertRaises(ValueError):
            sparse_one_hot(np.array([[20]]))

    def test_sklearn_additive_fit_and_saved_sequence_predictions(self):
        sequences = ['AA', 'AC', 'CA', 'CC']
        y = np.array([0., 2., 1., 3.])
        x = sparse_one_hot(encode_sequences(sequences, 2).numpy())
        estimator = LinearRegression(tol=1e-8).fit(x, y)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.pkl'
            with path.open('wb') as stream:
                pickle.dump(dict(alphabet=ALPHABET, estimator=estimator, sequence_length=2), stream)
            model, _ = load_baseline(path)
            np.testing.assert_allclose(model.predict(sequences), y, atol=1e-12)
            with self.assertRaises(ValueError):
                model.predict(['AAA'])

    def test_mlp_checkpoint_roundtrip(self):
        for architecture in ['mlp_small', 'mlp_deep']:
            model = MLPModel(2, architecture)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/'model.pt'
                torch.save(dict(alphabet=ALPHABET, architecture=architecture, sequence_length=2,
                                state_dict=model.state_dict()), path)
                restored, _ = load_baseline(path)
                np.testing.assert_array_equal(model.predict(['AC', 'YA']), restored.predict(['AC', 'YA']))

    def test_ridge_selection_uses_validation_loss(self):
        x = np.array([[0.], [1.], [2.]])
        y = x.ravel()
        model, trials = fit_ridge(x, y, x, y, [0.1, 100.])
        self.assertEqual(model.alpha, 0.1)
        self.assertLess(trials[0]['validation_mse'], trials[1]['validation_mse'])
        # Alter validation targets only: strongest shrinkage should now win.
        model, _ = fit_ridge(x, y, x, np.ones(3), [0.1, 100.])
        self.assertEqual(model.alpha, 100.)

    def test_mean_predictor_has_undefined_rank_correlation(self):
        result = metrics(np.array([1., 2., 3.]), np.array([2., 2., 2.]))
        self.assertIsNone(result['spearman'])
        self.assertIsNone(result['pearson'])
        self.assertEqual(result['r2'], 0.)

    def test_pearson_is_not_r_squared(self):
        result = metrics(np.array([1., 2., 3.]), np.array([11., 12., 13.]))
        self.assertAlmostEqual(result['pearson'], 1.)
        self.assertLess(result['r2'], 0.)

    def test_knn_search_matches_sklearn_including_ties_and_exact_matches(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0., 1., 2., 3.])
        query = np.array([[0, 0], [2, 2]])
        truth = np.array([0., 1.5])
        model, trials = fit_knn(x, y, query, truth, [1, 3])
        for trial in trials:
            candidate = KNeighborsRegressor(n_neighbors=trial['k'], weights=trial['weights'],
                                             metric='hamming', algorithm='brute').fit(x, y)
            pred = candidate.predict(query)
            self.assertAlmostEqual(trial['validation_mse'], float(np.mean((truth-pred)**2)))
        best = min(trials, key=lambda r: r['validation_mse'])
        self.assertEqual((model.n_neighbors, model.weights), (best['k'], best['weights']))
        result = neighbor_predictions(np.array([[0., 0., 0.5]]), np.array([[0, 1, 2]]), y, 'distance')
        np.testing.assert_array_equal(result, [0.5])

    def test_knn_checkpoint_preserves_hamming_encoding(self):
        sequences = ['AA', 'AC', 'CA', 'CC']
        x = encode_sequences(sequences, 2).numpy()
        estimator = KNeighborsRegressor(n_neighbors=1, metric='hamming', algorithm='brute').fit(x, [0., 1., 2., 3.])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.pkl'
            with path.open('wb') as stream:
                pickle.dump(dict(alphabet=ALPHABET, estimator=estimator, sequence_length=2, encoding='hamming'), stream)
            model, _ = load_baseline(path)
            np.testing.assert_array_equal(model.predict(sequences), [0., 1., 2., 3.])


if __name__ == '__main__':
    unittest.main()
