from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from scripts.cross_validate import build_folds
from scripts.cv_neighbors import make_distance_cache, query_cache
from scripts.train_baselines import neighbor_predictions


class CrossValidationTests(unittest.TestCase):
    def test_every_record_held_out_exactly_once_and_validation_disjoint(self):
        folds = build_folds(103)
        seen = []
        for split in folds:
            train, valid, test = [set(split[k]) for k in ['train','validation','test']]
            self.assertFalse(train & valid or train & test or valid & test)
            self.assertEqual(train | valid | test, set(range(103)))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(103)))
        for first, second in zip(folds, build_folds(103)):
            for split in first:
                np.testing.assert_array_equal(first[split], second[split])

    def test_cached_distances_and_tied_neighbors_match_sklearn(self):
        x = np.array([[0,0,0], [0,1,0], [1,0,0], [1,1,0], [1,1,1], [2,2,2]])
        y = np.array([0.,1.,2.,3.,4.,5.])
        with tempfile.TemporaryDirectory() as temporary:
            cache = make_distance_cache(x, Path(temporary)/'counts.npy', chunk_size=2)
            expected = np.count_nonzero(x[:,None,:] != x[None,:,:], axis=2)
            np.testing.assert_array_equal(cache, expected)
            train_ids = np.array([0,1,2,3,4])
            query_ids = np.array([0,5])
            for k in [1,3,5]:
                for weights in ['uniform','distance']:
                    model = KNeighborsRegressor(n_neighbors=k, weights=weights, metric='hamming', algorithm='brute').fit(x[train_ids],y[train_ids])
                    distances, indices = query_cache(cache, query_ids, train_ids, k, x.shape[1], batch_size=1)
                    pred = neighbor_predictions(distances, indices, y[train_ids], weights)
                    np.testing.assert_allclose(pred, model.predict(x[query_ids]), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
