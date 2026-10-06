"""Scientific invariants of active learning: budgets, features and disjoint data."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.active_learning.acquisition import acquisition_scores, cluster_budgets, diverse_sequences, select_queries
from scripts.active_learning.run import Predictor, infer, split_indices
from scripts.active_learning import run as runner


class ActiveLearningTests(unittest.TestCase):
    def test_common_split_is_disjoint_and_reproducible(self):
        a, b = split_indices(24516, 42), split_indices(24516, 42)
        self.assertEqual(sum(map(len, a.values())), 24516)
        self.assertEqual(len(set(np.concatenate(list(a.values())))), 24516)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(len(a['train']), 245)
        self.assertEqual(len(a['validation']), 2451)
        self.assertEqual(len(a['pool']), 2452)
        self.assertEqual(len(a['test']), 4904)

    def test_uncertainty_free_score_is_distance_only(self):
        score = acquisition_scores([[0.]], [[1.], [2.], [3.]], np.zeros(3))
        np.testing.assert_allclose(score, [0., .19, .38])
        self.assertTrue(np.isfinite(acquisition_scores([[1.]], [[1.], [1.]], [0., 0.])).all())

    def test_capacity_and_small_budgets(self):
        assignments = np.array([0] + [1]*9)
        for budget in range(11):
            labels, counts = cluster_budgets(assignments, budget)
            self.assertEqual(counts.sum(), budget)
            self.assertTrue((counts <= [1, 9]).all())
        with self.assertRaises(ValueError):
            cluster_budgets(assignments, 11)

    def test_reference_diversity_when_source_available(self):
        path = Path('master_thesis_jaca00001/src/utls.py')
        if not path.exists():
            self.skipTest('External reference checkout is optional')
        module = ast.parse(path.read_text())
        functions = [n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name in ['kmers','select_distance_based','compute_assignments']]
        namespace = dict(np=np, List=list, Set=set)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        seqs = ['AAAAAA', 'AAACAA', 'CAACAA', 'CCCCCC', 'AAAACC']
        for count in range(1, len(seqs)+1):
            self.assertEqual(diverse_sequences(seqs, count), namespace['select_distance_based'](seqs, count))
        # The reference caller mistakenly passes OHE tensors. Their slice sets
        # have empty intersections, so all distances tie and diversity vanishes.
        tensors = [torch.eye(20)[:, :6].clone() for _ in seqs]
        self.assertEqual(namespace['select_distance_based'](tensors, 3), [0,1,2])
        self.assertNotEqual(diverse_sequences(seqs, 3), [0,1,2])
        assignments = np.array([0]*7+[1]*13)
        expected = namespace['compute_assignments'](np.array([0,1]), assignments, np.arange(20), 8)
        np.testing.assert_array_equal(cluster_budgets(assignments, 8)[1], expected)

    def test_model_features_preserve_forward_and_zero_variance(self):
        torch.set_num_threads(1)
        tokens = np.array([[0,1,2,3], [3,2,1,0]])
        for name, width in [('aubin_1_10_1',10),('mlp_small',64),('mlp_deep',32)]:
            model = Predictor(name, 4)
            expected = model.base(torch.from_numpy(tokens)).detach().numpy()
            mean, variance = infer(model, tokens, np.arange(2), 'cpu', samples=25)
            np.testing.assert_array_equal(mean, expected)
            np.testing.assert_array_equal(variance, np.zeros(2))
            self.assertEqual(infer(model, tokens, np.arange(2), 'cpu', embeddings=True).shape, (2,width))
        model = Predictor('CNN_Jannis_OHE', 4)
        x = np.eye(20,dtype=np.float32)[tokens]
        self.assertEqual(infer(model, x, np.arange(2), 'cpu', embeddings=True).shape, (2,128))
        _, variance = infer(model, x, np.arange(2), 'cpu', samples=3)
        self.assertTrue(np.any(variance > 0))

    def test_selection_unique_and_within_pool(self):
        selected = select_queries(np.arange(6.), np.arange(12.).reshape(6,2)/10,
                                  ['AAAA','AAAC','AACC','ACCC','CCCC','CCCA'], 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertTrue(((selected >= 0) & (selected < 6)).all())

    def test_round_boundary_restart_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            alphabet = 'ACDEFGHIKLMNPQRSTVWY'
            for i in range(400):
                rows.append(dict(record_id=str(i), gene='cgreGFP',
                                 sequence='AA'+alphabet[i//20]+alphabet[i%20],
                                 target_log10=2+np.sin(i/10)))
            data = root/'sequences.csv'
            runner.write_csv(data, rows)
            args = SimpleNamespace(model='mlp_small', seed=42, rounds=2, budget=8,
                batch_size=64, epochs=1, threads=1, device='cpu', data=data,
                embeddings=root/'absent', directory=root/'clean')
            runner.run(args)
            args.directory = root/'interrupted'
            original = runner.train_round

            def interrupt_after_checkpoint(*positional):
                if positional[-1] == 1:
                    raise RuntimeError('Simulated interruption')
                return original(*positional)

            with patch.object(runner, 'train_round', side_effect=interrupt_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'Simulated interruption'):
                    runner.run(args)
            runner.run(args)
            relative = Path('seed_42/mlp_small')
            for name in ['queries.csv','history.csv','predictions_round_02.csv','splits.csv']:
                self.assertEqual((root/'clean'/relative/name).read_bytes(),
                                 (root/'interrupted'/relative/name).read_bytes())


if __name__ == '__main__':
    unittest.main()
