import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.aubin_model import ALPHABET, AubinModel, encode_sequences, load_model
from scripts.train_aubin import train


class AubinTests(unittest.TestCase):
    def test_one_hot_matches_explicit_additive_weights(self):
        model = AubinModel(2)
        with torch.no_grad():
            model.network[0].weight.copy_(torch.arange(60).reshape(1, -1))
            model.network[0].bias.fill_(0.5)
        np.testing.assert_array_equal(model.predict(['AC', 'YA']), [21.5, 39.5])
        for sequence in ['A', 'AX']:
            with self.assertRaises(ValueError):
                model.predict([sequence])

    def test_checkpoint_roundtrip_both_architectures(self):
        for architecture in ['linear', '1_10_1']:
            model = AubinModel(2, architecture)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/'model.pt'
                torch.save(dict(state_dict=model.state_dict(), sequence_length=2,
                                architecture=architecture, alphabet=ALPHABET), path)
                restored, _ = load_model(path)
                np.testing.assert_array_equal(model.predict(['AC', 'YA']), restored.predict(['AC', 'YA']))

    def test_restores_best_validation_checkpoint(self):
        # Opposing train/validation targets make later updates worse on validation.
        torch.manual_seed(42)
        model = AubinModel(2)
        with torch.no_grad():
            model.network[0].weight.zero_()
            model.network[0].bias.zero_()
        x = encode_sequences(['AC']*4, 2)
        history, best = train(model, x, torch.ones(4), x, -torch.ones(4),
                              seed=42, epochs=5, patience=2, batch_size=4, learning_rate=0.1)
        self.assertEqual(best, 1)
        self.assertEqual(len(history), 3)
        mse = float(np.mean((model.predict(['AC']*4)+1)**2))
        self.assertAlmostEqual(mse, history[0]['validation_mse'], places=6)


if __name__ == '__main__':
    unittest.main()
