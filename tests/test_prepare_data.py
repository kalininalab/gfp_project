"""Scientific invariants for sequence reconstruction and benchmark membership."""

import unittest

from scripts.prepare_data import Excluded, reconstruct, split_indices


class ReconstructionTests(unittest.TestCase):
    def test_zero_based_positions_and_terminal_stop(self):
        self.assertEqual(reconstruct('A1F:M0V', 'MAG*'), ('VFG', 'M0V:A1F', 2))
        self.assertEqual(reconstruct('wt', 'MAG*'), ('MAG', 'wt', 0))

    def test_wrong_reference_and_duplicate_positions_fail(self):
        for genotype in ['A2F', 'G5A', 'A1F:A1G', 'A1A']:
            with self.subTest(genotype=genotype), self.assertRaises(ValueError):
                reconstruct(genotype, 'MAG*')

    def test_exclusion_policies(self):
        for genotype, reason in [('A1*', 'stop_codon_mutation'),
                                 ('*3A', 'stop_codon_mutation'),
                                 ('A1.', 'unsupported_notation'),
                                 ('wt_ctrl_cgre132', 'control')]:
            with self.subTest(genotype=genotype), self.assertRaisesRegex(Excluded, reason):
                reconstruct(genotype, 'MAG*')

    def test_split_is_exhaustive_reproducible_and_seed_sensitive(self):
        split = split_indices(100, 42)
        self.assertEqual(split, split_indices(100, 42))
        self.assertNotEqual(split, split_indices(100, 43))
        self.assertEqual(set(split), set(range(100)))
        self.assertEqual([list(split.values()).count(s) for s in ['train', 'validation', 'test']], [60, 20, 20])


if __name__ == '__main__':
    unittest.main()
