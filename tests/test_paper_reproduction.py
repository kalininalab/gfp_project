"""Checks for the manuscript-described random 80/20 reproduction split."""
import unittest
import numpy as np
from scripts.paper_reproduction.run import split_indices


class PaperReproductionTests(unittest.TestCase):
    def test_split_is_exact_disjoint_exhaustive_and_reproducible(self):
        a_train,a_test=split_indices(24516,42)
        b_train,b_test=split_indices(24516,42)
        self.assertEqual((len(a_train),len(a_test)),(19612,4904))
        self.assertEqual(len(set(a_train)&set(a_test)),0)
        self.assertEqual(set(a_train)|set(a_test),set(range(24516)))
        np.testing.assert_array_equal(a_train,b_train)
        np.testing.assert_array_equal(a_test,b_test)

    def test_seed_changes_split(self):
        _,a=split_indices(100,42);_,b=split_indices(100,43)
        self.assertFalse(np.array_equal(a,b))


if __name__=='__main__':unittest.main()
