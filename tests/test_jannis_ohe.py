"""Native OHE must retain residue positions and train with the shared CNN recipe."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scripts.train_jannis_ohe import load_ohe, run
from scripts.esm_benchmark.predict import load_predictor


class JannisOHETests(unittest.TestCase):
    def test_ohe_positions_split_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data.csv'
            alphabet='ACDEFGHIKLMNPQRSTVWY'
            rows=[dict(record_id=str(i),gene='cgreGFP',sequence=aa+'CDEFGHIK',
                       target_log10=2+i/20,split='train' if i<12 else 'validation' if i<16 else 'test')
                  for i,aa in enumerate(alphabet)]
            with data.open('w') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            _,ids,x,y=load_ohe(data,'holdout')
            self.assertEqual(x.shape,(20,9,20))
            np.testing.assert_array_equal(x.sum(-1),np.ones((20,9)))
            np.testing.assert_array_equal(x[:,0].argmax(-1),np.arange(20))
            self.assertEqual([len(ids[s]) for s in ['train','validation','test']],[12,4,4])
            args=argparse.Namespace(data=str(data),output=str(root/'out'),split='holdout',split_name='holdout',seed=42,device='cpu',epochs=1)
            run(args)
            folder=root/'out/holdout/CNN_Jannis_OHE'
            fn,report=load_predictor(folder)
            pred=fn(x[ids['test']])
            with (folder/'predictions.csv').open() as stream:
                saved=list(csv.DictReader(stream))
            np.testing.assert_allclose(pred,[float(r['y_pred']) for r in saved if r['split']=='test'],rtol=0,atol=0)
            self.assertEqual(report['input_shape'],[20,9,20])
            self.assertEqual(report['parameters'],4698881)
            run(args)  # Completed-run provenance and artifact checks.
