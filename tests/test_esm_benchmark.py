"""Checks that protect the original CNN failure modes and architecture fidelity."""
import ast
from pathlib import Path
import unittest
import argparse
import csv
from contextlib import redirect_stdout
import io
import json
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.esm_benchmark.models import CNN_old, CNN_new, CNN_Jannis
from scripts.esm_benchmark.embed import digest
from scripts.esm_benchmark.run import run, load_data
from scripts.esm_benchmark.predict import load_predictor


class CNNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    @unittest.skipUnless(Path("master_thesis_jaca00001/src/model.py").is_file(),
                         "Optional external Jannis repository is not installed")
    def test_jannis_forward_matches_original(self):
        # Extract only the original architecture; do not import its active-learning pipeline.
        source = ast.parse(Path('master_thesis_jaca00001/src/model.py').read_text())
        cls = next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ProtCNN')
        scope = {'torch':torch,'nn':nn,'F':F}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'Jannis architecture','exec'),scope)
        original = scope['ProtCNN'](640,512,4,4,.13249619898782194).eval()
        port = CNN_Jannis().eval()
        port.load_state_dict(original.state_dict())
        x = torch.randn(2,35,640)
        with torch.inference_mode():
            np.testing.assert_allclose(port(x).numpy(),original(x.transpose(1,2)).squeeze(-1).numpy(),atol=1e-6)

    def test_old_forward_and_gradients_match_original(self):
        source=ast.parse(Path('CNN/model_legacy.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ConvNet')
        scope={'torch':torch,'nn':nn}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'original CNN','exec'),scope)
        original=scope['ConvNet'](640)
        port=CNN_old()
        for p,q in zip(port.parameters(),original.parameters()):
            p.data.copy_(q.data)
        x=torch.randn(2,35,640)
        a=original(x.transpose(1,2)).squeeze(-1); b=port(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.sum().backward(); b.sum().backward()
        for p,q in zip(port.parameters(),original.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_all_cnn_gradients_and_checkpoint_copy(self):
        for cls in [CNN_old,CNN_new,CNN_Jannis]:
            model = cls()
            saved = {k:v.detach().clone() for k,v in model.state_dict().items()}
            expected = {k:v.clone() for k,v in saved.items()}
            pred = model(torch.randn(2,35,640))
            self.assertEqual(tuple(pred.shape),(2,))
            pred.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            torch.optim.Adam(model.parameters(),lr=.001).step()
            self.assertTrue(all(torch.equal(saved[k],expected[k]) for k in saved))

    def test_training_and_saved_predictions_remain_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emb = root/'emb'; emb.mkdir()
            rng = np.random.default_rng(42)
            full = rng.normal(size=(120,9,640)).astype(np.float16)
            mean = full.astype(np.float32).mean(1)
            np.save(emb/'residue_embeddings.npy',full)
            np.save(emb/'mean_embeddings.npy',mean)
            rows = [dict(record_id=str(i),sequence=f'{i:09d}',gene='cgreGFP',target_log10=float(mean[i,0]+2),
                         split='train' if i<80 else 'validation' if i<100 else 'test') for i in range(120)]
            data = root/'data.csv'
            with data.open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            with (emb/'records.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=['record_id','sequence']); w.writeheader()
                w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
            (emb/'manifest.json').write_text(json.dumps(dict(gene='cgreGFP',dataset_sha256=digest(data),
                files={n:digest(emb/n) for n in ['residue_embeddings.npy','mean_embeddings.npy','records.csv']})))
            for name in ['ridge','knn','aubin_1_10_1','CNN_old']:
                args=argparse.Namespace(output=str(root/'results'),model=name,split_name='holdout',
                    seed=42,data=str(data),embeddings=str(emb),split='holdout',device='cpu',epochs=2)
                with redirect_stdout(io.StringIO()):
                    run(args)
                directory=root/'results'/'holdout'/name
                saved=list(csv.DictReader((directory/'predictions.csv').open()))
                self.assertEqual([r['record_id'] for r in saved],[str(i) for i in range(80,120)])
                fn,report=load_predictor(directory)
                result=fn((full if name.startswith('CNN_') else mean)[80:])
                np.testing.assert_allclose(result,[float(r['y_pred']) for r in saved],rtol=1e-5,atol=1e-6)
                self.assertGreater(report['fit_seconds'],0)
            records=list(csv.DictReader((emb/'records.csv').open()))
            records[0],records[1]=records[1],records[0]
            with (emb/'records.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=['record_id','sequence']); w.writeheader(); w.writerows(records)
            with self.assertRaisesRegex(ValueError,'order/sequence mismatch'):
                load_data(data,emb,'holdout')


if __name__=='__main__':
    unittest.main()
