"""Protect nested sampling, source-only selection, paired targets and saved predictions."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scripts.learning_curves.common import read_csv, write_csv, fit_directory
from scripts.learning_curves.prepare import prepare, nested_samples
from scripts.learning_curves.run import load_context, fit


class LearningCurveTests(unittest.TestCase):
    def make_data(self,path):
        alphabet='ACDEFGHIKLMNPQRSTVWY'; rows=[]
        for g,gene in enumerate(['cgreGFP','cgre132','cgre1338','cgre4111','cgre9708']):
            for i in range(40 if g==0 else 20):
                rows.append(dict(record_id=f'{gene}:{i}',gene=gene,
                    sequence=alphabet[g]+alphabet[i//20]+alphabet[i%20]+'CDEFGH',target_log10=2+(i%7)/10+g/20,split='train'))
        write_csv(path,rows)
        return rows

    def test_nested_peak_proportions(self):
        rows=[dict(gene=g) for g,n in [('a',20),('b',40),('c',60),('d',80)] for _ in range(n)]
        samples=nested_samples(rows,np.arange(200),[10,50,100,200],42)
        previous=set()
        for size,idx in samples.items():
            self.assertEqual(len(idx),size)
            self.assertTrue(previous<=set(idx));previous=set(idx)
            counts=Counter(rows[i]['gene'] for i in idx)
            self.assertEqual([counts[g] for g in ['a','b','c','d']],[size//10,size*2//10,size*3//10,size*4//10])

    def test_folds_source_only_training_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);data=root/'data.csv';rows=self.make_data(data)
            directory=root/'experiment';protocol=prepare(data,directory,[4,8])
            for source,expected in [('natural',40),('artificial',80)]:
                seen=Counter()
                for fold in range(10):
                    split=read_csv(directory/'splits'/f'{source}_fold_{fold:02d}.csv')
                    seen.update(r['record_id'] for r in split if r['split']=='test')
                self.assertEqual(len(seen),expected)
                self.assertEqual(set(seen.values()),{1})
            for model in ['aubin_1_10_1','CNN_Jannis_OHE']:
                c=load_context(directory,'artificial',0,model=='CNN_Jannis_OHE')
                self.assertTrue(all(c.rows[i]['gene']!='cgreGFP' for i in c.ids['validation']))
                self.assertTrue(all(c.rows[i]['gene']!='cgreGFP' for i in c.subsets[4]))
                fit(c,model,4,'cpu',epochs=1)
                folder=fit_directory(directory,'artificial',0,4,model)
                report=json.loads((folder/'metrics.json').read_text())
                self.assertEqual(set(report['metrics']),{'validation','natural','artificial'})
                self.assertEqual(report['counts']['train'],4)
                predicted=read_csv(folder/'predictions.csv')
                self.assertEqual(len(predicted),sum(report['counts'][s] for s in ['validation','natural','artificial']))
                fit(c,model,4,'cpu',epochs=1)  # Verify completed-fit provenance and saved artifact checks.
            with self.assertRaises(FileExistsError):
                prepare(data,directory,[4,8])
            split=directory/'subsets/artificial_fold_00_n_00004.csv'
            split.write_text(split.read_text()+'\n')
            with self.assertRaises(AssertionError):
                load_context(directory,'artificial',0,False)
