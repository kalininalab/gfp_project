import unittest
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import numpy as np
import torch

from scripts.transfer_benchmark.prepare import allocate,star_alignment
from scripts.transfer_benchmark.common import GENES,NATURAL,PEAKS,ROOT
from scripts.transfer_benchmark.models import AlignedModel,encode_aligned
from scripts.transfer_benchmark.run import ResidueFeatures
from scripts.prepare_data import read_fasta
from scripts.aubin_model import AubinModel,encode_sequences
from scripts.transfer_benchmark.common import digest,write_csv,scientific_sources
from scripts.transfer_benchmark.run import run


class TransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)

    def test_alignment_preserves_every_residue_and_mutation_position(self):
        refs={g:s.rstrip('*') for g,s in read_fasta(ROOT/'data/raw/fasta_sequences/protein_seqs.fa').items() if g in GENES}
        a=star_alignment(refs)
        for gene in GENES:
            self.assertEqual(a['aligned_wildtypes'][gene].replace('-',''),refs[gene])
            self.assertEqual(len(set(a['mapping'][gene])),len(refs[gene]))
            seq=refs[gene];changed=seq[:13]+('A' if seq[13]!='A' else 'F')+seq[14:]
            x=encode_aligned([{'gene':gene,'sequence':seq},{'gene':gene,'sequence':changed}],a)
            self.assertEqual(torch.nonzero(x[0]!=x[1]).flatten().tolist(),[a['mapping'][gene][13]])
        for peak in PEAKS:
            self.assertEqual(a['mapping'][peak],a['mapping']['cgreGFP'])

    def test_gap_free_encoding_matches_original_aubin(self):
        a=AlignedModel(4,'aubin_1_10_1');b=AubinModel(4,'1_10_1')
        b.load_state_dict(a.state_dict())
        x=encode_sequences(['ACDE','FCHI'],4)
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
        x[0,1]=-1
        a(x).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in a.parameters()))

    def test_balanced_and_proportional_budgets(self):
        self.assertEqual(allocate(14709,[1,1,1]),[4903]*3)
        self.assertEqual(allocate(14709,[1,1]),[7355,7354])
        self.assertEqual(sum(allocate(4903,[2560,6144,4928,2508])),4903)

    def test_variable_lengths_preserve_order_and_gradients(self):
        # A synthetic head verifies grouped forward passes without padding/cropping.
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
            def forward(self,x):
                return x.mean((1,2))*self.weight
        features=ResidueFeatures.__new__(ResidueFeatures)
        features.features={'a':np.ones((2,7,3),dtype=np.float16),'b':np.full((2,11,3),3,dtype=np.float16)}
        features.lookup={'a':np.array([0,-1,1,-1]),'b':np.array([-1,0,-1,1])}
        model=Head();pred=features.forward(model,np.array([3,0,1,2]),'cpu')
        torch.testing.assert_close(pred,torch.tensor([6.,2.,6.,2.]))
        pred.sum().backward();self.assertEqual(model.weight.grad.item(),8.)

    def test_complete_source_only_fit_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data.csv'
            aa='ACDEFGHIKLMN'
            rows=[dict(record_id=str(i),gene='amacGFP' if i<9 else 'cgreGFP',sequence=aa[i]+'AAA',
                       split='train' if i<6 else 'validation' if i<9 else 'test',target_log10=2+i*.1) for i in range(12)]
            write_csv(data,rows)
            subset=root/'subsets/amac_seed42.csv'
            write_csv(subset,[{k:r[k] for k in ['record_id','gene','split']} for r in rows[:9]])
            write_csv(root/'test_records.csv',[dict(record_id=r['record_id'],gene=r['gene'],target_group='natural') for r in rows[9:]])
            alignment=dict(length=4,mapping={g:list(range(4)) for g in ['amacGFP','cgreGFP']})
            (root/'alignment.json').write_text(json.dumps(alignment))
            protocol=dict(source_hashes=scientific_sources(),dataset_sha256=digest(data),seeds=[42],mixes={'amac':['amacGFP']},
                n_train=6,n_validation=3,subset_hashes={'subsets/amac_seed42.csv':digest(subset)},
                test_records_sha256=digest(root/'test_records.csv'),alignment_sha256=digest(root/'alignment.json'),
                onehot_settings=dict(epochs=2,patience=10,batch_size=3,lr=.001))
            (root/'protocol.json').write_text(json.dumps(protocol))
            with patch('scripts.transfer_benchmark.run.DATA',data),redirect_stdout(io.StringIO()):
                for model in ['aubin_1_10_1','mlp_deep']:
                    run(argparse.Namespace(directory=root,mix='amac',seed=42,model=model,device='cpu'))
                    report=json.loads((root/'fits/amac/seed42'/model/'metrics.json').read_text())
                    self.assertEqual(report['counts']['train'],{'amacGFP':6})
                    self.assertEqual(report['counts']['validation'],{'amacGFP':3})
                    self.assertEqual(report['metrics']['natural']['n'],3)


if __name__=='__main__':
    unittest.main()
