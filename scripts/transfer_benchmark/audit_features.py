"""Check full ESM caches once before transfer CNN jobs; freeze file fingerprints."""
import argparse
import json
from pathlib import Path

import numpy as np
from .common import ROOT,DEFAULT,DATA,GENES,digest,read_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=DEFAULT)
    a=p.parse_args(); rows=read_csv(DATA); data_hash=digest(DATA);report={}
    for gene in GENES:
        directory=ROOT/'esm_embeddings'/f'{gene}_t30'
        manifest=json.loads((directory/'manifest.json').read_text())
        expected=[r for r in rows if r['gene']==gene];actual=read_csv(directory/'records.csv')
        assert manifest['dataset_sha256']==data_hash
        assert manifest['model']=='facebook/esm2_t30_150M_UR50D'
        assert manifest['revision']=='a695f6045e2e32885fa60af20c13cb35398ce30c'
        assert [(r['record_id'],r['sequence']) for r in expected]==[(r['record_id'],r['sequence']) for r in actual]
        for name in ['records.csv','residue_embeddings.npy','mean_embeddings.npy']:
            if digest(directory/name)!=manifest['files'][name]:
                raise ValueError(f'{gene}: corrupted {name}')
        full=np.load(directory/'residue_embeddings.npy',mmap_mode='r')
        mean=np.load(directory/'mean_embeddings.npy',mmap_mode='r')
        assert full.shape==(len(expected),len(expected[0]['sequence']),640)
        assert np.isfinite(mean).all()
        ix=np.random.default_rng(42).choice(len(full),min(100,len(full)),replace=False)
        np.testing.assert_array_equal(full[ix].astype(np.float32).mean(1),mean[ix])
        files={name:dict(sha256=manifest['files'][name],size=(directory/name).stat().st_size,
                         mtime_ns=(directory/name).stat().st_mtime_ns)
               for name in ['records.csv','residue_embeddings.npy']}
        report[gene]=dict(directory=str(directory),manifest_sha256=digest(directory/'manifest.json'),files=files,shape=list(full.shape))
        print(gene,'validated',flush=True)
    (a.directory/'features_audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
