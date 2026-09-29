"""Generate residue embeddings (no special tokens) and their arithmetic means."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from transformers import AutoModel, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/processed/baseline_v1/sequences.csv')
    p.add_argument('--gene', default='cgreGFP')
    p.add_argument('--model', default='facebook/esm2_t30_150M_UR50D')
    p.add_argument('--revision', required=True, help='Immutable Hugging Face commit hash')
    p.add_argument('--output', default='esm_embeddings/cgreGFP_t30')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [r for r in csv.DictReader(open(a.data)) if r['gene'] == a.gene]
    lengths = {len(r['sequence']) for r in rows}
    if len(lengths) != 1 or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Require unique records and equal sequence lengths within a protein')
    length = lengths.pop()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Completed embeddings exist; use a new output directory')
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision, local_files_only=True)
    model = AutoModel.from_pretrained(a.model, revision=a.revision, local_files_only=True,
                                     add_pooling_layer=False).eval().to(a.device)
    assert model.config.num_hidden_layers == 30 and model.config.hidden_size == 640
    full = np.lib.format.open_memmap(out/'residue_embeddings.npy', mode='w+', dtype='float16', shape=(len(rows), length, 640))
    mean = np.lib.format.open_memmap(out/'mean_embeddings.npy', mode='w+', dtype='float32', shape=(len(rows), 640))
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), a.batch_size):
            seqs = [r['sequence'] for r in rows[offset:offset+a.batch_size]]
            tokens = tokenizer(seqs, return_tensors='pt', padding=True, return_special_tokens_mask=True)
            mask = tokens.pop('special_tokens_mask').bool()
            valid = tokens['attention_mask'].bool() & ~mask
            assert (valid.sum(1) == length).all()
            result = model(**{k:v.to(a.device) for k,v in tokens.items()}).last_hidden_state.cpu()
            # Float32 inference; compact float16 storage. Both heads use identical stored features.
            residues = result[valid].reshape(len(seqs), length, 640).numpy().astype(np.float16)
            full[offset:offset+len(seqs)] = residues
            mean[offset:offset+len(seqs)] = residues.astype(np.float32).mean(axis=1)
            if offset % (a.batch_size*25) == 0:
                print(f'{offset}/{len(rows)} sequences; {time.perf_counter()-start:.1f}s', flush=True)
    full.flush(); mean.flush()
    inference_seconds = time.perf_counter()-inference_start
    with (out/'records.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['record_id','sequence'])
        writer.writeheader(); writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    manifest = dict(model=a.model, revision=a.revision, layer=30, shape=list(full.shape),
                    residue_dtype='float16', mean_dtype='float32', inference_dtype='float32',
                    special_tokens='excluded', pooling='arithmetic mean over residues only',
                    gene=a.gene, dataset_sha256=digest(a.data), script_sha256=digest(__file__),
                    inference_seconds=inference_seconds, elapsed_seconds=time.perf_counter()-start,
                    host=platform.node(), torch=torch.__version__, device=str(a.device),
                    gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
                    files={name:digest(out/name) for name in ['residue_embeddings.npy','mean_embeddings.npy','records.csv']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
