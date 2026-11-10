"""Freeze all source combinations for four-way leave-one-protein-out transfer."""
import argparse
import datetime
import json

from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np

from scripts.prepare_data import read_fasta
from .common import (ROOT, DATA, DEFAULT, GENES, MODELS, condition_id, digest, read_csv,
                     scientific_sources, sources_for, write_csv)

N_TRAIN, N_VALIDATION = 14709, 4903


def allocate(total, n):
    counts = [total // n] * n
    for index in range(total % n): counts[index] += 1
    return counts


def star_alignment(references):
    anchor = references['cgreGFP']
    aligner = PairwiseAligner(mode='global', substitution_matrix=substitution_matrices.load('BLOSUM62'),
                              open_gap_score=-10., extend_gap_score=-.5)
    layouts = {}
    for gene in GENES:
        query = references[gene]
        if gene == 'cgreGFP':
            aligned_anchor, aligned_query = anchor, anchor
        else:
            alignment = aligner.align(anchor, query)[0]
            aligned_anchor, aligned_query = alignment[0], alignment[1]
        insertions = [[] for _ in range(len(anchor) + 1)]; residues = [None] * len(anchor)
        anchor_i = query_i = 0
        for a, b in zip(aligned_anchor, aligned_query):
            if a == '-': insertions[anchor_i].append(query_i); query_i += 1
            else:
                if b != '-': residues[anchor_i] = query_i; query_i += 1
                anchor_i += 1
        layouts[gene] = insertions, residues
    widths = [max(len(layouts[g][0][i]) for g in GENES) for i in range(len(anchor) + 1)]
    mappings, aligned = {}, {}
    for gene in GENES:
        insertions, residues = layouts[gene]; columns = []
        for i in range(len(anchor) + 1):
            columns += insertions[i] + [None] * (widths[i] - len(insertions[i]))
            if i < len(anchor): columns.append(residues[i])
        mapping = [None] * len(references[gene])
        for column, position in enumerate(columns):
            if position is not None: mapping[position] = column
        assert None not in mapping
        mappings[gene] = mapping
        aligned[gene] = ''.join('-' if position is None else references[gene][position] for position in columns)
    return dict(length=len(aligned['cgreGFP']), mapping=mappings, aligned_wildtypes=aligned,
                method='cgreGFP-anchored WT star alignment; BLOSUM62; global affine gaps -10/-0.5')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=type(DEFAULT), default=DEFAULT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists(): raise FileExistsError('Choose a new output directory')
    rows = [row for row in read_csv(DATA) if row['gene'] in GENES]
    assert len({row['record_id'] for row in rows}) == len(rows)
    # amacGFP/avGFP contain two cross-landscape duplicate sequences. Assign
    # each source sequence to the first gene in GENES so mixtures never
    # double-weight it; target tests remain intact and are excluded separately.
    owner = {}
    for gene in GENES:
        for row in rows:
            if row['gene'] == gene: owner.setdefault(row['sequence'], gene)
    pools = {(gene, split): [row for row in rows if row['gene'] == gene and row['split'] == split
                             and owner[row['sequence']] == gene]
             for gene in GENES for split in ['train', 'validation', 'test']}
    references = {gene: sequence.rstrip('*') for gene, sequence in
                  read_fasta(ROOT / 'data/raw/fasta_sequences/protein_seqs.fa').items() if gene in GENES}
    alignment = star_alignment(references)
    (out / 'alignment.json').write_text(json.dumps(alignment, indent=2) + '\n')
    tests = [dict(target=gene, record_id=row['record_id'], gene=gene)
             for gene in GENES for row in rows if row['gene'] == gene and row['split'] == 'test']
    write_csv(out / 'test_records.csv', tests)
    subset_hashes, counts, jobs = {}, [], []
    for target in GENES:
        target_test_sequences = {row['sequence'] for row in rows if row['gene'] == target and row['split'] == 'test'}
        for sources in sources_for(target):
            condition = condition_id(sources)
            for seed in args.seeds:
                selected = []
                for split, total in [('train', N_TRAIN), ('validation', N_VALIDATION)]:
                    for gene, budget in zip(sources, allocate(total, len(sources))):
                        pool = [row for row in pools[gene, split] if row['sequence'] not in target_test_sequences]
                        assert budget <= len(pool)
                        rng = np.random.default_rng(np.random.SeedSequence([seed, GENES.index(gene), int(split == 'validation')]))
                        chosen = sorted(rng.permutation(len(pool))[:budget].tolist())
                        selected += [dict(record_id=pool[i]['record_id'], gene=gene, split=split) for i in chosen]
                        counts.append(dict(target=target, condition=condition, seed=seed, split=split,
                                           source=gene, n=budget, available=len(pool)))
                path = out / 'subsets' / target / f'{condition}_seed{seed}.csv'
                write_csv(path, selected); subset_hashes[str(path.relative_to(out))] = digest(path)
                jobs += [dict(target=target, condition=condition, seed=seed, model=model) for model in MODELS]
    write_csv(out / 'sample_counts.csv', counts); write_csv(out / 'jobs.csv', jobs)
    (out / 'cpu_jobs.txt').write_text(''.join(f'{j["target"]} {j["condition"]} {j["seed"]} {j["model"]}\n' for j in jobs if j['model'] != 'CNN_Jannis_OHE'))
    (out / 'gpu_jobs.txt').write_text(''.join(f'{j["target"]} {j["condition"]} {j["seed"]}\n' for j in jobs if j['model'] == 'CNN_Jannis_OHE'))
    (out / 'condor').mkdir(exist_ok=True)
    protocol = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dataset_sha256=digest(DATA), genes=GENES, seeds=args.seeds, models=MODELS,
        conditions={target:[condition_id(x) for x in sources_for(target)] for target in GENES},
        n_train=N_TRAIN, n_validation=N_VALIDATION,
        design='Each target uses all seven nonempty combinations of the other three proteins; target contributes no fit labels.',
        source_balance='Fixed total budget; equal source contributions within every combination, rounded by at most one record.',
        model_selection='source-only validation MSE; frozen target test labels never select checkpoints',
        test_counts={gene:sum(row['gene']==gene and row['split']=='test' for row in rows) for gene in GENES},
        alignment_sha256=digest(out/'alignment.json'), test_records_sha256=digest(out/'test_records.csv'),
        subset_hashes=subset_hashes, source_hashes=scientific_sources(),
        onehot_settings=dict(epochs=30,patience=10,batch_size=32,lr=.001),
        cnn_settings=dict(epochs=60,patience=10,batch_size=64,lr=.0003,weight_decay=.0001))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'fits':len(jobs),'cpu':len(jobs)*3//4,'gpu':len(jobs)//4,'test_counts':protocol['test_counts']}, indent=2))


if __name__ == '__main__': main()
