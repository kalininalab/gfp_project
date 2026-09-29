"""Validate GFP genotypes, prepare log10 targets, and freeze per-landscape splits."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
COMBINED = 'data/raw/amacGFP_cgreGFP_ppluGFP2__final_aminoacid_genotypes_to_brightness.csv'
RESCALED = 'data/processed/amac_pplu__aadata_rescaled_by_WTctrls.csv'
AVGFP = 'data/raw/avGFP__rf_aminoacid_genotypes_to_brightness.csv'
FASTA = 'data/raw/fasta_sequences/protein_seqs.fa'
PEAKS = ('cgre132', 'cgre1338', 'cgre4111', 'cgre9708')
AA = 'ACDEFGHIKLMNPQRSTVWY'
TOKEN = re.compile(r'([A-Z*])(\d+)([A-Z*])')


class Excluded(ValueError):
    """An explicitly unsupported genotype, recorded in the exclusion ledger."""


def read_fasta(path):
    sequences = {}
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            name = line[1:].split()[0]
            if name in sequences:
                raise ValueError(f'Duplicate FASTA name: {name}')
            sequences[name] = ''
        elif line.strip():
            sequences[name] += line.strip()
    return sequences


def reconstruct(genotype, reference):
    """Apply zero-based substitutions only after validating all reference residues."""
    if genotype.startswith('wt_ctrl_'):
        raise Excluded('control')
    if genotype == 'wt':
        return reference.rstrip('*'), 'wt', 0
    tokens = genotype.split(':')
    matches = [TOKEN.fullmatch(token) for token in tokens]
    if not all(matches):
        raise Excluded('unsupported_notation')
    sequence = list(reference)
    positions = set()
    mutations = []
    for match in matches:
        old, position, new = match.groups()
        position = int(position)
        if position >= len(reference) or reference[position] != old:
            raise ValueError(f'Reference mismatch: {match.group()}')
        if position in positions:
            raise ValueError(f'Repeated mutation position: {genotype}')
        positions.add(position)
        if old == new:
            raise ValueError(f'No-op mutation: {match.group()}')
        sequence[position] = new
        mutations.append((position, match.group()))
    if '*' in genotype:
        raise Excluded('stop_codon_mutation')
    if any(old not in AA or new not in AA for old, _, new in [m.groups() for m in matches]):
        raise Excluded('noncanonical_amino_acid')
    return ''.join(sequence).rstrip('*'), ':'.join(t for _, t in sorted(mutations)), len(tokens)


def split_indices(size, seed):
    """Same two ShuffleSplit calls as Aubin's 01_preprocessing notebook."""
    train, remainder = train_test_split(list(range(size)), test_size=0.4, random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed)
    return {i: label for label, ids in [('train', train), ('validation', validation), ('test', test)] for i in ids}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(root, output, seed):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be empty: {output}')
    references = read_fasta(root / FASTA)
    specifications = [
        (RESCALED, {'amacGFP', 'ppluGFP'}, 'aa_genotype_native', 'scaled_WTctrl_fit', False, False),
        (COMBINED, {'cgreGFP'}, 'aa_genotype_native', 'replicates_mean_brightness', False, False),
        (AVGFP, {'avGFP'}, 'aa_genotype_native', 'log_brightness', True, False),
    ]
    specifications += [(f'data/raw/240228__ntdata_{gene}_c075-genotypes__err1__cgreWTgates.csv',
                        {gene}, 'aa_genotype', 'brightness', False, True) for gene in PEAKS]
    groups = defaultdict(dict)
    rejected = []
    counts = Counter()
    for source, genes, genotype_col, target_col, logged, peak in specifications:
        with (root / source).open(newline='') as stream:
            for source_row, row in enumerate(csv.DictReader(stream), start=2):
                gene = row.get('gene', 'avGFP')
                if gene not in genes:
                    continue
                counts[gene] += 1
                genotype = row[genotype_col]
                try:
                    sequence, canonical, n_mutations = reconstruct(genotype, references[gene])
                    value = float(row[target_col])
                    if not math.isfinite(value):
                        raise Excluded('nonfinite_target')
                    if not logged and value <= 0:
                        raise Excluded('nonpositive_fluorescence')
                    target = value if logged else math.log10(value)
                    # Notebook threshold: positive log10 fluorescence.
                    if target <= 0:
                        raise Excluded('nonpositive_log10_target')
                    if source == RESCALED and not math.isclose(target, float(row['scaled_WTctrl_fit__log10']), abs_tol=1e-10):
                        raise ValueError('Provided log10 target disagrees with calibrated brightness')
                    weight = float(row['pseudocell_count']) if peak else 1.0
                    if not math.isfinite(weight) or weight <= 0:
                        raise Excluded('invalid_aggregation_weight')
                except Excluded as exc:
                    rejected.append(dict(source_file=source, source_row=source_row, gene=gene,
                                         genotype=genotype, reason=str(exc)))
                    continue
                except ValueError as exc:
                    raise ValueError(f'{source}:{source_row}: {gene}: {exc}') from exc
                group = groups[gene].setdefault(sequence, dict(
                    gene=gene, dataset_kind='artificial_peak' if peak else 'natural',
                    genotype=canonical, sequence=sequence, n_mutations=n_mutations,
                    source_file=source, source_target_column=target_col,
                    source_target_is_log10=logged, values=[], weights=[], source_rows=[]))
                if not peak and group['source_rows']:
                    raise ValueError(f'Duplicate protein sequence in {source}: {genotype}')
                group['values'].append(value)
                group['weights'].append(weight)
                group['source_rows'].append(source_row)

    records = []
    summaries = {}
    for gene in sorted(groups):
        # Preserve first source occurrence, as the reference notebooks do.
        rows = list(groups[gene].values())
        splits = split_indices(len(rows), seed)
        for i, row in enumerate(rows):
            values, weights = row.pop('values'), row.pop('weights')
            mean = math.fsum(v*w for v, w in zip(values, weights)) / math.fsum(weights)
            row['source_value'] = mean
            row['target_log10'] = mean if row['source_target_is_log10'] else math.log10(mean)
            row['n_source_rows'] = len(row['source_rows'])
            row['aggregation_weight_sum'] = math.fsum(weights)
            row['source_rows'] = ';'.join(map(str, row['source_rows']))
            row['sequence_id'] = hashlib.sha256(row['sequence'].encode()).hexdigest()
            row['record_id'] = f"{gene}:{row['sequence_id']}"
            row['split'] = splits[i]
        summaries[gene] = dict(input_rows=counts[gene], retained_sequences=len(rows),
                               retained_source_rows=sum(r['n_source_rows'] for r in rows),
                               exclusions=dict(Counter(r['reason'] for r in rejected if r['gene']==gene)),
                               splits=dict(Counter(r['split'] for r in rows)))
        records.extend(rows)
    # Per-landscape splits must not be silently pooled into a transfer benchmark.
    backgrounds = defaultdict(set)
    for row in records:
        backgrounds[row['sequence_id']].add(row['gene'])
    overlaps = {s: sorted(g) for s, g in backgrounds.items() if len(g)>1}
    manifest = dict(schema_version=1, seed=seed, split_fractions=[0.6, 0.2, 0.2],
                    split_scope='within_landscape', versions=dict(python=platform.python_version(),
                    numpy=np.__version__, scikit_learn=sklearn.__version__),
                    inputs={p: digest(root/p) for p in [FASTA]+[s[0] for s in specifications]},
                    script_sha256=digest(Path(__file__)), landscapes=summaries,
                    cross_landscape_sequence_overlaps=overlaps)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'sequences.csv', records, list(records[0]))
    write_csv(output/'exclusions.csv', rejected, ['source_file','source_row','gene','genotype','reason'])
    manifest['outputs'] = {p: digest(output/p) for p in ['sequences.csv', 'exclusions.csv']}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/baseline_v1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve(), args.seed)
