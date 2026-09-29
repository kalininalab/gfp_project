"""Audit sequence identity before adopting an externally supplied split."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

try:
    from .train_aubin import ROOT, sha256
except ImportError:
    from train_aubin import ROOT, sha256


def audit(external, prepared, output, gene):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Audit output must be empty')
    with external.open() as stream:
        incoming = list(csv.DictReader(stream))
    with prepared.open() as stream:
        local = list(csv.DictReader(stream))
    by_sequence = defaultdict(list)
    for row in incoming:
        by_sequence[row['sequence']].append(row)
    matches, counts, split_counts = [], Counter(), defaultdict(Counter)
    matched_external = set()
    for row in local:
        for supplied in by_sequence.get(row['sequence'], []):
            counts[row['gene']] += 1
            split_counts[row['gene']][supplied['split']] += 1
            matched_external.add(supplied['ID'])
            matches.append(dict(external_id=supplied['ID'], gene=row['gene'], record_id=row['record_id'],
                                sequence_id=row['sequence_id'], supplied_split=supplied['split'],
                                our_target_log10=row['target_log10']))
    conflicts = [seq for seq, rows in by_sequence.items() if len({r['split'] for r in rows}) > 1]
    target_records = [r for r in local if r['gene'] == gene]
    report = dict(requested_gene=gene, external_rows=len(incoming), unique_external_sequences=len(by_sequence),
                  external_splits=dict(Counter(r['split'] for r in incoming)),
                  external_lengths=dict(Counter(len(r['sequence']) for r in incoming)),
                  duplicate_sequence_groups=sum(len(v)>1 for v in by_sequence.values()),
                  cross_split_sequence_conflicts=len(conflicts),
                  exact_matches_by_gene=dict(counts), matching_splits_by_gene={k:dict(v) for k,v in split_counts.items()},
                  unmatched_external_rows=sum(r['ID'] not in matched_external for r in incoming),
                  requested_gene_sequences=len(target_records),
                  requested_gene_matched=sum(r['sequence'] in by_sequence for r in target_records),
                  inputs={str(external):sha256(external), str(prepared):sha256(prepared)},
                  script_sha256=sha256(Path(__file__)),
                  note='Exact sequence equality only; supplied labels are not used. Matching another gene does not authorize transferring its split to the requested gene.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'sequence_matches.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['external_id','gene','record_id','sequence_id','supplied_split','our_target_log10'])
        writer.writeheader()
        writer.writerows(matches)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', type=Path, default=ROOT/'data/processed/fluorescence.csv')
    parser.add_argument('--prepared', type=Path, default=ROOT/'data/processed/baseline_v1/sequences.csv')
    parser.add_argument('--gene', default='cgreGFP')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.external, args.prepared, args.output, args.gene)
