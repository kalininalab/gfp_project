"""Freeze WT-based alignment and balanced-source subsets without changing test sets."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from Bio.Align import PairwiseAligner, substitution_matrices
import Bio
import numpy as np

from scripts.prepare_data import read_fasta
from .common import ROOT,DEFAULT,DATA,PEAKS,NATURAL,GENES,MIXES,MODELS,digest,read_csv,write_csv,scientific_sources


def star_alignment(references):
    """Align WT orthologs to cgre; all substitutions inherit the WT coordinate map."""
    anchor=references['cgreGFP']
    aligner=PairwiseAligner(mode='global',substitution_matrix=substitution_matrices.load('BLOSUM62'),
                          open_gap_score=-10.,extend_gap_score=-.5)
    layouts={}; reports={}
    for gene in NATURAL:
        query=references[gene]
        if gene=='cgreGFP':
            a,b=anchor,anchor;score=float(aligner.score(anchor,anchor));alternatives=1
        else:
            alignments=aligner.align(anchor,query)
            a,b=alignments[0][0],alignments[0][1]
            score=float(alignments.score); alternatives=len(alignments)
        insertions=[[] for _ in range(len(anchor)+1)]; residues=[None]*len(anchor)
        ai=qi=0
        for ca,cb in zip(a,b):
            if ca=='-':
                assert cb!='-'
                insertions[ai].append(qi);qi+=1
            else:
                assert ca==anchor[ai]
                if cb!='-':
                    residues[ai]=qi;qi+=1
                ai+=1
        assert ai==len(anchor) and qi==len(query)
        layouts[gene]=(insertions,residues)
        comparable=[(x,y) for x,y in zip(a,b) if x!='-' and y!='-']
        reports[gene]=dict(score=score,optimal_alignments=alternatives,
                          paired_residues=len(comparable),identity=sum(x==y for x,y in comparable)/len(comparable),
                          pairwise_anchor=a,pairwise_query=b)
    insertion_widths=[max(len(layouts[g][0][i]) for g in NATURAL) for i in range(len(anchor)+1)]
    mappings={}; aligned={}
    for gene in NATURAL:
        ins,res=layouts[gene]; columns=[]
        for i in range(len(anchor)+1):
            columns += ins[i]+[None]*(insertion_widths[i]-len(ins[i]))
            if i<len(anchor):
                columns.append(res[i])
        mapping=[None]*len(references[gene])
        for col,position in enumerate(columns):
            if position is not None:
                mapping[position]=col
        assert None not in mapping and len(set(mapping))==len(mapping)
        mappings[gene]=mapping
        aligned[gene]=''.join('-' if i is None else references[gene][i] for i in columns)
        assert aligned[gene].replace('-','')==references[gene]
    for gene in PEAKS:
        assert len(references[gene])==len(anchor)
        mappings[gene]=mappings['cgreGFP'].copy()
        chars=['-']*len(aligned['cgreGFP'])
        for residue,col in zip(references[gene],mappings[gene]):
            chars[col]=residue
        aligned[gene]=''.join(chars)
    return dict(length=len(aligned['cgreGFP']),mapping=mappings,aligned_wildtypes=aligned,
                method='cgre-anchored WT star alignment; BLOSUM62; global affine gaps -10/-0.5; first optimum; insertions left-aligned',
                biopython=Bio.__version__,pairwise=reports)


def allocate(total,weights):
    exact=np.array(weights,dtype=float)/sum(weights)*total
    counts=np.floor(exact).astype(int)
    order=np.argsort(-(exact-counts),kind='stable')
    counts[order[:total-int(counts.sum())]]+=1
    assert sum(counts)==total
    return counts.tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('Frozen protocol already exists; use a new directory')
    all_rows=[r for r in read_csv(DATA) if r['gene'] in GENES]
    # The audited input has no cross-landscape duplicates in these seven landscapes.
    # Refuse future datasets with ambiguous identities rather than silently leaking them.
    assert len({r['record_id'] for r in all_rows})==len(all_rows)
    assert len({r['sequence'] for r in all_rows})==len(all_rows), 'Cross-landscape sequence duplicate; group-split audit required'
    pools={(g,s):[r for r in all_rows if r['gene']==g and r['split']==s] for g in GENES for s in ['train','validation','test']}
    n_train=len(pools['cgreGFP','train']); n_val=len(pools['cgreGFP','validation'])
    references={g:s.rstrip('*') for g,s in read_fasta(ROOT/'data/raw/fasta_sequences/protein_seqs.fa').items() if g in GENES}
    for r in all_rows:
        assert len(r['sequence'])==len(references[r['gene']])
    alignment=star_alignment(references)
    (out/'alignment.json').write_text(json.dumps(alignment,indent=2)+'\n')
    (out/'aligned_wildtypes.fasta').write_text(''.join(f'>{g}\n{s}\n' for g,s in alignment['aligned_wildtypes'].items()))
    write_csv(out/'alignment_coordinates.csv',[dict(gene=g,native_position=i,aligned_column=c,wildtype_residue=references[g][i])
                                              for g,mp in alignment['mapping'].items() for i,c in enumerate(mp)])
    test_rows=[dict(record_id=r['record_id'],gene=r['gene'],target_group='natural' if r['gene']=='cgreGFP' else 'artificial')
               for r in all_rows if r['split']=='test' and r['gene'] in ['cgreGFP']+PEAKS]
    write_csv(out/'test_records.csv',test_rows)
    jobs=[]; counts=[]; subset_hashes={}
    for mix,genes in MIXES.items():
        for seed in a.seeds:
            selected=[]
            for split,total in [('train',n_train),('validation',n_val)]:
                sizes=[len(pools[g,split]) for g in genes]
                budgets=allocate(total,sizes if mix=='artificial' else [1]*len(genes))
                for gene,budget in zip(genes,budgets):
                    pool=pools[gene,split]
                    assert budget<=len(pool)
                    # Shared within-gene permutation makes subsets nested across mixtures.
                    rng=np.random.default_rng(np.random.SeedSequence([seed,GENES.index(gene),int(split=='validation')]))
                    chosen=sorted(rng.permutation(len(pool))[:budget].tolist())
                    selected += [dict(record_id=pool[i]['record_id'],gene=gene,split=split) for i in chosen]
                    counts.append(dict(mix=mix,seed=seed,gene=gene,split=split,n=budget,
                                       available_in_partition=len(pool),fraction_of_landscape=budget/sum(len(pools[gene,s]) for s in ['train','validation','test'])))
            assert len({r['record_id'] for r in selected})==n_train+n_val
            path=out/'subsets'/f'{mix}_seed{seed}.csv';write_csv(path,selected)
            subset_hashes[str(path.relative_to(out))]=digest(path)
            jobs += [dict(mix=mix,seed=seed,model=model) for model in MODELS]
    write_csv(out/'sample_counts.csv',counts);write_csv(out/'jobs.csv',jobs)
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset_sha256=digest(DATA),
        seeds=a.seeds,models=MODELS,mixes=MIXES,n_train=n_train,n_validation=n_val,
        test_counts=dict(Counter(r['target_group'] for r in test_rows)),
        landscape_counts={g:{s:len(pools[g,s]) for s in ['train','validation','test']} for g in GENES},
        partition_ratios='original per-landscape 60% train / 20% validation / 20% test; subsets drawn only inside these partitions',
        natural_mixture_ratios='equal source contributions: 1:1 or 1:1:1, rounded to fixed budgets',
        artificial_mixture_ratios='proportional to each peak partition size, rounded to fixed budgets',
        model_selection='source-only validation MSE; target-domain test labels never select checkpoints',
        repeats='fixed test; seeds vary initialization/shuffling and subsampling where applicable; error bars are run SD, not CI',
        alignment_sha256=digest(out/'alignment.json'),test_records_sha256=digest(out/'test_records.csv'),
        subset_hashes=subset_hashes,source_hashes=scientific_sources(),
        onehot_settings=dict(epochs=30,patience=10,batch_size=32,lr=.001,optimizer='Adam',loss='MSE'),
        cnn_settings=dict(epochs=60,patience=10,batch_size=64,lr=.0003,weight_decay=.0001,optimizer='AdamW',loss='MSE',target_scaling='training mean/std'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({k:protocol[k] for k in ['n_train','n_validation','test_counts','landscape_counts']},indent=2))
    print('Alignment length:',alignment['length'])


if __name__=='__main__':
    main()
