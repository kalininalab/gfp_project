"""Freeze the five-seed, four-model transfer-AL protocol."""
import argparse,datetime,json
from pathlib import Path
from scripts.transfer_benchmark.common import DATA,MODELS,MIXES,digest,read_csv,write_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=Path('results/transfer_cgreGFP_seed42_46_runs'));p.add_argument('--output',type=Path,default=Path('results/transfer_al_cgreGFP_seed42_46'));a=p.parse_args()
 if (a.output/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
 ref=json.loads((a.reference/'protocol.json').read_text());seeds=list(range(42,47))
 if not set(seeds)<=set(ref['seeds']):raise ValueError('Reference protocol lacks requested seeds')
 publication=Path('results/transfer_cgreGFP_seed42_46/test_records.csv')
 if digest(publication)!=ref['test_records_sha256']:raise ValueError('Publication/reference test manifest mismatch')
 a.output.mkdir(parents=True,exist_ok=True);write_csv(a.output/'test_records.csv',read_csv(publication))
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reference_directory':str(a.reference),
  'reference_protocol_sha256':digest(a.reference/'protocol.json'),'dataset_sha256':digest(DATA),'test_records_sha256':digest(a.output/'test_records.csv'),
  'seeds':seeds,'models':MODELS,'mixes':ref['mixes'],'rounds':10,'initial_fraction_of_final_budget':.1,
  'final_training_budget':ref['n_train'],'validation_count':ref['n_validation'],
  'acquisition':'0.62 * scaled projected-penultimate distance + 0.38 * scaled MC-dropout variance',
  'uncertainty_samples':25,'projection_dimensions':32,'distance_anchors':256,
  'selection':'global descending acquisition score; no dense spectral clustering at full-pool scale',
  'round_fitting':'from-scratch model with the same architecture, optimizer, validation checkpointing and maximum epochs as non-AL',
  'test_policy':'exact frozen test_records.csv; unavailable to training, validation, acquisition and checkpoint selection',
  'onehot_settings':ref['onehot_settings'],'cnn_settings':ref['cnn_settings']}
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 write_csv(a.output/'jobs.csv',[{'mix':mix,'seed':seed,'model':model} for mix in MIXES for seed in seeds for model in MODELS])
 print(f'Frozen {len(MIXES)*len(seeds)*len(MODELS)} transfer-AL trajectories')
if __name__=='__main__':main()
