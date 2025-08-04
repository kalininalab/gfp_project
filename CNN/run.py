
# run.py (обновлённый)
import subprocess
import random
import pandas as pd
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_seeds", type=int, required=True)
parser.add_argument("--n_selections", type=int, required=True)
parser.add_argument("--min_train_points", type=int, required=True)
parser.add_argument("--start_train_points", type=int, default=19631)
parser.add_argument("--test_points", type=int, default=4908)
args = parser.parse_args()

train_sets = ["cgreGFP ppluGFP amacGFP"]
test_sets = ["cgreGFP ppluGFP amacGFP"]

selections = np.linspace(args.start_train_points, args.min_train_points, args.n_selections, dtype=int)
final_table = []

for train_points in selections:
    print(f"=== Train points: {train_points} ===")
    rows = []
    for train_genes, test_genes in zip(train_sets, test_sets):
        seeds = random.sample(range(1000000), args.n_seeds)
        rhos = []
        for i, seed in enumerate(seeds):
            print(f"Seed {seed} ({i+1}/{args.n_seeds})")
            try:
                subprocess.run([
                    "python", "model.py",
                    "--train", *train_genes.split(),
                    "--test", *test_genes.split(),
                    "--seed", str(seed),
                    "--output_file", "temp_results.csv",
                    "--train_points", str(train_points),
                    "--test_points", str(args.test_points)
                ], check=True)
                df = pd.read_csv("temp_results.csv")
                rho = df.iloc[0]['rho']
                rhos.append(rho)
                os.remove("temp_results.csv")
            except subprocess.CalledProcessError as e:
                print(f"Failed seed {seed}")
                continue
        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos)
        row = {"train": train_genes, "test": test_genes}
        for k, v in enumerate(rhos):
            row[f"seed_{k+1}"] = v
        row['final'] = mean_rho
        row['std'] = std_rho
        rows.append(row)
    df_block = pd.DataFrame(rows)
    df_block.to_csv(f"results_seeds_{train_points}.csv", index=False)