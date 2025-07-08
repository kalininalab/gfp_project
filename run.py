import subprocess
import random
import pandas as pd
import os
import argparse
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="Run CNN model across multiple seeds and selections")
parser.add_argument("--n_seeds", type=int, required=True)
parser.add_argument("--n_selections", type=int, required=True)
parser.add_argument("--min_train_points", type=int, required=True)
parser.add_argument("--start_train_points", type=int, default=19631)
parser.add_argument("--test_points", type=int, default=4908)
args = parser.parse_args()

# Config
train_sets = [
    "cgreGFP",
    "cgreGFP",
    "cgreGFP",
    "cgre1338 cgre132 cgre9708 cgre4111",
    "cgre1338 cgre132 cgre9708 cgre4111",
    "cgre1338 cgre132 cgre9708 cgre4111"
]

test_sets = [
    "cgreGFP",
    "cgre1338 cgre132 cgre9708 cgre4111",
    "cgreGFP cgre1338 cgre132 cgre9708 cgre4111",
    "cgreGFP",
    "cgre1338 cgre132 cgre9708 cgre4111",
    "cgreGFP cgre1338 cgre132 cgre9708 cgre4111"
]

# Run
final_table = []

# Compute selections first (global)
selections = np.linspace(args.start_train_points, args.min_train_points, args.n_selections, dtype=int)

# For each selection → one big CSV per selection
for train_points in selections:
    print(f"\n=== Running SELECTION: train_points={train_points} ===")

    # This will collect all rows for current seeds_XXX.csv
    rows_for_this_selection = []

    for i, (train_genes, test_genes) in enumerate(zip(train_sets, test_sets)):
        print(f"\n--- Running for train: ({train_genes}), test: ({test_genes}) ---")

        seeds = random.sample(range(1000000), args.n_seeds)
        rho_values = []

        for j, seed in enumerate(seeds):
            print(f"    Seed {seed} ({j+1}/{args.n_seeds})")

            subprocess.run([
                "python", "model.py",
                "--train", *train_genes.split(),
                "--test", *test_genes.split(),
                "--seed", str(seed),
                "--output_file", "temp_results.csv",  # temporary, 1 line per run
                "--train_points", str(train_points),
                "--test_points", str(args.test_points)
            ])

            # Read single result line
            df_temp = pd.read_csv("temp_results.csv")
            rho = df_temp.iloc[0]["rho"]
            rho_values.append(rho)

            # Clean temp_results.csv to avoid mixups
            os.remove("temp_results.csv")

        # After all seeds for this train/test pair:
        mean_rho = sum(rho_values) / len(rho_values)
        std_rho = np.std(rho_values)

        # Build row for per-selection table
        df_row = {
            'train': train_genes,
            'test': test_genes
        }
        for k, rho in enumerate(rho_values):
            df_row[f'seed_{k+1}'] = rho
        df_row['final'] = mean_rho
        df_row['std'] = std_rho

        # Add to list of rows for current selection
        rows_for_this_selection.append(df_row)

        # Also add this result to final_results row (one row per train/test pair)
        # Column name will be "train_points"
        # Value is mean_std formatted
        # After processing ALL train/test pairs → save seeds_XXX.csv
        temp_output = f"seeds_{train_points}_train_{args.test_points}_test.csv"
        df_selection = pd.DataFrame(rows_for_this_selection)
        df_selection.to_csv(temp_output, index=False)
    print(f"\nSaved {temp_output}")

    # Now update final_results row
    for df_row in rows_for_this_selection:
        train_genes = df_row['train']
        test_genes = df_row['test']
        # Check if this pair is already in final_table → find matching row
        found = False
        for row in final_table:
            if row['train'] == train_genes and row['test'] == test_genes:
                # Add new column
                row[str(train_points)] = f"{df_row['final']:.4f}_{df_row['std']:.4f}"
                found = True
                break
        if not found:
            # Create new row
            new_row = {
                'train': train_genes,
                'test': test_genes,
                str(train_points): f"{df_row['final']:.4f}_{df_row['std']:.4f}"
            }
            final_table.append(new_row)

# After ALL selections → save final_results.csv
df_final = pd.DataFrame(final_table)
df_final.to_csv("final_results.csv", index=False)
print("\nFinal results saved to final_results.csv")