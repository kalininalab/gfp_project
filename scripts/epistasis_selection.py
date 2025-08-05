import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import FunctionTransformer
from Bio import SeqIO

# Calculate weighted mean and weighted stdev
def weighted_stats(group):
    weights = group["total_cell_count"]
    brightness = group["brightness"]
    stdev = group["brightness_stdev"]

    # Weighted mean
    weighted_mean = np.average(brightness, weights=weights)

    # Weighted standard deviation of the mean:
    # Combine individual standard deviations using weights
    weighted_var = np.sum((stdev ** 2) * (weights ** 2)) / (weights.sum() ** 2)
    weighted_stdev = np.sqrt(weighted_var)
    return pd.Series({
        "aa_genotype": group.name,
        "brightness": weighted_mean,
        "brightness_stdev": weighted_stdev
    })

def table_parser(df):
    # Group by aa_genotype
    grouped = df.groupby("aa_genotype")
    result = grouped.apply(weighted_stats, include_groups=False)
    result.reset_index(drop=True, inplace=True)

    # Calculate delta from wt and remove it
    wt_row = result.loc[result["aa_genotype"] == "wt"]

    if wt_row.empty:
        print("No wild type (wt) genotype found in the data.")
        return {}
    wt_brightness = wt_row["brightness"].values[0]
    wt_brightness_stdev = wt_row["brightness_stdev"].values[0]

    result["brightness"] -= wt_brightness
    result["brightness_stdev"] = np.sqrt(result["brightness_stdev"] ** 2 + wt_brightness_stdev ** 2)

    # Remove rows that are wt or contain 'wt' in the label
    result = result[~result["aa_genotype"].str.contains("wt", case=False, na=False)].reset_index(drop=True)

    # Count number of mutations
    result["num_mutations"] = result["aa_genotype"].apply(lambda x: len(x.split(":")))
    return result


# Recreate mutated seq from wt_seq + aaMutations
def genotype_to_seq(df, wt_seqs, gene_name, verbose=False):
    new_seqs = []
    # genotype to seq
    for index, row in df.iterrows():

        wt_seq=wt_seqs[gene_name]
        new_seq = str(wt_seq)

        if row["aa_genotype"]=="wt":
            aaMutations=[]
        else:
            aaMutations = [(elt[0], int(elt[1:-1]), elt[-1]) for elt in str(row["aa_genotype"]).split(":")]

        if verbose:
            print(new_seq)
            print(len(new_seq))
            print(aaMutations)

        for elt in aaMutations:
            new_seq = new_seq[0:elt[1]]+elt[2]+new_seq[elt[1]+1:]
            if verbose: print("*"*elt[1])

        if verbose:
            print(new_seq)
            print(len(new_seq))
            print("\n\n\n")

        new_seqs.append(new_seq)

    new_seqs = pd.DataFrame(new_seqs,
                            columns=["sequence"],
                            index=df.index)

    return(new_seqs)


directory = "data/raw"
for file in os.listdir(directory):
    if not file.endswith(".csv"):
        continue  # Skip non-CSV files
    if file == "avGFP__rf_nucleotide_genotypes_to_brightness.csv":
        continue # Skip avGFP for now
    file_path = os.path.join(directory, file)
    big_df = pd.read_csv(file_path)
    big_df.rename(columns={"pseudocell_count": "total_cell_count",
                           "aa_genotype_native": "aa_genotype",
                           "replicates_mean_brightness": "brightness",
                           "replicates_stdev_weighted": "brightness_stdev"}, inplace=True)
    proteins = big_df["gene"].unique()
    for protein in proteins:
        df = big_df[big_df["gene"] == protein].copy()
        df = table_parser(df)

        if len(df) == 0:
            print(f"No data for {protein}, skipping.")
            continue

        singles = df[df['num_mutations'] == 1].copy()
        multi = df[df['num_mutations'] > 1].copy()

        # Explicitly create new columns on a copy
        multi["sum_brightness"] = pd.NA
        multi["epistatic"] = pd.NA

        for i, row in multi.iterrows():
            muts = row["aa_genotype"].split(":")
            sum_brightness = 0
            missing = False

            for mut in muts:
                match = singles[singles["aa_genotype"] == mut]
                if match.empty:
                    missing = True
                    break
                sum_brightness += match["brightness"].values[0]

            if missing:
                continue

            multi.loc[i, "sum_brightness"] = sum_brightness

            # Compute sum of standard deviations
            sum_errors = sum(
                singles[singles["aa_genotype"] == mut]["brightness_stdev"].values[0] for mut in muts
            )

            sum_errors += row["brightness_stdev"]
            # Lower bound for sum_brightness
            sum_brightness = max(sum_brightness, min(multi["brightness"]))

            # Apply epistasis condition
            is_epistatic = abs(row["brightness"] - sum_brightness) > sum_errors
            multi.loc[i, "epistatic"] = is_epistatic

        # Save result
        print(protein)
        print("Number of all sequences:", len(multi[multi["epistatic"] == False]))
        print("Number of epistatic sequences:", len(multi[multi["epistatic"] == True]))

        # Get sequence from genotype
        wt_seqs={}
        input_file = 'data/raw/fasta_sequences/protein_seqs.fa'
        fasta_sequences = SeqIO.parse(open(input_file),'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            wt_seqs[name] = sequence

        multi = multi[~multi["aa_genotype"].str.contains("\*")]
        get_seq_from_genotype = FunctionTransformer(genotype_to_seq, kw_args={'wt_seqs': wt_seqs, 'gene_name': protein})
        multi["sequence"] = get_seq_from_genotype.fit_transform(multi)
        multi.to_csv(f"data/processed/epistatic/{protein}.csv", index=False)