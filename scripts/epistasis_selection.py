import pandas as pd
import numpy as np
import os

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
        "brightness": weighted_mean,
        "brightness_stdev": weighted_stdev
    })

def table_parser(df):
    # Group by aa_genotype
    grouped = df.groupby("aa_genotype")
    result = grouped.apply(weighted_stats).reset_index()

    # Calculate delta from wt and remove it
    wt_row = result.loc[result["aa_genotype"] == "wt"]

    wt_brightness = wt_row["brightness"].values[0]
    wt_brightness_stdev = wt_row["brightness_stdev"].values[0]

    result["brightness"] -= wt_brightness
    result["brightness_stdev"] = np.sqrt(result["brightness_stdev"] ** 2 + wt_brightness_stdev ** 2)

    # Remove rows that are wt or contain 'wt' in the label
    result = result[~result["aa_genotype"].str.contains("wt", case=False, na=False)].reset_index(drop=True)

    # Count number of mutations
    result["num_mutations"] = result["aa_genotype"].apply(lambda x: len(x.split(":")))
    return result