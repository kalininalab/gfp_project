import os
import re

folder = "esm_embeddings"  # change this if needed

# Pattern to match target files
pattern = re.compile(r"^black_dots_cgreGFP_(.+_ESM_(?:embeddings\.pt|labels\.npy))$")

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        new_name = match.group(1)
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        print(f"Renaming: {filename} → {new_name}")
        os.rename(old_path, new_path)