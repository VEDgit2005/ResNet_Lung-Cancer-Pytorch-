import pandas as pd
import re

# Load data
expr_df = pd.read_csv("tcga_expr.csv", index_col=0)
immune_df = pd.read_csv("Subtype_Immune_Model_Based.txt", sep="\t", index_col=0)

# Normalize barcodes
expr_df.index = expr_df.index.str[:15].str.upper()
immune_df.index = immune_df.index.str[:15].str.upper()

# Merge immune subtype with expression sample list
labels_df = pd.DataFrame({"sample_id": expr_df.index.unique()})
labels_df = labels_df.merge(
    immune_df,
    left_on="sample_id",
    right_index=True,
    how="inner"
)

# Extract just the C1–C6 code from the string
labels_df["subtype_code"] = labels_df["Subtype_Immune_Model_Based"].apply(
    lambda x: re.search(r"C\d", x).group(0) if pd.notnull(x) else None
)

# Map C1–C6 to integers
code_to_int = {f"C{i}": i-1 for i in range(1, 7)}  # C1=0, C2=1, ...
labels_df["label"] = labels_df["subtype_code"].map(code_to_int)

# Keep only sample_id and label
labels_df = labels_df[["sample_id", "label"]]

# Save cleaned labels
labels_df.to_csv("tcga_labels.csv", index=False)
print("Saved numeric labels to tcga_labels.csv")
print(labels_df.head())
