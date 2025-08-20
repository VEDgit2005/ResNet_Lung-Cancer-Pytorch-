import pandas as pd

# Load your files
expr_df = pd.read_csv("tcga_expr.csv", index_col=0)
immune_df = pd.read_csv("Subtype_Immune_Model_Based.txt", sep="\t", index_col=0)

# Normalize barcodes to first 15 chars
expr_df.index = expr_df.index.str[:15].str.upper()
immune_df.index = immune_df.index.str[:15].str.upper()

# Prepare labels dataframe from expression samples
labels_df = pd.DataFrame({"sample_id": expr_df.index.unique()})

# Merge immune subtype info
labels_df = labels_df.merge(
    immune_df,
    left_on="sample_id",
    right_index=True,
    how="inner"
)

# Check
print(f"Matched samples: {len(labels_df)} out of {len(expr_df.index.unique())}")
print(labels_df.head())

# Rename columns
labels_df.columns = ["sample_id", "label"]

# Save
labels_df.to_csv("tcga_labels.csv", index=False)
print("tcga_labels.csv saved")
