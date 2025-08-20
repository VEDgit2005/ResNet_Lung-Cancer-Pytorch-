''' THIS CODE HELPS TO CHECK IF THE CSV FILE IS READY TO BE TRAINED AND THE TRAIN>PY WONT GIVE 
ANY ERRORS'''
import pandas as pd

# Load both files
expr_df = pd.read_csv("tcga_expr.csv", index_col=0)
labels_df = pd.read_csv("tcga_labels.csv")

# Normalize barcodes in expression file
expr_df.index = expr_df.index.str[:15].str.upper()

# Keep only samples present in labels_df
expr_df = expr_df.loc[expr_df.index.isin(labels_df["sample_id"])]

# Optional: sort to match labels_df order
expr_df = expr_df.loc[labels_df["sample_id"]]

# Save filtered expression data
expr_df.to_csv("tcga_expr_filtered.csv")
print(f"Saved filtered expression data with {expr_df.shape[0]} samples and {expr_df.shape[1]} genes.")
