import pandas as pd

# Load the HiSeqV2 file (tab-separated)
df = pd.read_csv("HiSeqV2", sep="\t")

# First column = gene names
genes = df.iloc[:, 0]
expr = df.iloc[:, 1:]

# Transpose so samples are rows
expr_T = expr.T
expr_T.columns = genes

# Save with sample IDs as first column
expr_T.index.name = "sample_id"
expr_T.to_csv("tcga_expr.csv")
