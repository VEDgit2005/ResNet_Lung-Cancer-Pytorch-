# explain.py
import torch
import pandas as pd
from captum.attr import IntegratedGradients
from model import ResMLP
from dataset import load_from_csv, ExpressionDataset

def explain(expr_csv, label_csv, checkpoint_path, target_class=1, topk=50):
    expr_df, labels = load_from_csv(expr_csv, label_csv)
    ds = ExpressionDataset(expr_df, labels)
    model = ResMLP(input_dim=expr_df.shape[1], hidden_dim=512, n_blocks=4, n_classes=2)
    ck = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    ig = IntegratedGradients(model)
    # pick a sample to explain (example: first sample)
    x, y = ds[0]
    x = x.unsqueeze(0)  # batch
    attributions = ig.attribute(x, target=target_class, n_steps=50)
    at = attributions.squeeze().detach().numpy()
    genes = expr_df.columns.tolist()
    df = pd.DataFrame({'gene':genes, 'importance': at})
    df = df.reindex(df.importance.abs().sort_values(ascending=False).index)
    return df.head(topk)

if __name__ == "__main__":
    print(explain('data/expr.csv','data/labels.csv','best_checkpoint.pth').head(30))
