import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from Model import ResNet18_1D  # your model file

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
TOP_GENES = 500  # reduce features from 1000 → 500
DROPOUT_RATE = 0.5
PATIENCE = 5

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
def load_tcga_data(file_path_X, file_path_y):
    print(f"Loading data from {file_path_X} and {file_path_y}...")
    start_time = time.time()

    # Load data
    X = pd.read_csv(file_path_X, index_col=0).astype(np.float32)
    y = pd.read_csv(file_path_y)["label"].astype(int).values

    # Reindex labels to 0..n_classes-1
    unique_labels = sorted(set(y))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    print(f"Label mapping: {label_map}")

    # Log transform
    X_np = np.log1p(np.maximum(X.values, 0))

    # Standardization
    X_mean, X_std = np.mean(X_np, axis=0), np.std(X_np, axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X_np - X_mean) / X_std

    # Feature selection: top variable genes
    gene_var = np.var(X_scaled, axis=0)
    top_genes_idx = np.argsort(gene_var)[-min(TOP_GENES, X_scaled.shape[1]):]
    X_selected = X_scaled[:, top_genes_idx]

    X_tensor = torch.tensor(X_selected, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    load_time = time.time() - start_time
    print(f"Selected {X_selected.shape[1]} most variable genes from {X_scaled.shape[1]} total genes")
    print(f"Data loaded and preprocessed in {load_time:.2f}s")
    print(f"Final shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")

    return X_tensor, y_tensor

# ============================================================
# TRAIN / EVAL FUNCTIONS
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
        total_samples += X_batch.size(0)

    return total_loss / total_samples, total_correct / total_samples

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch.unsqueeze(1))
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total_samples += X_batch.size(0)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    acc = total_correct / total_samples
    f1 = f1_score(y_true, y_pred, average="macro")
    return total_loss / total_samples, acc, f1

# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("=== TCGA ResNet Training ===")

    # Load and preprocess data
    X, y = load_tcga_data("tcga_expr_filtered.csv", "tcga_labels.csv")
    actual_num_genes = X.shape[1]  # number of features after preprocessing
    num_classes = len(torch.unique(y))
    print(f"Detected {num_classes} unique classes")

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Weighted sampler for class imbalance
    class_counts = np.bincount(y_train.numpy())
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train.numpy()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Model (with dropout)
    model = ResNet18_1D(num_input_features=actual_num_genes, num_classes=num_classes, dropout=DROPOUT_RATE)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop with early stopping
    best_f1, patience_counter = 0, 0
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train: {train_loss:.3f}/{train_acc*100:.1f}% | "
              f"Val: {val_loss:.3f}/{val_acc*100:.1f}%/F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "resnet_tcga_best.pth")
            print(f"  ✓ Best model saved (F1: {val_f1:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test with best model
    model.load_state_dict(torch.load("resnet_tcga_best.pth"))
    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc*100:.2f}%, Test F1: {test_f1:.4f}")
