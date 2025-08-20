import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from Model import ResNet18_1D  # your model file

# ============================================================
# CONFIG - More conservative settings to reduce overfitting
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16  # Smaller batch size
LR = 1e-4  # Lower learning rate
WEIGHT_DECAY = 1e-2  # Higher weight decay for regularization
TOP_GENES = 200  # Fewer features to reduce overfitting
DROPOUT_RATE = 0.7  # Higher dropout
PATIENCE = 10  # More patience for early stopping
MIN_DELTA = 0.001  # Minimum improvement threshold

# ============================================================
# DATA LOADING & PREPROCESSING - Enhanced
# ============================================================
def load_tcga_data(file_path_X, file_path_y):
    print(f"Loading data from {file_path_X} and {file_path_y}...")
    start_time = time.time()

    # Load data
    X = pd.read_csv(file_path_X, index_col=0).astype(np.float32)
    y = pd.read_csv(file_path_y)["label"].astype(int).values

    print(f"Original data shape: {X.shape}")
    print(f"Original label distribution: {np.bincount(y)}")

    # Reindex labels to 0..n_classes-1
    unique_labels = sorted(set(y))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    print(f"Label mapping: {label_map}")
    print(f"New label distribution: {np.bincount(y)}")

    # Log transform with small constant to handle zeros
    X_np = np.log1p(np.maximum(X.values, 1e-8))

    # Remove low-variance genes first
    gene_var = np.var(X_np, axis=0)
    high_var_mask = gene_var > np.percentile(gene_var, 25)  # Keep top 75% by variance
    X_filtered = X_np[:, high_var_mask]
    print(f"After variance filtering: {X_filtered.shape[1]} genes")

    # Robust scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Feature selection: top variable genes from filtered set
    gene_var_filtered = np.var(X_scaled, axis=0)
    top_genes_idx = np.argsort(gene_var_filtered)[-min(TOP_GENES, X_scaled.shape[1]):]
    X_selected = X_scaled[:, top_genes_idx]

    # Additional noise reduction
    X_selected = np.clip(X_selected, -3, 3)  # Clip extreme values

    X_tensor = torch.tensor(X_selected, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    load_time = time.time() - start_time
    print(f"Selected {X_selected.shape[1]} most variable genes from {X_filtered.shape[1]} filtered genes")
    print(f"Data loaded and preprocessed in {load_time:.2f}s")
    print(f"Final shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")

    return X_tensor, y_tensor, scaler

# ============================================================
# DATA AUGMENTATION - Enhanced for class imbalance
# ============================================================
def augment_data(X, y, noise_factor=0.05, target_samples=50):
    """Add gaussian noise for data augmentation, focusing on minority classes"""
    augmented_X = []
    augmented_y = []
    
    # Count samples per class
    unique_classes, class_counts = torch.unique(y, return_counts=True)
    print(f"Original class distribution: {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")
    
    for class_idx in unique_classes:
        class_mask = y == class_idx
        class_samples = X[class_mask]
        class_labels = y[class_mask]
        
        # Add original samples
        augmented_X.extend(class_samples)
        augmented_y.extend(class_labels)
        
        # Calculate augmentation needed for minority classes
        current_count = len(class_samples)
        if current_count < target_samples:
            augment_needed = target_samples - current_count
            print(f"Class {class_idx.item()}: augmenting {augment_needed} samples (from {current_count})")
            
            for _ in range(augment_needed):
                # Randomly select a sample from this class
                idx = torch.randint(0, len(class_samples), (1,)).item()
                base_sample = class_samples[idx]
                
                # Add noise
                noise = torch.randn_like(base_sample) * noise_factor
                augmented_sample = base_sample + noise
                
                augmented_X.append(augmented_sample)
                augmented_y.append(class_idx)
    
    final_X = torch.stack(augmented_X)
    final_y = torch.tensor(augmented_y)
    
    # Print final distribution
    unique_final, final_counts = torch.unique(final_y, return_counts=True)
    print(f"Final class distribution: {dict(zip(unique_final.tolist(), final_counts.tolist()))}")
    
    return final_X, final_y

# ============================================================
# TRAIN / EVAL FUNCTIONS - Enhanced with gradient clipping
# ============================================================
def train_epoch(model, loader, criterion, optimizer, gradient_clip=1.0):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
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
    f1 = f1_score(y_true, y_pred, average="weighted")  # Use weighted F1 for imbalanced classes
    return total_loss / total_samples, acc, f1, y_true, y_pred

# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("=== Enhanced TCGA ResNet Training ===")

    # Load and preprocess data
    X, y, scaler = load_tcga_data("tcga_expr_filtered.csv", "tcga_labels.csv")
    actual_num_genes = X.shape[1]
    num_classes = len(torch.unique(y))
    print(f"Detected {num_classes} unique classes")

    # Train/val/test split with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train label dist: {np.bincount(y_train.numpy())}")
    print(f"Val label dist: {np.bincount(y_val.numpy())}")
    print(f"Test label dist: {np.bincount(y_test.numpy())}")

    # Data augmentation for training set only - focus on minority classes
    print("Applying data augmentation...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_factor=0.03, target_samples=40)
    print(f"Augmented training set size: {len(X_train_aug)}")

    # Weighted sampler for class imbalance
    class_counts = torch.bincount(y_train_aug)
    class_weights = 1. / (class_counts.float() + 1e-8)  # Convert to float and add epsilon
    sample_weights = class_weights[y_train_aug]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_aug, y_train_aug), 
        batch_size=BATCH_SIZE, 
        sampler=sampler
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    # Model with enhanced regularization
    model = ResNet18_1D(
        num_input_features=actual_num_genes, 
        num_classes=num_classes, 
        dropout=DROPOUT_RATE
    )
    model.to(DEVICE)

    # Loss with class weights
    class_weights_tensor = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = torch.optim.AdamW(  # Use AdamW for better regularization
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = get_scheduler(optimizer)

    # Training loop with enhanced early stopping
    best_f1, patience_counter, best_val_loss = 0, 0, float('inf')
    train_losses, val_losses, val_f1s = [], [], []
    
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion)
        
        # Store metrics for analysis
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f}/{train_acc*100:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc*100:.1f}%/F1={val_f1:.3f} | "
              f"LR: {current_lr:.2e}")

        # Enhanced early stopping
        improved = False
        if val_f1 > best_f1 + MIN_DELTA:
            best_f1 = val_f1
            torch.save(model.state_dict(), "resnet_tcga_best.pth")
            print(f"  ✓ Best model saved (F1: {val_f1:.3f})")
            patience_counter = 0
            improved = True
        
        if not improved:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # Stop if learning rate becomes too small
        if current_lr < 1e-6:
            print(f"  Learning rate too small, stopping at epoch {epoch}")
            break

    # Test with best model
    print("\n=== FINAL EVALUATION ===")
    model.load_state_dict(torch.load("resnet_tcga_best.pth"))
    test_loss, test_acc, test_f1, y_true, y_pred = eval_epoch(model, test_loader, criterion)
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[f"Class_{i}" for i in range(num_classes)]))
    
    # Training analysis
    print(f"\nTraining Analysis:")
    print(f"  Best validation F1: {best_f1:.4f}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")
    print(f"  Overfitting ratio: {val_losses[-1]/train_losses[-1]:.2f}")
    
    if val_losses[-1]/train_losses[-1] > 3:
        print("  ⚠️  High overfitting detected!")
    else:
        print("  ✓  Overfitting under control")