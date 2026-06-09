"""
Sign Language Translation Pipeline:
1. Data Extraction: Hand landmarks are extracted from raw images/video using MediaPipe Tasks API.
2. Normalization & Translation: Coordinates are centered relative to the wrist and normalized.
3. Skeletal Rendering: Normalized coordinates are drawn as a skeleton on a blank 224x224 image.
4. Deep Learning Classification: The skeleton image is passed through a pre-trained CNN backbone 
   (ResNet18 or EfficientNet-B0) to classify the hand gesture into a sign language character.
5. Evaluation: Generates training metrics, ROC curves, confusion matrices, and fit analysis.
"""

import os
import pandas as pd
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE    = './data/hand_data_clean.csv'
MODEL_FILE   = './model/sign_dl_model.pth'
CLASSES_FILE = './model/classes.p'
RESULTS_FILE = './results_table.md'
SAMPLE_LIMIT = 100   # samples per class used for training
EPOCHS       = 10    # epochs for every experiment

# ── Fixed MediaPipe hand connections ─────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# ── Dataset ───────────────────────────────────────────────────────────────────
class SkeletonDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        label    = self.y[idx]

        img    = np.zeros((224, 224, 3), dtype=np.uint8)
        points = []
        for i in range(21):
            x_val = features[i * 2]
            y_val = features[i * 2 + 1]
            x = int(((x_val + 1) / 2.0) * 224 * 0.8 + 224 * 0.1)
            y = int(((y_val + 1) / 2.0) * 224 * 0.8 + 224 * 0.1)
            x = max(0, min(223, x))
            y = max(0, min(223, y))
            points.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(img, points[p1], points[p2], (255, 255, 255), 2)

        if self.transform:
            img = self.transform(img)
        return img, label

# ── Model factory ─────────────────────────────────────────────────────────────
def get_model(num_classes, backbone_name='resnet18'):
    if backbone_name == 'resnet18':
        model    = models.resnet18(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone_name == 'efficientnet_b0':
        model    = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        model    = models.resnet18(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ── Training loop ─────────────────────────────────────────────────────────────
def train_experiment(X_train, y_train, X_test, y_test,
                     num_classes, lr, batch_size, backbone_name, epochs):
    print(f"\n--- Experiment: {backbone_name} | LR={lr} | BS={batch_size} | Epochs={epochs} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(SkeletonDataset(X_train, y_train, transform),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(SkeletonDataset(X_test,  y_test,  transform),
                              batch_size=batch_size, shuffle=False)

    model     = get_model(num_classes, backbone_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs  = model(inputs)
                loss     = criterion(outputs, labels)
                probs    = F.softmax(outputs, dim=1)

                val_loss_sum += loss.item()
                _, predicted  = torch.max(outputs.data, 1)
                val_total    += labels.size(0)
                val_correct  += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = val_loss_sum / len(test_loader)
        val_acc  = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Epoch [{epoch+1:2d}/{epochs}] "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        'accuracy':    accuracy_score(y_true, y_pred),
        'precision':   precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':      recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1':          f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs':   train_accs,   'val_accs':   val_accs,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
        'model':  model,
    }

# ── Overfitting / Underfitting analysis ───────────────────────────────────────
def analyse_fit(train_accs, val_accs, exp_name):
    final_train = train_accs[-1]
    final_val   = val_accs[-1]
    gap         = final_train - final_val

    print(f"\n  [Fit Analysis] {exp_name}")
    print(f"    Final Train Acc : {final_train:.4f}")
    print(f"    Final Val Acc   : {final_val:.4f}")
    print(f"    Gap (Train-Val) : {gap:.4f}")

    if final_train < 0.70 and final_val < 0.70:
        verdict = "UNDERFITTING — both train and val accuracy are low. " \
                  "Try more epochs, a larger model, or more data."
    elif gap > 0.15:
        verdict = "OVERFITTING — large gap between train and val accuracy. " \
                  "Try dropout, weight decay, data augmentation, or more data."
    elif gap > 0.05:
        verdict = "SLIGHT OVERFITTING — minor gap. Model is generalising " \
                  "reasonably well but could benefit from regularisation."
    else:
        verdict = "GOOD FIT — train and val accuracy are close and both high. " \
                  "Model is generalising well."

    print(f"    Conclusion      : {verdict}")
    return verdict

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_metrics(metrics, classes, exp_name):
    epochs_range = range(1, len(metrics['train_losses']) + 1)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {exp_name}", fontsize=13)

    axes[0].plot(epochs_range, metrics['train_losses'], label='Train Loss',  marker='o')
    axes[0].plot(epochs_range, metrics['val_losses'],   label='Val Loss',    marker='s')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Epochs'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_range, metrics['train_accs'], label='Train Accuracy', marker='o')
    axes[1].plot(epochs_range, metrics['val_accs'],   label='Val Accuracy',   marker='s')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Epochs'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('./plots/training_curves.png', dpi=120)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes) - 2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix — {exp_name}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('./plots/confusion_matrix.png', dpi=120)
    plt.close()

    # ROC Curve (one-vs-rest, macro average)
    n_classes  = len(classes)
    y_bin      = label_binarize(metrics['y_true'], classes=list(range(n_classes)))
    y_prob     = metrics['y_prob']

    plt.figure(figsize=(10, 7))
    all_fpr, all_tpr = [], []

    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:          # skip classes absent in test set
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.5, lw=1.2, label=f'{cls} (AUC={roc_auc:.2f})')
        all_fpr.extend(fpr); all_tpr.extend(tpr)

    # Macro average
    fpr_all = np.linspace(0, 1, 100)
    # simple: interpolate each class
    tpr_avg = np.zeros(100)
    count   = 0
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        tpr_avg += np.interp(fpr_all, fpr_i, tpr_i)
        count   += 1
    if count:
        tpr_avg /= count
        macro_auc = auc(fpr_all, tpr_avg)
        plt.plot(fpr_all, tpr_avg, 'k--', lw=2.5,
                 label=f'Macro Avg (AUC={macro_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'r:', lw=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (One-vs-Rest) — {exp_name}')
    plt.legend(loc='lower right', fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./plots/roc_curve.png', dpi=120)
    plt.close()
    print("  Saved: training_curves.png, confusion_matrix.png, roc_curve.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    # Sub-sample: up to SAMPLE_LIMIT rows per class
    np.random.seed(42)
    sampled = []
    for lv in df.iloc[:, 0].unique():
        idx = df.index[df.iloc[:, 0] == lv].tolist()
        if len(idx) > SAMPLE_LIMIT:
            idx = np.random.choice(idx, SAMPLE_LIMIT, replace=False).tolist()
        sampled.extend(idx)
    df = df.iloc[sampled].reset_index(drop=True)

    labels         = df.iloc[:, 0].values
    X              = df.iloc[:, 1:].values.astype(np.float32)
    unique_classes = sorted(set(labels))
    class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
    y              = np.array([class_to_idx[l] for l in labels])

    print(f"Dataset: {len(df)} samples across {len(unique_classes)} classes")
    print(f"Classes: {unique_classes}\n")

    with open(CLASSES_FILE, 'wb') as f:
        pickle.dump(unique_classes, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Experiments ──────────────────────────────────────────────────────────
    experiments = [
        {'lr': 1e-3,  'batch_size': 32, 'backbone': 'resnet18',        'epochs': EPOCHS},
        {'lr': 1e-4,  'batch_size': 32, 'backbone': 'resnet18',        'epochs': EPOCHS},
        {'lr': 1e-3,  'batch_size': 64, 'backbone': 'efficientnet_b0', 'epochs': EPOCHS},
    ]

    results          = []
    best_acc         = 0.0
    best_model       = None
    best_metrics     = None
    best_backbone    = ""
    fit_conclusions  = []

    for i, exp in enumerate(experiments):
        m = train_experiment(
            X_train, y_train, X_test, y_test,
            len(unique_classes),
            exp['lr'], exp['batch_size'], exp['backbone'], exp['epochs'],
        )

        exp_name = f"Exp{i+1} {exp['backbone']} lr={exp['lr']} bs={exp['batch_size']}"
        verdict  = analyse_fit(m['train_accs'], m['val_accs'], exp_name)
        fit_conclusions.append((exp_name, verdict))

        results.append({
            'Experiment':    i + 1,
            'Backbone':      exp['backbone'],
            'Learning Rate': exp['lr'],
            'Batch Size':    exp['batch_size'],
            'Epochs':        exp['epochs'],
            'Accuracy':      f"{m['accuracy']:.4f}",
            'Precision':     f"{m['precision']:.4f}",
            'Recall':        f"{m['recall']:.4f}",
            'F1 Score':      f"{m['f1']:.4f}",
        })

        if m['accuracy'] > best_acc:
            best_acc      = m['accuracy']
            best_model    = m['model']
            best_metrics  = m
            best_backbone = exp['backbone']

    # ── Save best model ───────────────────────────────────────────────────────
    torch.save(best_model.state_dict(), MODEL_FILE)
    with open('./model/best_backbone.p', 'wb') as f:
        pickle.dump(best_backbone, f)
    print(f"\nBest model: {best_backbone}  (Val Acc={best_acc:.4f})  -> {MODEL_FILE}")

    # ── Plots for best model ──────────────────────────────────────────────────
    plot_metrics(best_metrics, unique_classes,
                 f"Best Model ({best_backbone})")

    # ── Results table  ────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write("# Training Experiments — Results Table\n\n")
        f.write(f"**Dataset:** {len(df)} samples, {len(unique_classes)} classes, "
                f"{SAMPLE_LIMIT} samples/class limit, {EPOCHS} epochs\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n\n## Overfitting / Underfitting Analysis\n\n")
        for name, verdict in fit_conclusions:
            f.write(f"**{name}**\n> {verdict}\n\n")

    print(f"Results written to {RESULTS_FILE}")
    print("\nDone.")