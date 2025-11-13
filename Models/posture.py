#!/usr/bin/env python3
import argparse
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# -----------------------------------
# Dataset with normalization & augment
# -----------------------------------
class TwoStreamDataset(Dataset):
    def __init__(self, df, belt_scaler=None, grid_scaler=None,
                 augment=False, per_sample_norm=True, rng=None):
        self.augment = augment
        self.per_sample_norm = per_sample_norm
        self.rng = rng or np.random.RandomState(42)

        # Labels and features
        self.y = df['index'].astype(int).values
        belt_cols = [c for c in df if c.startswith('s')]
        grid_cols = [c for c in df if c.startswith('g')]
        B = df[belt_cols].values.reshape(-1, 8, 1)
        G = df[grid_cols].values.reshape(-1, 1, 15, 15)

        # Global scaling fit on training
        if belt_scaler is None:
            self.belt_scaler = StandardScaler().fit(B.reshape(len(B), -1))
        else:
            self.belt_scaler = belt_scaler
        if grid_scaler is None:
            self.grid_scaler = StandardScaler().fit(G.reshape(len(G), -1))
        else:
            self.grid_scaler = grid_scaler

        # Apply global scaling
        B = self.belt_scaler.transform(B.reshape(len(B), -1)).reshape(-1, 8, 1)
        G = self.grid_scaler.transform(G.reshape(len(G), -1)).reshape(-1, 1, 15, 15)

        self.B = torch.tensor(B, dtype=torch.float32)
        self.G = torch.tensor(G, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        b = self.B[idx].clone()
        g = self.G[idx].clone()

        # Per-sample normalization centers grid
        if self.per_sample_norm:
            g = g - g.mean()

        # Augmentation at train time
        if self.augment:
            # random shift up to ±2 pixels
            dx = self.rng.randint(-2, 3)
            dy = self.rng.randint(-2, 3)
            # roll dims 1,2 for H,W
            g = torch.roll(g, shifts=(dx, dy), dims=(1, 2))
            # gaussian noise
            g = g + torch.randn_like(g) * 0.01

        return (b, g), self.y[idx]

# -----------------------------
# Two-stream CNN with dropout
# -----------------------------
class TwoStreamNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # belt branch
        self.belt_net = nn.Sequential(
            nn.Conv1d(8, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        # grid branch
        self.grid_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, belt, grid):
        b = self.belt_net(belt)
        g = self.grid_net(grid)
        x = torch.cat([b, g], dim=1)
        return self.classifier(x)

# ------------------------------------------------
# Training loops for split, cross, and LOSO modes
# ------------------------------------------------
def train_and_eval(train_df, test_df, args, belt_scaler=None, grid_scaler=None):
    # Build datasets
    train_ds = TwoStreamDataset(train_df,
                                belt_scaler=belt_scaler,
                                grid_scaler=grid_scaler,
                                augment=True,
                                per_sample_norm=True)
    # reuse scalers for val/test
    val_ds = TwoStreamDataset(test_df,
                              belt_scaler=train_ds.belt_scaler,
                              grid_scaler=train_ds.grid_scaler,
                              augment=False,
                              per_sample_norm=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(train_ds.y.numpy()))
    model = TwoStreamNet(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    criterion = nn.CrossEntropyLoss()

    best_val, wait = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        for (b, g), y in train_loader:
            b, g, y = b.to(device), g.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(b, g)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for (b, g), y in val_loader:
                b, g = b.to(device), g.to(device)
                p = model(b, g).argmax(1).cpu().numpy()
                preds.extend(p); trues.extend(y.numpy())
        val_acc = accuracy_score(trues, preds)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d} val_acc={val_acc:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val:
            best_val = val_acc; wait = 0; torch.save(model.state_dict(), 'best.pt')
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # final test eval
    model.load_state_dict(torch.load('best.pt'))
    preds, trues = [], []
    with torch.no_grad():
        for (b, g), y in val_loader:
            b, g = b.to(device), g.to(device)
            p = model(b, g).argmax(1).cpu().numpy()
            preds.extend(p); trues.extend(y.numpy())
    print("Classification Report:")
    print(classification_report(trues, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(trues, preds))

    return model, train_ds.belt_scaler, train_ds.grid_scaler

# ------------------------------------------------
# Main function with modes
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # --- MODIFIED THIS LINE ---
    parser.add_argument('--mode', choices=['split','cross'], required=True)
    parser.add_argument('--csv', help='CSV for split or LOSO mode')
    parser.add_argument('--train_csv', help='Train CSV for cross mode')
    parser.add_argument('--test_csv', help='Test CSV for cross mode')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.mode == 'split':
        df = pd.read_csv(args.csv)
        df_tr, df_te = train_test_split(df, test_size=args.test_size,
                                        stratify=df['index'], random_state=args.seed)
        df_train, df_val = train_test_split(df_tr, test_size=args.val_size,
                                            stratify=df_tr['index'], random_state=args.seed)
        train_and_eval(df_train, df_val, args)
    elif args.mode == 'cross':
        df_tr = pd.read_csv(args.train_csv)
        df_te = pd.read_csv(args.test_csv)
        # split train into train/val
        df_train, df_val = train_test_split(df_tr, test_size=args.val_size,
                                            stratify=df_tr['index'], random_state=args.seed)
        train_and_eval(df_train, df_val, args)
        print("Final test on independent set:")
        # evaluate on test set
        train_and_eval(df_tr, df_te, args)
if __name__ == '__main__':
    main()
