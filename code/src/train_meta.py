"""
Train meta-model: market regime features -> optimal blending weight.
Tiny MLP (13 -> 8 -> 1) with TimeSeriesSplit CV to validate generalization.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Features that are actually available and non-constant in meta_train.csv
META_FEATURE_COLS = [
    'trend_score', 'breadth_score', 'accel_decline_score',
    'volatility_score', 'composite',
    'ret_5d', 'ret_10d', 'ret_20d',
    'market_up_ratio_5d', 'market_return_5d',
    'market_volatility_5d', 'consecutive_downs',
]
N_FEATURES = len(META_FEATURE_COLS)


class MarketRegimeMLP(nn.Module):
    def __init__(self, input_dim=N_FEATURES, hidden=8, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_meta_model():
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'model', 'meta_model')
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(os.path.join(data_dir, 'meta_train.csv'))
    X = df[META_FEATURE_COLS].values.astype(np.float32)
    y = df['w_opt_smoothed'].values.astype(np.float32)

    print(f"Training samples: {len(df)}")
    print(f"Features: {META_FEATURE_COLS}")
    print(f"Target: w_opt_smoothed (mean={y.mean():.3f}, std={y.std():.3f})")

    # Naive baseline: always predict mean
    baseline_mse = float(np.mean((y - y.mean()) ** 2))
    baseline_mae = float(np.mean(np.abs(y - y.mean())))
    print(f"\nNaive baseline (predict mean={y.mean():.3f}):")
    print(f"  MSE: {baseline_mse:.4f}")
    print(f"  MAE: {baseline_mae:.4f}")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mses = []
    cv_maes = []

    print(f"\n{'='*50}")
    print("TimeSeriesSplit CV (5 folds)")
    print(f"{'='*50}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Train
        model = MarketRegimeMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        criterion = nn.MSELoss()

        X_tr_t = torch.FloatTensor(X_tr_s).to(device)
        y_tr_t = torch.FloatTensor(y_tr).to(device)
        X_val_t = torch.FloatTensor(X_val_s).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)

        best_val_loss = float('inf')
        best_state = None
        patience = 0

        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t), y_tr_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
                val_mae = float(torch.abs(model(X_val_t) - y_val_t).mean())

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 200:
                    break

        cv_mses.append(best_val_loss)
        cv_maes.append(best_mae)
        fold_dates = f"{df.iloc[val_idx[0]]['date']}..{df.iloc[val_idx[-1]]['date']}"
        print(f"Fold {fold+1}: MSE={best_val_loss:.4f} MAE={best_mae:.4f} "
              f"(val dates: {fold_dates})")

    cv_mse_mean = np.mean(cv_mses)
    cv_mse_std = np.std(cv_mses)
    cv_mae_mean = np.mean(cv_maes)

    print(f"\nCV MSE: {cv_mse_mean:.4f} ± {cv_mse_std:.4f}")
    print(f"CV MAE: {cv_mae_mean:.4f}")
    print(f"Baseline MSE: {baseline_mse:.4f}")
    print(f"Improvement: {(1 - cv_mse_mean/baseline_mse)*100:+.1f}% vs baseline")

    # Train final model on all data
    print(f"\n{'='*50}")
    print("Training final model on all data")
    print(f"{'='*50}")

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = MarketRegimeMLP().to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-2, weight_decay=1e-3)
    criterion = nn.MSELoss()
    X_t = torch.FloatTensor(X_scaled).to(device)
    y_t = torch.FloatTensor(y).to(device)

    for epoch in range(2000):
        final_model.train()
        optimizer.zero_grad()
        loss = criterion(final_model(X_t), y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            final_model.eval()
            with torch.no_grad():
                pred = final_model(X_t)
                mae = float(torch.abs(pred - y_t).mean())
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} mae={mae:.4f}")

    # Save
    torch.save(final_model.state_dict(), os.path.join(model_dir, 'meta_model.pth'))
    joblib.dump(final_scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
    with open(os.path.join(model_dir, 'meta_config.json'), 'w') as f:
        json.dump({
            'feature_cols': META_FEATURE_COLS,
            'input_dim': N_FEATURES,
            'hidden_dim': 8,
            'training_samples': len(df),
            'date_range': [str(df['date'].min()), str(df['date'].max())],
            'cv_mse_mean': float(cv_mse_mean),
            'cv_mse_std': float(cv_mse_std),
            'cv_mae_mean': float(cv_mae_mean),
            'baseline_mse': float(baseline_mse),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {model_dir}")
    print(f"  meta_model.pth ({sum(p.numel() for p in final_model.parameters())} params)")
    print(f"  feature_scaler.pkl")
    print(f"  meta_config.json")
    print("Done!")


if __name__ == '__main__':
    train_meta_model()
