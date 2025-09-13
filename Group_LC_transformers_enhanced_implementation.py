
"""
Group_LC_transformers_enhanced_implementation.py

Enhanced implementation of Transformer-style models for credit risk with *sampling* on the
training folds to address class imbalance. Uses SMOTETomek by default.

Models:
  - CNNTransformerClassifier
  - GRUTransformerClassifier

This script mirrors the initial implementation but applies sampling ONLY on the training
split of each CV fold. The validation portion remains untouched to avoid leakage.

It reuses dataset loading + preprocessing from `Group_LC_initial_implementation.py`.
If your initial file exposes a different function name, pass `--entrypoint` or modify
the POSSIBLE_ENTRYPOINTS list in the companion script.

Usage
-----
python Group_LC_transformers_enhanced_implementation.py \
    --folds 5 \
    --epochs 20 \
    --batch-size 512 \
    --sampler smotetomek \
    --models cnn,gru

Author: Luc Kouassi
Date of creation : August 2025
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Sampling
from imblearn.combine import SMOTETomek

# We import model classes and helpers from the initial implementation to keep one source of truth.
from Group_LC_transformers_initial_implementation import (
    set_seed, to_tensor, try_resolve_data, CNNTransformerClassifier, GRUTransformerClassifier, TrainConfig, DEVICE
)

SEED = 42


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = to_tensor(X.astype(np.float32))
    y_t = to_tensor(y.astype(np.float32))
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig
) -> Tuple[nn.Module, Dict[str, float]]:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_auc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            ys = []
            ps = []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                ps.append(prob)
                ys.append(yb.detach().cpu().numpy())
            y_true = np.concatenate(ys).ravel()
            y_prob = np.concatenate(ps).ravel()
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = float("nan")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_auc": float(best_auc)}


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        xb = to_tensor(X.astype(np.float32))
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()
    y_hat = (prob >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "f1": float(f1_score(y, y_hat)),
        "precision": float(precision_score(y, y_hat)),
        "recall": float(recall_score(y, y_hat)),
        "accuracy": float(accuracy_score(y, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
    }
    return metrics


def apply_sampler(X: np.ndarray, y: np.ndarray, sampler_name: str = "smotetomek") -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the chosen sampler on the features and labels.

    Currently supported:
      - 'smotetomek'

    Hooks for GAN/CGAN can be added here if your GAN generators are available.
    """
    name = sampler_name.lower()
    if name == "smotetomek":
        sampler = SMOTETomek(random_state=SEED)
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res

    raise ValueError(f"Unsupported sampler: {sampler_name}")


def cross_validate_with_sampling(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    cfg: TrainConfig,
    sampler_name: str = "smotetomek",
    folds: int = 5,
    **model_kwargs
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    fold_idx = 1
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Apply sampling ONLY on training split
        X_tr_res, y_tr_res = apply_sampler(X_tr, y_tr, sampler_name=sampler_name)

        # model factory
        if model_name == "cnn":
            model = CNNTransformerClassifier(in_features=X.shape[1], **model_kwargs)
        elif model_name == "gru":
            model = GRUTransformerClassifier(in_features=X.shape[1], **model_kwargs)
        else:
            raise ValueError("Unknown model_name: choose from {'cnn','gru'}")

        train_loader = make_loader(X_tr_res, y_tr_res, cfg.batch_size, shuffle=True)
        val_loader = make_loader(X_va, y_va, cfg.batch_size, shuffle=False)
        model, best = train_one_model(model, train_loader, val_loader, cfg)

        # evaluate on validation fold
        fold_metrics = evaluate_model(model, X_va, y_va)
        fold_metrics.update({"fold": fold_idx, "best_val_auc": best["val_auc"], "model": model_name, "sampler": sampler_name})
        rows.append(fold_metrics)
        fold_idx += 1

    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.std(numeric_only=True)
    return df


def maybe_merge_with_previous(new_results: Dict[str, str], previous_csv: Optional[str]) -> Optional[pd.DataFrame]:
    """
    If `previous_csv` is provided and exists, merge previous metrics with the new ones
    for quick comparison.
    """
    if previous_csv is None:
        return None
    if not os.path.exists(previous_csv):
        print(f"[info] previous metrics file not found: {previous_csv}")
        return None
    prev = pd.read_csv(previous_csv)
    frames = [prev]
    for name, path in new_results.items():
        frames.append(pd.read_csv(path))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv("metrics_all_models_comparison.csv", index=False)
    print("Merged metrics saved to metrics_all_models_comparison.csv")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--models", type=str, default="cnn,gru", help="comma list: cnn,gru")
    parser.add_argument("--sampler", type=str, default="smotetomek")
    parser.add_argument("--entrypoint", type=str, default="auto", help="auto or function name in Group_LC_initial_implementation")
    parser.add_argument("--previous-metrics", type=str, default=None, help="CSV of earlier models to merge for comparison")
    args = parser.parse_args()

    set_seed(SEED)
    X_train, X_test, y_train, y_test = try_resolve_data(args.entrypoint)

    # Use all available data for CV. Adjust if you prefer only training set.
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train.ravel(), y_test.ravel()])

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results = {}
    for m in model_list:
        print(f"\n=== CV with sampling for {m.upper()}-Transformer ({args.sampler}) ===")
        df = cross_validate_with_sampling(X, y, m, cfg, sampler_name=args.sampler, folds=args.folds)
        csv_path = f"metrics_{m}_transformer_enhanced_{args.sampler}.csv"
        df.to_csv(csv_path, index=True)
        print(df.tail(3))
        print(f"Saved: {csv_path}")
        all_results[m] = csv_path

    # Optionally merge with previous metrics
    maybe_merge_with_previous(all_results, args.previous_metrics)

    print("\nDone. CSVs saved for each model, plus optional merged comparison.")


if __name__ == "__main__":
    main()
