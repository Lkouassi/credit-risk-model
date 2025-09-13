
"""
Group_LC_transformers_initial_implementation.py  (patched)

Models:
  1) CNNTransformerClassifier
  2) GRUTransformerClassifier

Author : Luc Kouassi
Date of creation : August 2025
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# --- NEW: ensure non-interactive backend so plt.show() doesn't error/warn on import ---
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score, confusion_matrix)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ------------------------
# Utilities
# ------------------------

SEED = 42
rng = np.random.RandomState(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across numpy, random, and torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensure a float32 torch.Tensor on the proper device."""
    if isinstance(array, torch.Tensor):
        t = array
    else:
        t = torch.tensor(array, dtype=torch.float32)
    return t.to(DEVICE)


# ------------------------
# Data access helpers
# ------------------------

POSSIBLE_ENTRYPOINTS: List[str] = [
    # (X_train, X_test, y_train, y_test)
    "get_splits",
    "get_train_test",
    "load_train_test_splits",
    "prepare_train_test",
    # (X, y) or (df, target)
    "load_and_preprocess",
    "prepare_data",
    "load_data",
]


def _as_numpy(x):
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return x
    try:
        # pandas DataFrame/Series -> ndarray
        return x.to_numpy()
    except Exception:
        return _np.asarray(x)


def try_resolve_data(entrypoint: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Attempt to import Group_LC_initial_implementation and retrieve train/test splits.
    First tries functions, then falls back to module-level variables commonly used in your script.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    module_name = "Group_LC_initial_implementation"
    mod = importlib.import_module(module_name)

    # 1) If user provided a function name or we can auto-detect one
    candidates = [entrypoint] if entrypoint and entrypoint != "auto" else POSSIBLE_ENTRYPOINTS
    for fn_name in candidates:
        if not hasattr(mod, fn_name):
            continue
        fn = getattr(mod, fn_name)
        try:
            out = fn() if callable(fn) else None
        except Exception:
            out = None
        if isinstance(out, tuple):
            if len(out) == 4:
                X_train, X_test, y_train, y_test = out
                return _as_numpy(X_train), _as_numpy(X_test), _as_numpy(y_train), _as_numpy(y_test)
            if len(out) == 2:
                X, y = out
                X = _as_numpy(X); y = _as_numpy(y).ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
                return X_train, X_test, y_train, y_test

    # 2) Try dict-like returns from any callable (very permissive)
    for name, obj in inspect.getmembers(mod):
        if callable(obj):
            try:
                out = obj()
                if isinstance(out, dict):
                    keys = set(k.lower() for k in out.keys())
                    if {"x_train", "x_test", "y_train", "y_test"} <= keys:
                        return _as_numpy(out["x_train"]), _as_numpy(out["x_test"]), _as_numpy(out["y_train"]), _as_numpy(out["y_test"])
            except Exception:
                pass

    # 3) NEW: fall back to module-level variables frequently defined in your file
    variable_candidates = [
        ("X_train_scaled", "X_test_scaled", "y_train", "y_test"),
        ("X_train", "X_test", "y_train", "y_test"),
    ]
    for xs, xt, ys, yt in variable_candidates:
        if all(hasattr(mod, v) for v in (xs, xt, ys, yt)):
            X_train = _as_numpy(getattr(mod, xs))
            X_test  = _as_numpy(getattr(mod, xt))
            y_train = _as_numpy(getattr(mod, ys)).ravel()
            y_test  = _as_numpy(getattr(mod, yt)).ravel()
            return X_train, X_test, y_train, y_test

    # 4) Maybe the module exposes pandas DataFrame `df` and `target`
    if hasattr(mod, "df") and hasattr(mod, "target"):
        X = _as_numpy(getattr(mod, "df"))
        y = _as_numpy(getattr(mod, "target")).ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        return X_train, X_test, y_train, y_test

    raise RuntimeError(
        "Could not resolve data from Group_LC_initial_implementation.py. "
        "Please expose a function that returns (X_train, X_test, y_train, y_test) "
        "or ensure module-level arrays X_train_scaled/X_test_scaled/y_train/y_test exist."
    )


# ------------------------
# Model components
# ------------------------

class FeedForward(nn.Module):
    """Simple MLP block used inside Transformer blocks."""
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block tailored for tabular tokens."""
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, int(d_model * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class PatchTokenizer(nn.Module):
    """
    Tokenizes a flat feature vector into a short sequence of 'patch' tokens.
    Uses a 1D convolution over the feature dimension to learn local groupings.

    (B, F) -> (B, T, d_model)
    """
    def __init__(self, in_features: int, d_model: int, patch_size: int = 8, stride: Optional[int] = None):
        super().__init__()
        stride = stride or patch_size
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride)
        self.pos = nn.Parameter(torch.zeros(1, 256, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f = x.shape
        z = x.unsqueeze(1)      # (B,1,F)
        z = self.proj(z)        # (B,d_model,T)
        z = z.transpose(1, 2)   # (B,T,d_model)
        cls = self.cls.expand(b, -1, -1)
        z = torch.cat([cls, z], dim=1)
        pos = self.pos[:, :z.size(1), :]
        z = z + pos
        return z


class CNNTransformerClassifier(nn.Module):
    """CNN + Transformer encoder classifier for tabular data."""
    def __init__(self, in_features: int, d_model: int = 64, depth: int = 2, n_heads: int = 4,
                 mlp_ratio: float = 2.0, patch_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = PatchTokenizer(in_features, d_model, patch_size=patch_size)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tokenizer(x)   # (B,T,d)
        z = self.blocks(z)
        z = self.norm(z[:, 0])
        logit = self.head(z).squeeze(-1)
        return logit


class GRUTransformerClassifier(nn.Module):
    """GRU + Transformer encoder classifier for tabular data."""
    def __init__(self, in_features: int, d_model: int = 64, depth: int = 2, n_heads: int = 4,
                 num_tokens: int = 32, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(in_features, d_model * num_tokens)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True, bidirectional=False)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        z = self.proj(x).view(b, self.num_tokens, -1)  # (B,T,d)
        z, _ = self.gru(z)
        cls = self.cls.expand(b, -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.blocks(z)
        z = self.norm(z[:, 0])
        logit = self.head(z).squeeze(-1)
        return logit


# ------------------------
# Training / evaluation
# ------------------------

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4


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


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    cfg: TrainConfig,
    folds: int = 5,
    **model_kwargs
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    fold_idx = 1
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if model_name == "cnn":
            model = CNNTransformerClassifier(in_features=X.shape[1], **model_kwargs)
        elif model_name == "gru":
            model = GRUTransformerClassifier(in_features=X.shape[1], **model_kwargs)
        else:
            raise ValueError("Unknown model_name: choose from {'cnn','gru'}")

        train_loader = make_loader(X_tr, y_tr, cfg.batch_size, shuffle=True)
        val_loader = make_loader(X_va, y_va, cfg.batch_size, shuffle=False)
        model, best = train_one_model(model, train_loader, val_loader, cfg)

        fold_metrics = evaluate_model(model, X_va, y_va)
        fold_metrics.update({"fold": fold_idx, "best_val_auc": best["val_auc"], "model": model_name})
        rows.append(fold_metrics)
        fold_idx += 1

    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.std(numeric_only=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--models", type=str, default="cnn,gru", help="comma list: cnn,gru")
    parser.add_argument("--entrypoint", type=str, default="auto", help="auto or function name in Group_LC_initial_implementation")
    args = parser.parse_args()

    set_seed(SEED)
    X_train, X_test, y_train, y_test = try_resolve_data(args.entrypoint)

    # Use all available data for CV (train+test). Adjust if you prefer only training set.
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train.ravel(), y_test.ravel()])

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results = {}
    for m in model_list:
        print(f"\n=== Cross-validating {m.upper()}-Transformer ===")
        df = cross_validate(X, y, m, cfg, folds=args.folds)
        csv_path = f"metrics_{m}_transformer_initial.csv"
        df.to_csv(csv_path, index=True)
        print(df.tail(3))
        print(f"Saved: {csv_path}")
        all_results[m] = csv_path

    print("\nFinished. Per-model CSVs saved in the working directory.")
    for k, v in all_results.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
