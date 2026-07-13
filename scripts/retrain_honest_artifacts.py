"""Generate the metadata leakage demonstration and honest metadata comparator.

The amplitude/strike_type probe is intentionally leakage-prone and is retained
as a negative result. The lat/lon/time-only probe is the honest comparator.
Neither is the headline satellite result."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def _decode_strings(values):
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return np.array(out)


def _build_features(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:].astype(int)
        lat = f["latitudes"][:].astype(np.float32)
        lon = f["longitudes"][:].astype(np.float32)
        amp = f["amplitudes"][:].astype(np.float32)
        strike_types = _decode_strings(f["strike_types"][:])
        dates = _decode_strings(f["dates"][:])
        train_idx = f["train_indices"][:]
        val_idx = f["val_indices"][:]
        test_idx = f["test_indices"][:]

    strike_map = {"Cloud": 0.0, "Ground": 1.0, "None": 2.0}
    strike_num = np.array([strike_map.get(s, 2.0) for s in strike_types], dtype=np.float32)
    months = np.array([int(d[5:7]) for d in dates], dtype=np.float32)
    doy = np.array(
        [
            int(
                np.datetime64(d).astype("datetime64[D]").astype(int)
                - np.datetime64(d[:4] + "-01-01").astype("datetime64[D]").astype(int)
                + 1
            )
            for d in dates
        ],
        dtype=np.float32,
    )
    season = ((months % 12) // 3).astype(np.float32) / 3.0

    x_meta = np.column_stack(
        [
            (lat - 3.0) / 3.0,
            (lon - 102.0) / 2.5,
            np.clip(amp / 10.0, -1.0, 1.0),
            strike_num,
        ]
    ).astype(np.float32)

    x_clean = np.column_stack(
        [
            (lat - 3.0) / 3.0,
            (lon - 102.0) / 2.5,
            (months - 6.5) / 5.5,
            (doy - 183.0) / 182.0,
            season,
        ]
    ).astype(np.float32)

    return labels, x_meta, x_clean, train_idx, val_idx, test_idx


def _sample_balanced(labels, split_idx, pos_count, neg_count, seed):
    rng = np.random.default_rng(seed)
    y = labels[split_idx]
    pos = split_idx[y == 1]
    neg = split_idx[y == 0]
    pos_count = min(pos_count, len(pos))
    neg_count = min(neg_count, len(neg))
    chosen = np.concatenate(
        [
            rng.choice(pos, size=pos_count, replace=False),
            rng.choice(neg, size=neg_count, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen


class ProbeMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _minority_metrics(y_true, prob_pos, threshold=0.5):
    pred_pos = (prob_pos >= threshold).astype(int)
    y_no = (y_true == 0).astype(int)
    pred_no = (pred_pos == 0).astype(int)
    return {
        "precision_no_strike": float(precision_score(y_no, pred_no, zero_division=0)),
        "recall_no_strike": float(recall_score(y_no, pred_no, zero_division=0)),
        "f1_no_strike": float(f1_score(y_no, pred_no, zero_division=0)),
        "pr_auc_no_strike": float(average_precision_score(y_no, 1.0 - prob_pos)),
        "support_no_strike": int(y_no.sum()),
        "support_strike": int((y_true == 1).sum()),
    }


def _train_probe(name, x, labels, train_s, val_s, test_idx, epochs, batch_size, model_out):
    model = ProbeMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    x_train = torch.from_numpy(x[train_s])
    y_train = torch.from_numpy(labels[train_s]).float().unsqueeze(1)
    x_val = torch.from_numpy(x[val_s])
    y_val = labels[val_s]
    x_test = torch.from_numpy(x[test_idx])
    y_test = labels[test_idx]

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    history = []
    best_val = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            p = model(xb)
            loss = criterion(p, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_prob = model(x_val).numpy().reshape(-1)
        val_m = _minority_metrics(y_val, val_prob)
        val_score = val_m["f1_no_strike"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / max(1, len(loader)),
                **val_m,
            }
        )
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_prob = model(x_test).numpy().reshape(-1)

    test_metrics = _minority_metrics(y_test, test_prob)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)

    return {
        "name": name,
        "checkpoint": str(model_out).replace("\\", "/"),
        "history": history,
        "test_metrics": test_metrics,
    }


def main():
    root = Path(__file__).resolve().parent
    h5_path = root / "data/processed/lightning_dataset.h5"
    out_json = root / "results/metadata_honest_probe_metrics.json"
    model_meta = root / "models/lightning_classifier_metadata_probe.pth"
    model_clean = root / "models/lightning_classifier_clean_probe.pth"

    labels, x_meta, x_clean, train_idx, val_idx, test_idx = _build_features(h5_path)

    train_s = _sample_balanced(labels, train_idx, pos_count=40000, neg_count=40000, seed=42)
    val_s = _sample_balanced(labels, val_idx, pos_count=8000, neg_count=8000, seed=43)

    metadata_run = _train_probe(
        "metadata",
        x_meta,
        labels,
        train_s,
        val_s,
        test_idx,
        epochs=5,
        batch_size=4096,
        model_out=model_meta,
    )
    clean_run = _train_probe(
        "clean",
        x_clean,
        labels,
        train_s,
        val_s,
        test_idx,
        epochs=5,
        batch_size=4096,
        model_out=model_clean,
    )

    always_strike_accuracy = float((labels[test_idx] == 1).mean())
    base = {
        "always_predict_strike_accuracy": always_strike_accuracy,
        "test_no_strike_base_rate": float((labels[test_idx] == 0).mean()),
        "test_support_no_strike": int((labels[test_idx] == 0).sum()),
        "test_support_strike": int((labels[test_idx] == 1).sum()),
    }

    artifact = {
        "artifact_date": "2026-06-26",
        "note": "Probe retrains to compare leakage-prone metadata features versus clean lat/lon/time-only features.",
        "sampling": {
            "train": {"strike": int((labels[train_s] == 1).sum()), "no_strike": int((labels[train_s] == 0).sum())},
            "val": {"strike": int((labels[val_s] == 1).sum()), "no_strike": int((labels[val_s] == 0).sum())},
            "test": {"strike": int((labels[test_idx] == 1).sum()), "no_strike": int((labels[test_idx] == 0).sum())},
        },
        "baseline": base,
        "runs": {
            "metadata": metadata_run,
            "clean": clean_run,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved artifact: {out_json}")
    print("Metadata test metrics:", metadata_run["test_metrics"])
    print("Clean test metrics:", clean_run["test_metrics"])
    print("Baseline:", base)


if __name__ == "__main__":
    main()
