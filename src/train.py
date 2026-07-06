"""Minimal compatibility wrapper for the legacy training helpers used by older tests."""

from __future__ import annotations

import numpy as np
import torch


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(images).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(train_loader))


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in val_loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device).float().view(-1, 1)
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, batch_labels)
            total_loss += float(loss.item())
            preds.append(outputs.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    return total_loss / max(1, len(val_loader)), np.concatenate(preds), np.concatenate(labels)
