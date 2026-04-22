from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def evaluate_binary_classifier(
    loader,
    model,
    criterion,
    device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probabilities.cpu().numpy().tolist())

    predictions = [1 if probability >= threshold else 0 for probability in all_probs]
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, predictions) * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "auc": auc * 100,
        "confusion_matrix": confusion_matrix(all_labels, predictions).tolist(),
        "labels": all_labels,
        "probabilities": all_probs,
        "predictions": predictions,
    }

