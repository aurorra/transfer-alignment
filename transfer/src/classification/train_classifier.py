import time
import os
from typing import Tuple, Dict
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptedClassifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2)
        self._device = torch.device('cpu')  # will be updated in .to()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def to(self, device: torch.device) -> "AdaptedClassifier":
        self._device = device
        super().to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device


class AdaptedClassifierAdvanced(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dims: tuple[int, ...] = (),   # e.g., (128, 64)
        dropout: float = 0.0,
        activation: str = 'relu',            # 'relu' | 'gelu' | 'tanh'
        batchnorm: bool = False,
    ):
        super().__init__()
        self._device = torch.device('cpu')

        act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}[activation]

        layers = []
        in_dim = dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = nn.Linear(in_dim, 2)  # binary classification logits

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x)

    def to(self, device: torch.device) -> "AdaptedClassifierAdvanced":
        self._device = device
        super().to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device


def load_dataset(embeddings_path: str, labels_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    X = np.load(embeddings_path)
    y = np.load(labels_path)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_dataloaders(X: torch.Tensor, y: torch.Tensor, batch_size: int, train_frac: float = 0.8, seed: int = 64) \
        -> Tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(X, y)
    n = len(dataset)

    # Stratified train/validation split
    idx = np.arange(n)
    y_np = y.detach().cpu().numpy()
    if not (y_np.ndim == 1 and np.unique(y_np).size >= 2):
        raise ValueError('Stratified split requires 1-D labels with at least two classes.')

    train_idx, val_idx = train_test_split(
        idx,
        train_size=train_frac,
        shuffle=True,
        stratify=y_np,       # force stratification
        random_state=seed      # reproducible split
    )

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    # reproducible shuffling for the train loader
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size)
    return train_loader, val_loader


def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    all_scores = []  # probability/score for class 1
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            logits = model(inputs)

            preds = torch.argmax(logits, dim=1)

            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    # ROC-AUC and Average Precision need both classes to be present.
    # This avoids crashes if a batch/test subset contains only one class.
    try:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_scores)
    except ValueError:
        metrics['roc_auc'] = float('nan')

    try:
        metrics['average_precision'] = average_precision_score(all_labels, all_scores)
    except ValueError:
        metrics['average_precision'] = float('nan')

    return metrics


def train_model(
    model: AdaptedClassifier | AdaptedClassifierAdvanced,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    selection_metric: str,
    lr: float = 1e-4,
) -> Dict[str, float]:
    valid_metrics = {'accuracy', 'precision', 'recall', 'f1'}
    if selection_metric not in valid_metrics:
        raise ValueError(
            f'Unknown selection_metric={selection_metric}. '
            f'Expected one of {sorted(valid_metrics)}.'
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_state = None
    best_score = -float('inf')

    for _ in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        current = evaluate_model(model, val_loader)
        current_score = float(current[selection_metric])

        if current_score > best_score:
            best_score = current_score
            best_metrics = current
            best_state = deepcopy(model.state_dict())

    if best_state is None or best_metrics is None:
        raise RuntimeError('No best model state was selected during training.')

    model.load_state_dict(best_state)
    return best_metrics


def save_metrics_classifier(metrics: Dict[str, float | str], output_path: str, elapsed: float) -> None:
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, 'metrics.txt')
    with open(path, 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"F1: {metrics['f1']}\n")
        f.write(f"Recall: {metrics['recall']}\n")
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Metric used for selecting best weights: {metrics['selection_metric']}\n")
        f.write(f"Runtime: {elapsed:.2f} seconds\n")


def run_classification(embeddings_path: str, labels_path: str, output_path: str, batch_size: int = 64,
                       epochs: int = 10, mode: str = 'train', save_model: bool = False, model_save_path: str = None,
                       model_load_path: str = None, selection_metric: str = 'recall', seed: int = 64) -> Dict[str, float]:
    """
    mode: 'train' or 'predict'
    save_model: if True, dump state_dict to model_save_path after training
    model_load_path: if mode=='predict', load state_dict from here
    """
    X, y = load_dataset(embeddings_path, labels_path)

    if mode == 'train':
        train_loader, val_loader = make_dataloaders(X, y, batch_size, seed=seed)
        model = AdaptedClassifierAdvanced(dim=X.shape[1], hidden_dims=(256, 128), dropout=0.2, activation='gelu').to(device)
        start = time.time()
        best_metrics = train_model(model, train_loader, val_loader, epochs, selection_metric)
        elapsed = time.time() - start

        if save_model:
            if not model_save_path:
                raise ValueError('model_save_path must be provided to save the model.')
            torch.save(model.state_dict(), model_save_path)

    elif mode == 'predict':
        if not model_load_path:
            raise ValueError('model_load_path is required for prediction mode.')
        model = AdaptedClassifierAdvanced(dim=X.shape[1], hidden_dims=(256, 128), dropout=0.2, activation='gelu').to(device)
        state = torch.load(model_load_path, map_location=device)
        model.load_state_dict(state)
        full_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
        start = time.time()
        best_metrics = evaluate_model(model, full_loader)
        elapsed = time.time() - start

    else:
        raise ValueError("mode must be 'train' or 'predict'.")

    best_metrics['selection_metric'] = selection_metric
    save_metrics_classifier(best_metrics, output_path, elapsed)
    print(f'Mode={mode}. Metrics: {best_metrics}. Runtime: {elapsed:.2f}s')
    if save_model and mode == 'train':
        print(f'Model saved to {model_save_path}')

    return best_metrics
