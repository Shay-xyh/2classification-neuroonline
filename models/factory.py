"""CBraMod model adapter and checkpoint helpers."""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.cbramod import CBraModClassifier

LOGGER = logging.getLogger(__name__)
DEFAULT_CBRAMOD_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "pretrained"
    / "cbramod_pretrained_weights.pth"
)


def atomic_torch_save(payload: object, path: Path) -> None:
    """Write a torch checkpoint completely before promoting it into place."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def enable_mc_dropout(module: nn.Module) -> None:
    """Enable stochastic dropout without updating normalization statistics."""

    module.eval()
    for child in module.modules():
        if isinstance(child, (nn.modules.dropout._DropoutNd, nn.MultiheadAttention)):
            child.train()


def split_train_validation_indices(
    y: np.ndarray,
    *,
    groups: np.ndarray | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by trial when group IDs are available, otherwise by window."""

    labels = np.asarray(y, dtype=np.int64)
    indices = np.arange(labels.shape[0])
    if groups is not None:
        group_ids = np.asarray(groups, dtype=np.int64)
        if group_ids.shape != labels.shape:
            raise ValueError(
                f"groups must match labels shape {labels.shape}, got {group_ids.shape}."
            )
        unique_groups = np.unique(group_ids)
        group_labels_list: list[int] = []
        for group in unique_groups:
            labels_in_group = np.unique(labels[group_ids == group])
            if labels_in_group.size != 1:
                raise ValueError(
                    f"Trial group {int(group)} contains multiple labels: "
                    f"{labels_in_group.tolist()}."
                )
            group_labels_list.append(int(labels_in_group[0]))
        group_labels = np.asarray(group_labels_list, dtype=np.int64)
        groups_per_class = [
            int(np.sum(group_labels == label))
            for label in np.unique(group_labels)
        ]
        n_splits = min(5, min(groups_per_class, default=0))
        if n_splits < 2:
            counts = {
                int(label): int(count)
                for label, count in zip(
                    np.unique(group_labels),
                    groups_per_class,
                    strict=True,
                )
            }
            raise ValueError(
                "Trial-group validation requires at least two independent "
                f"groups per observed class; got {counts}."
            )
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        train_indices, validation_indices = next(
            splitter.split(indices, labels, groups=group_ids)
        )
        return train_indices, validation_indices

    from sklearn.model_selection import train_test_split

    return train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )


class BaseModelAdapter(ABC):
    """Common interface for all training and inference backends."""

    model_name: str

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        head_only: bool = False,
        groups: np.ndarray | None = None,
        progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Train the model and return summary metrics."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray, mc_dropout_passes: int = 1) -> np.ndarray:
        """Predict class probabilities for one or more windows."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load persisted weights from disk."""

class TorchModelAdapter(BaseModelAdapter):
    """Simple training wrapper around PyTorch-based EEG models."""

    def __init__(self, model_name: str, model: nn.Module) -> None:
        self.model_name = model_name
        self.model = model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        head_only: bool = False,
        groups: np.ndarray | None = None,
        progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        self._configure_trainable_layers(head_only=head_only)
        train_indices, validation_indices = split_train_validation_indices(
            y,
            groups=groups,
            random_state=42,
        )
        X_train, X_val = X[train_indices], X[validation_indices]
        y_train, y_val = y[train_indices], y[validation_indices]
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_inputs = torch.tensor(X_val, dtype=torch.float32, device=self._device)
        val_targets = torch.tensor(y_val, dtype=torch.long, device=self._device)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        best_state = None
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self._device)
                batch_targets = batch_targets.to(self._device)
                optimizer.zero_grad()
                logits = self.model(batch_inputs)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_inputs)
                val_loss = float(criterion(val_logits, val_targets).item())
                val_predictions = torch.argmax(val_logits, dim=1)
                val_acc = float((val_predictions == val_targets).float().mean().item())

            LOGGER.info(
                "Epoch %s/%s train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1,
                epochs,
                train_loss / max(len(loader), 1),
                val_loss,
                val_acc,
            )
            if progress_callback is not None:
                progress_callback(
                    epoch + 1,
                    epochs,
                    {
                        "train_loss": train_loss / max(len(loader), 1),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                )
            if best_state is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "epochs_completed": float(epochs),
            "best_epoch": float(best_epoch),
        }

    def predict_proba(self, X: np.ndarray, mc_dropout_passes: int = 1) -> np.ndarray:
        inputs = torch.tensor(X, dtype=torch.float32, device=self._device)
        passes = max(mc_dropout_passes, 1)
        outputs: list[np.ndarray] = []
        if passes > 1:
            enable_mc_dropout(self.model)
        else:
            self.model.eval()
        for _ in range(passes):
            with torch.no_grad():
                logits = self.model(inputs)
                probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            outputs.append(probabilities)
        return np.mean(np.stack(outputs, axis=0), axis=0)

    def save(self, path: Path) -> None:
        atomic_torch_save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self._device)

    def _configure_trainable_layers(self, head_only: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = not head_only
        if not head_only:
            return

        classifier = getattr(self.model, "classifier", None)
        if isinstance(classifier, nn.Module):
            for parameter in classifier.parameters():
                parameter.requires_grad = True
            return

        last_trainable: nn.Module | None = None
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                last_trainable = module
        if last_trainable is not None:
            for parameter in last_trainable.parameters():
                parameter.requires_grad = True


class ModelFactory:
    """Model registry for all built-in motor imagery decoders."""

    @staticmethod
    def get(
        model_name: str,
        n_chans: int,
        sfreq: float,
        n_classes: int = 2,
        n_times: int | None = None,
    ) -> BaseModelAdapter:
        n_times = n_times or int(sfreq * 4.0)

        if model_name == "cbramod":
            return TorchModelAdapter(
                model_name,
                CBraModClassifier(
                    n_chans=n_chans,
                    n_times=n_times,
                    n_classes=n_classes,
                    sfreq=sfreq,
                    pretrained_path=DEFAULT_CBRAMOD_WEIGHTS,
                ),
            )
        raise ValueError(
            "Unknown model '%s'. Available deployment model: cbramod" % model_name
        )

    @staticmethod
    def list_models() -> list[str]:
        return ["cbramod"]
