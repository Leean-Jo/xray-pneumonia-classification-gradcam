from typing import Dict

import torch
from tqdm import tqdm

from xray_cls.utils.metrics import compute_classification_metrics


class Trainer:
    def __init__(self, model, criterion, optimizer=None, device: str = "cuda"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        for images, labels, _ in tqdm(dataloader, desc="Train", leave=False):
            images = images.to(self.device)
            labels = labels.unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_targets.extend(labels.detach().cpu().numpy().ravel().tolist())
            all_probs.extend(probs.detach().cpu().numpy().ravel().tolist())

        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = compute_classification_metrics(all_targets, all_probs)
        metrics["loss"] = epoch_loss
        return metrics

    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        for images, labels, _ in tqdm(dataloader, desc="Valid", leave=False):
            images = images.to(self.device)
            labels = labels.unsqueeze(1).to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_targets.extend(labels.detach().cpu().numpy().ravel().tolist())
            all_probs.extend(probs.detach().cpu().numpy().ravel().tolist())

        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = compute_classification_metrics(all_targets, all_probs)
        metrics["loss"] = epoch_loss
        return metrics