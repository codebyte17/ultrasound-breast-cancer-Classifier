from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple,List
from dataclasses import dataclass
from transformers import get_cosine_schedule_with_warmup
from src import data_loader
from src import CustomCnnModel
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


@dataclass
class History:
    train_losses: List[float]
    val_losses: List[float]
    train_accs: List[float]
    val_accs: List[float]

class Trainer:
    def __init__(self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        scheduler,
        loss_fn : nn.Module,):

        self.epoch = 0
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.history = History([],[],[],[])

    # -------------------------
    # Factory (classmethod)
    # -------------------------
    @classmethod
    def from_config(cls,
                    config: object,
                    model: nn.Module,
                    device: torch.device,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    loss_fn: nn.Module
                    ) -> Trainer:

        lr = float(config.lr)
        weight_decay = float(config.weight_decay)
        warmup_epochs = int(config.warmup_epochs)

        # Note: scheduler needs total steps and warmup steps
        num_training_steps = int(config.epochs) * len(train_loader)
        num_warmup_steps = warmup_epochs * len(train_loader)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return cls(model, device,optimizer ,train_loader, test_loader, scheduler,loss_fn)
        
    def train(self,epochs : int) -> History:
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self.validate()

            self.history.train_losses.append(train_loss)
            self.history.train_accs.append(train_acc)
            self.history.val_losses.append(val_loss)
            self.history.val_accs.append(val_acc)

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}"
            )

            # Save best + last (optional but common)


        return self.history

    def _train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)

            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()

            # ✅ Your behavior: scheduler.step() every batch
            if self.scheduler is not None:
                self.scheduler.step()

            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    def save_model(self) -> None:
        path = Path(__file__)
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = path.parent.parent.parent / "models" /f"model_{now}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)  # create folder if not exists
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


    def plot_history(self) -> None:
        # Loss plot
        plt.figure()
        plt.plot(self.history.train_losses, label="train_loss")
        plt.plot(self.history.val_losses, label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.savefig("Loss Curve.png")
        plt.show()

        # Accuracy plot
        plt.figure()
        plt.plot(self.history.train_accs, label="train_acc")
        plt.plot(self.history.val_accs, label="test_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")
        plt.savefig("Accuracy Curve.png")
        plt.show()
