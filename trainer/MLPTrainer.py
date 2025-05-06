import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple

from sklearn.metrics import confusion_matrix
import numpy as np

from itertools import cycle

from datetime import datetime


class MLPTrainer:
  def __init__(
    self,
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int
  ) -> None:
    
    self.model: nn.Module = model
    self.train_loader: data.DataLoader = train_loader
    self.val_loader: data.DataLoader = val_loader
    self.loss_fn: nn.Module = loss_fn
    self.optimizer: optim.Optimizer = optimizer
    self.epochs: int = epochs

    # Store metrics for visualization
    self.train_losses: List[float] = []
    self.val_losses: List[float] = []
    self.val_accuracies: List[float] = []
    self.best_vloss: float = float("inf")
    self.accuracy: float = 0.0

  def train_one_epoch(self) -> float:
    self.model.train()
    running_loss: float = 0.0

    for i, data in enumerate(self.train_loader):
      inputs, labels = data
      self.optimizer.zero_grad()
      outputs: torch.Tensor = self.model(inputs)

      loss: torch.Tensor = self.loss_fn(outputs, labels)
      loss.backward()
      self.optimizer.step()

      running_loss += loss.item()

    avg_loss: float = running_loss / len(self.train_loader)
    self.train_losses.append(avg_loss)
    return avg_loss

  def validate(self) -> Tuple[float, float]:
    self.model.eval()
    running_vloss: float = 0.0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
      for inputs, labels in self.val_loader:
        outputs: torch.Tensor = self.model(inputs)
        labels = labels.float()

        loss: torch.Tensor = self.loss_fn(outputs, labels)
        running_vloss += loss.item()

        preds: torch.Tensor = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_vloss: float = running_vloss / len(self.val_loader)
    accuracy: float = correct / total

    self.val_losses.append(avg_vloss)
    self.val_accuracies.append(accuracy)

    return avg_vloss, accuracy

  def train(self) -> None:

    best_model_state = None
    for _ in range(self.epochs):
      _ = self.train_one_epoch()
      avg_vloss, accuracy = self.validate()

      if avg_vloss < self.best_vloss:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.best_vloss = avg_vloss
        self.accuracy = accuracy
        best_model_state = self.model.state_dict()

    if best_model_state is not None:
      torch.save(best_model_state, f"models/modelparams/iteration.params.{timestamp}.pth")
      # print(f"Saved new best model with validation loss {self.best_vloss:.4f}")

  def train_steps(self, total_steps: int) -> None:

    self.model.train()
    step = 0
    best_model_state = None
    train_iter = cycle(self.train_loader)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    self.train_errors = []
    self.val_errors = []

    while step < total_steps:
      inputs, labels = next(train_iter)
      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = self.loss_fn(outputs, labels)
      loss.backward()
      self.optimizer.step()

      self.model.eval()
      with torch.no_grad():
        preds = torch.sigmoid(outputs) > 0.5
        train_accuracy = (preds == labels).sum().item() / labels.size(0)
        train_error = 1.0 - train_accuracy

      self.model.train()
      self.train_losses.append(loss.item())
      self.train_errors.append(train_error)

      val_loss, val_accuracy = self.validate()
      val_error = 1.0 - val_accuracy
      self.val_errors.append(val_error)

      if val_loss < self.best_vloss:
        self.best_vloss = val_loss
        self.accuracy = val_accuracy
        best_model_state = self.model.state_dict()

      step += 1

    if best_model_state is not None:
      torch.save(best_model_state, f"models/modelparams/iteration.params.{timestamp}.pth")
