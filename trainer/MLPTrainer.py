import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import plotly.graph_objects as go
from typing import List, Tuple

from sklearn.metrics import confusion_matrix
import numpy as np


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

  def train_one_epoch(self) -> float:
    self.model.train()
    running_loss: float = 0.0

    for i, data in enumerate(self.train_loader):
      inputs, labels = data
      self.optimizer.zero_grad()
      outputs: torch.Tensor = self.model(inputs)
      labels = labels.float().unsqueeze(1)

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
        labels = labels.float().unsqueeze(1)

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
    for epoch in range(self.epochs):
      _ = self.train_one_epoch()
      avg_vloss, _ = self.validate()

      if avg_vloss < self.best_vloss:
        self.best_vloss = avg_vloss
        torch.save(self.model.state_dict(), "best_model.pth")
        print(f"Saved new best model with validation loss {self.best_vloss:.4f}")

  def plot_metrics(self) -> None:
    """Plot training loss, validation loss, and validation accuracy."""
    epoch_steps = range(1, self.epochs + 1)
    
    fig = go.Figure()
    traces = [
      {
        "name": "Training Loss",
        "y": self.train_losses,
        "mode": "lines+markers",
        "line": dict(color="blue"),
        "row": 1,
        "col": 1,
      },
      {
        "name": "Validation Loss",
        "y": self.val_losses,
        "mode": "lines+markers",
        "line": dict(color="red"),
        "row": 2,
        "col": 1,
      },
      {
        "name": "Validation Accuracy",
        "y": self.val_accuracies,
        "mode": "lines+markers",
        "line": dict(color="green"),
        "row": 3,
        "col": 1,
      },
    ]

    for trace in traces:
      fig.add_trace(
        go.Scatter(
          x=list(epoch_steps),
          y=trace["y"],
          mode=trace["mode"],
          name=trace["name"],
          line=trace["line"],
        ),
        row=trace["row"],
        col=trace["col"],
      )

    fig.update_layout(
      title="Training and Validation Metrics",
      showlegend=True,
      hovermode="x unified",
      height=800,
    )

    fig.update_yaxes(title_text="Training Loss", row=1, col=1)
    fig.update_yaxes(title_text="Validation Loss", row=2, col=1)
    fig.update_yaxes(title_text="Validation Accuracy", range=[0, 1], row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    
    fig.show()

  def plot_confusion_matrix(self) -> None:
    """Plot a confusion matrix for the validation set."""
    self.model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
      for inputs, labels in self.val_loader:
        outputs: torch.Tensor = self.model(inputs)
        preds: torch.Tensor = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize

    # Plot using Plotly
    fig = go.Figure(
      data=go.Heatmap(
        z=cm,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        colorscale="Blues",
        colorbar=dict(title="Normalized Count"),
      )
    )

    fig.update_layout(
      title="Confusion Matrix",
      xaxis_title="Predicted Label",
      yaxis_title="Actual Label",
    )

    fig.show()
