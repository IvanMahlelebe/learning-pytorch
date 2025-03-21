import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple

from sklearn.metrics import confusion_matrix
import numpy as np

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
    for _ in range(self.epochs):
      _ = self.train_one_epoch()
      avg_vloss, _ = self.validate()

      if avg_vloss < self.best_vloss:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.best_vloss = avg_vloss
        torch.save(self.model.state_dict(), f"models/modelparams/iteration.params.{timestamp}.pth")
        print(f"Saved new best model with validation loss {self.best_vloss:.4f}")

  def plot_metrics(self) -> None:
    """Plot training loss, validation loss, and validation accuracy on the same graph."""
    epoch_steps = list(range(1, self.epochs + 1))

    fig = go.Figure()

    traces = [
      {"name": "Training Loss", "y": self.train_losses, "color": "#FC4100"},
      {"name": "Validation Loss", "y": self.val_losses, "color": "#FFC55A"},
      {"name": "Validation Accuracy", "y": self.val_accuracies, "color": "#00215E"},
    ]

    for trace in traces:
      fig.add_trace(
        go.Scatter(
          x=epoch_steps,
          y=trace["y"],
          mode="lines",
          name=trace["name"],
          line=dict(color=trace["color"]),
        )
      )

    fig.update_layout(
      title="Training and Validation Metrics",
      xaxis_title="Epoch",
      yaxis_title="Metric Value",
      showlegend=True,
      hovermode="x unified",
      height=500,
      width=800,
      plot_bgcolor="white",  # White background for the graph area
      paper_bgcolor="white",
      xaxis=dict(range=[0, self.epochs + 1]),  # X-axis starts at 0 and ends at the last epoch
      yaxis=dict(range=[0, max(max(self.train_losses), max(self.val_losses), max(self.val_accuracies)) + 0.1]),  # White background for the entire plot
      legend=dict(
        x=0.02,  # Position the legend at 98% of the x-axis (right side)
        y=0.98,  # Position the legend at 98% of the y-axis (top side)
        xanchor="left",  # Anchor the legend to the right
        yanchor="top",  # Anchor the legend to the top
        bgcolor="rgba(255, 255, 255, 0)",  # Semi-transparent background
        # bordercolor="rgba(0, 0, 0, 0.5)",  # Border color
        #   borderwidth=1,  # Border width
      ),
    )

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
