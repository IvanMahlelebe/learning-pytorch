import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from typing import List

def plot_metrics(train_losses: List[float], val_losses: List[float], val_accuracies: List[float]) -> None:
  """Plot training loss, validation loss, and validation accuracy on the same graph."""
  epoch_steps = list(range(1, len(train_losses) + 1))

  fig = go.Figure()

  traces = [
    {"name": "Training Loss", "y": train_losses, "color": "#FC4100"},
    {"name": "Validation Loss", "y": val_losses, "color": "#FFC55A"},
    {"name": "Validation Accuracy", "y": val_accuracies, "color": "#00215E"},
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
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(range=[0, len(train_losses) + 1]),
    yaxis=dict(range=[0, max(max(train_losses), max(val_losses), max(val_accuracies)) + 0.1]),
    legend=dict(
      x=0.02,
      y=0.98,
      xanchor="left",
      yanchor="top",
      bgcolor="rgba(255, 255, 255, 0)"
    ),
  )

  fig.show()

def plot_confusion_matrix(model: nn.Module, val_loader: DataLoader) -> None:
  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for inputs, labels in val_loader:
      outputs: torch.Tensor = model(inputs)
      preds: torch.Tensor = torch.sigmoid(outputs) > 0.5
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  cm = confusion_matrix(all_labels, all_preds)
  cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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