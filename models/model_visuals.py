import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


def plot_metric(csv_path, metric_column, train_label="Train", val_label="Test"):
  df = pd.read_csv(csv_path)
  
  steps = df['train_step'].values
  train_values = df[f'train_{metric_column}'].values
  val_values = df[f'val_{metric_column}'].values
  
  fig = go.Figure()

  for values, label, color in [
    (train_values, train_label, "#FC4100"),
    (val_values, val_label, "#FFC55A")
  ]:
    fig.add_trace(
      go.Scatter(
        x=steps,
        y=values,
        mode="lines",
        name=f"{label} {metric_column}",
        line=dict(color=color, dash="solid")
      )
    )

  fig.update_layout(
    title=f"<b>{metric_column.capitalize()} Over Training Steps</b>",
    xaxis_title="Training Steps",
    yaxis_title=metric_column.capitalize(),
    height=500,
    width=700,
    showlegend=True,
    legend=dict(
      x=0.99,
      y=0.99,
      xanchor="right",
      yanchor="top",
      bgcolor="rgba(255,255,255,0.7)",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified"
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