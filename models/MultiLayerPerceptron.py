import torch
import torch.nn as nn
from typing import List, Callable

class MultiLayerPerceptron(nn.Module):
  def __init__(self, layer_dims: List[int], activations: List[Callable[[torch.Tensor], torch.Tensor]]):
    """
      Initializes the MultiLayerPerceptron.

      Args:
        layer_dims: A list of integers representing the number of units in each layer.
        activations: A list of activation functions for each hidden layer.
    """
    super(MultiLayerPerceptron, self).__init__()
    self.layers = nn.ModuleList()
    self.activations = activations

    if len(layer_dims) < 2:
      raise ValueError("layer_dims must have at least two elements")
    if len(activations) != len(layer_dims) - 2:
      raise ValueError("Number of activations must be equal to the number of hidden layers")

    for i in range(len(layer_dims) - 1):
      self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    """
      Define how layers are connected i.e. how data flows

      Args:
        x: Input tensor.

      Returns:
        Output tensor.
    """
    # Flatten the input (except the batch dimension)
    Al = torch.flatten(X, start_dim=1)
    for l in range(len(self.layers)):
      Zl = self.layers[l](Al)
      if l < len(self.layers) - 1:
        Al = self.activations[l](Zl)
      else:
        Al = Zl
    AL = Al # for clarity
    return AL
