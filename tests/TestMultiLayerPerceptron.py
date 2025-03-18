import pytest
import torch
from models.MultiLayerPerceptron import MultiLayerPerceptron

class TestMultiLayerPerceptron:
  @pytest.fixture
  def layer_dims(self):
    return [3, 4, 2]

  @pytest.fixture
  def activations(self):
    return [torch.relu]

  @pytest.fixture
  def batch_size(self):
    return 5

  def test_valid_initialization(self, layer_dims, activations):
    mlp = MultiLayerPerceptron(layer_dims, activations)
    assert len(mlp.layers) == len(layer_dims) - 1
    assert len(mlp.activations) == len(activations)

  def test_invalid_layer_dims(self):
    with pytest.raises(ValueError, match="layer_dims must have at least two elements"):
      _ = MultiLayerPerceptron([3], [torch.relu])

  def test_invalid_activations(self, layer_dims):
      with pytest.raises(ValueError, match="Number of activations must be equal to the number of hidden layers"):
        _ = MultiLayerPerceptron(layer_dims, [torch.relu, torch.sigmoid])

  def test_forward_pass_valid_input(self, layer_dims, activations, batch_size):
    mlp = MultiLayerPerceptron(layer_dims, activations)
    X = torch.randn(batch_size, 3)
    output = mlp(X)
    assert output.shape == torch.Size([batch_size, 2])

  def test_forward_pass_multi_dimensional_input(self, batch_size):
    mlp = MultiLayerPerceptron([784, 128, 10], [torch.relu])
    X = torch.randn(batch_size, 1, 28, 28)
    output = mlp(X)
    assert output.shape == torch.Size([batch_size, 10])

  def test_forward_pass_no_hidden_layers(self, batch_size):
    mlp = MultiLayerPerceptron([3, 2], [])
    X = torch.randn(batch_size, 3)
    output = mlp(X)
    assert output.shape == torch.Size([batch_size, 2])

  def test_forward_pass_large_layer_sizes(self):
    mlp = MultiLayerPerceptron([1000, 500, 100], [torch.relu])
    X = torch.randn(10, 1000)
    output = mlp(X)
    assert output.shape == torch.Size([10, 100])

  def test_forward_pass_empty_input(self, layer_dims, activations):
    mlp = MultiLayerPerceptron(layer_dims, activations)
    X = torch.randn(0, 3)
    output = mlp(X)
    assert output.shape == torch.Size([0, 2])