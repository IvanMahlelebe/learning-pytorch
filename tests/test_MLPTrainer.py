import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.MultiLayerPerceptron import MultiLayerPerceptron
from trainer.MLPTrainer import MLPTrainer

class TestMLPTrainer:
  @pytest.fixture
  def mock_data(self):
    """Create mock data for training and validation."""
    
    train_data = torch.randn(100, 3, 28, 28)
    train_labels = torch.randint(0, 2, (100,))
    val_data = torch.randn(20, 3, 28, 28)
    val_labels = torch.randint(0, 2, (20,))

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    return train_loader, val_loader

  @pytest.fixture
  def mock_model(self):
    """Create a MultiLayerPerceptron model."""
    layer_dims = [3 * 28 * 28, 128, 64, 1]
    activations = [torch.relu, torch.relu]
    return MultiLayerPerceptron(layer_dims, activations)
  
  @pytest.fixture
  def mock_epochs(self) -> int:
    return 10

  @pytest.fixture
  def mock_trainer(self, mock_model, mock_data, mock_epochs):
    """Create a mock MLPTrainer instance."""
    train_loader, val_loader = mock_data
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
    return MLPTrainer(
      mock_model, 
      train_loader, 
      val_loader, 
      loss_fn, 
      optimizer, 
      mock_epochs
    )


  def test_initialization(self, mock_trainer):
    """Test that the trainer initializes correctly."""
    assert mock_trainer.model is not None
    assert mock_trainer.train_loader is not None
    assert mock_trainer.val_loader is not None
    assert mock_trainer.loss_fn is not None
    assert mock_trainer.optimizer is not None
    assert len(mock_trainer.train_losses) == 0
    assert len(mock_trainer.val_losses) == 0
    assert len(mock_trainer.val_accuracies) == 0
    assert mock_trainer.best_vloss == float('inf')

  def test_train_one_epoch(self, mock_trainer):
    """Test that training one epoch updates the model and records loss."""
    initial_losses = len(mock_trainer.train_losses)
    avg_loss = mock_trainer.train_one_epoch()
    assert isinstance(avg_loss, float)
    assert len(mock_trainer.train_losses) == initial_losses + 1
    assert mock_trainer.train_losses[-1] == avg_loss

  def test_validate(self, mock_trainer):
    """Test that validation computes loss and accuracy correctly."""
    initial_losses = len(mock_trainer.val_losses)
    initial_accuracies = len(mock_trainer.val_accuracies)
    avg_vloss, accuracy = mock_trainer.validate()
    assert isinstance(avg_vloss, float)
    assert isinstance(accuracy, float)
    assert len(mock_trainer.val_losses) == initial_losses + 1
    assert len(mock_trainer.val_accuracies) == initial_accuracies + 1
    assert mock_trainer.val_losses[-1] == avg_vloss
    assert mock_trainer.val_accuracies[-1] == accuracy

  def test_train(self, mock_trainer):
    """Test that the full training loop runs and saves the best model."""
    epochs = 10
    mock_trainer.train()
    assert len(mock_trainer.train_losses) == epochs
    assert len(mock_trainer.val_losses) == epochs
    assert len(mock_trainer.val_accuracies) == epochs
    assert mock_trainer.best_vloss != float('inf')  # Best loss should be updated
    assert isinstance(mock_trainer.best_vloss, float)

  def test_save_best_model(self, mock_trainer):
    """Test that the best model is saved correctly."""
    mock_trainer.train()  # Train for 1 epoch
    
    import os
    assert os.path.exists("best_model.pth"), (
      "best_model file does not exist"
    )
    os.remove("best_model.pth") # Clean up