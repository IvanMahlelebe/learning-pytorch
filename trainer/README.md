# MLP Trainer with PyTorch

This repository demonstrates a clean and efficient way to train a **Multilayer Perceptron (MLP)** model using **PyTorch**. The code focuses on organizing the training pipeline, tracking performance metrics, and visualizing results using **Plotly** for easy inspection of model behavior.

## Key Features

- **Train an MLP model** on your dataset.
- **Visualize metrics** like training loss, validation loss, and validation accuracy across epochs.
- **Confusion Matrix plotting** for validation data to evaluate classification performance.
- **Early stopping** and **model saving** when validation loss improves.

## Structure of the Code

The code is organized around the `MLPTrainer` class, which encapsulates all the steps required for model training and evaluation.

### 1. **Initialization (`__init__`)**

The `MLPTrainer` class requires the following parameters for initialization:

- **`model`**: The MLP model to be trained. It should be a subclass of `torch.nn.Module`.
- **`train_loader`**: A PyTorch `DataLoader` providing the training data.
- **`val_loader`**: A PyTorch `DataLoader` providing the validation data.
- **`loss_fn`**: The loss function used for model training (e.g., `torch.nn.BCELoss` for binary classification).
- **`optimizer`**: Optimizer (e.g., `torch.optim.Adam`) to update model weights during training.
- **`epochs`**: The number of epochs to train the model.

### 2. **Training (`train_one_epoch`)**

The `train_one_epoch()` method handles the forward pass, loss computation, backpropagation, and weight update for each training epoch.

- **Forward Pass**: Inputs are passed through the model to compute outputs.
- **Loss Calculation**: The loss between the predicted outputs and true labels is computed.
- **Backpropagation**: Gradients are computed and applied to the model parameters.
- **Update Weights**: The optimizer updates the model weights to minimize the loss.

### 3. **Validation (`validate`)**

The `validate()` method is used to evaluate the model's performance on the validation set after each epoch.

- **Forward Pass**: Inputs from the validation set are passed through the model to compute predictions.
- **Loss Calculation**: The loss is computed for the validation set.
- **Accuracy Calculation**: The accuracy is computed by comparing predicted labels with the true labels.

The method returns the **average validation loss** and **accuracy**.

### 4. **Training Loop (`train`)**

The `train()` method orchestrates the training process for the specified number of epochs.

- **Epoch Loop**: For each epoch, it calls `train_one_epoch()` to train the model and `validate()` to evaluate it.
- **Model Saving**: If the validation loss improves, the model is saved as the best model.

### 5. **Metrics Visualization (`plot_metrics`)**

The `plot_metrics()` method visualizes the following metrics after training:

- **Training Loss**: The loss on the training set over the epochs.
- **Validation Loss**: The loss on the validation set over the epochs.
- **Validation Accuracy**: The accuracy on the validation set over the epochs.

The method uses **Plotly** to create interactive plots that help track the model’s progress and performance.

### 6. **Confusion Matrix Plotting (`plot_confusion_matrix`)**

The `plot_confusion_matrix()` method visualizes the performance of the model by plotting a confusion matrix. This is useful for understanding how well the model classifies each class.

- **Normalized Confusion Matrix**: The matrix is normalized to show the proportion of correct predictions for each class.

## How to Use

1. **Prepare Data**:
   - Load your data into `train_loader` and `val_loader` using PyTorch’s `DataLoader`.
   - Make sure the data is in the correct format (tensors, labels).

2. **Instantiate the Model**:
   - Define your model class (subclass of `nn.Module`), specifying the layers and the forward pass.

3. **Set Hyperparameters**:
   - Choose an appropriate **loss function** (e.g., `BCELoss` for binary classification).
   - Pick an **optimizer** (e.g., `Adam`).
   - Define the number of **epochs**.

4. **Train and Validate**:
   - Create an `MLPTrainer` object with the model, data loaders, loss function, optimizer, and number of epochs.
   - Call the `train()` method to start the training process.

5. **Visualize the Results**:
   - Call `plot_metrics()` to visualize training loss, validation loss, and accuracy over epochs.
   - Call `plot_confusion_matrix()` to inspect the confusion matrix for the validation data.

<!-- ## Example Code

```python
# Example of how to use MLPTrainer

from torch import nn, optim
from torch.utils.data import DataLoader
from your_dataset import YourDataset  # Assuming you have a dataset class

# Load your data
train_loader = DataLoader(YourDataset(train=True), batch_size=32, shuffle=True)
val_loader = DataLoader(YourDataset(train=False), batch_size=32, shuffle=False)

# Instantiate model, loss function, and optimizer
model = MultiLayerPerceptron()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Instantiate MLPTrainer and start training
trainer = MLPTrainer(model, train_loader, val_loader, loss_fn, optimizer, epochs=10)
trainer.train()

# Visualize metrics
trainer.plot_metrics()

# Plot confusion matrix
trainer.plot_confusion_matrix()
``` -->

---
**Author:** IvanMahlelebe