import h5py
import torch
import torch.utils.data as data
import numpy as np

class CatVsNonCatDataset(data.Dataset):
  def __init__(self, file_path, train=True):
    dataset = h5py.File(file_path, "r")
    if train:
      self.images = np.array(dataset["train_set_x"][:]) / 255.0  # Normalize
      self.labels = np.array(dataset["train_set_y"][:])
    else:
      self.images = np.array(dataset["test_set_x"][:]) / 255.0
      self.labels = np.array(dataset["test_set_y"][:])

    # Reshape labels to match the expected shape (batch_size, 1)
    self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
    
    # Reshape images from (N, 64, 64, 3) to (N, 3, 64, 64) for PyTorch
    self.images = torch.tensor(self.images, dtype=torch.float32).permute(0, 3, 1, 2)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.images[idx], self.labels[idx]
