"""Helpers for device placement and reproducibility."""

import random

import numpy as np
import torch


def resolve_device(device_name='auto'):
  """Returns the best available torch device."""
  if device_name and device_name != 'auto':
    return torch.device(device_name)
  if torch.cuda.is_available():
    return torch.device('cuda')
  if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    return torch.device('mps')
  return torch.device('cpu')


def get_module_device(module):
  """Returns the device of the first parameter in a module."""
  return next(module.parameters()).device


def seed_everything(seed):
  """Seeds Python, NumPy, and PyTorch."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
