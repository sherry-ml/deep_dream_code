
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch_lr_finder import LRFinder

def find_lr(opt, model,train_loader, test_loader, start_lr, end_lr, epochs):
  lr_epochs = epochs
  num_iterations = len(test_loader) * lr_epochs
  criterion = nn.CrossEntropyLoss()
  if(opt == 'Adm'):
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
  else:
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.90)
  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=end_lr, num_iter=num_iterations, step_mode="linear",diverge_th=50)
        
  # Plot
  max_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
  lr_finder.plot(suggest_lr=True,skip_start=0, skip_end=0)
  # Reset graph
  lr_finder.reset()
  return max_lr
