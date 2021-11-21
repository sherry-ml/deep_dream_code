import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from tqdm import tqdm

import model_class
from model_class import Net
import train_model
from train_model import train
import test_model
from test_model import test

def train_test_model(trainloader, testloader, norm_type='BN', EPOCHS=20, dropout=0.1, lr=0.001, device='cpu'):

  train_losses_BN = []
  test_losses_BN = []
  train_acc_BN = []
  test_acc_BN = []
  wrong_predictions_BN = []

  torch.manual_seed(42)
  lambda_l1 = 0
  model =  Net(dropout, norm_type).to(device)
  print(model)
  optimizer = Adam(model.parameters(), lr=lr)
  #scheduler = OneCycleLR(optimizer, max_lr=0.05,epochs=EPOCHS,steps_per_epoch=len(trainloader))
  if(norm_type == 'BN'):
    train_losses = train_losses_BN
    train_acc    = train_acc_BN
    test_losses  = test_losses_BN
    test_acc     = test_acc_BN
    lambda_l1 = 0.002
    wrong_prediction_list = wrong_predictions_BN
  elif(norm_type == 'LN'):
    train_losses = train_losses_LN
    train_acc    = train_acc_LN
    test_losses  = test_losses_LN
    test_acc     = test_acc_LN
    wrong_prediction_list = wrong_predictions_LN
  else:
    train_losses = train_losses_GN
    train_acc    = train_acc_GN
    test_losses  = test_losses_GN
    test_acc     = test_acc_GN
    wrong_prediction_list = wrong_predictions_GN
  
  for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    #train(model, device, trainloader, optimizer, epoch, scheduler,train_losses, train_acc, lambda_l1)
    train(model, device, trainloader, optimizer, epoch, train_losses, train_acc, lambda_l1)
    eval_test_acc = test(model, device, testloader, test_losses, test_acc, epoch)
    if(eval_test_acc > 85):
        break
  
  model.eval()
  for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    match = pred.eq(labels.view_as(pred)).to('cpu').numpy()
    for j, i in enumerate(match):
      if(i == False):
        wrong_prediction_list.append((images[j], pred[j].item(), labels[j].item()))

  print(f'Total Number of incorrectly predicted images by model type {norm_type} is {len(wrong_prediction_list)}')
  return model
