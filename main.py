
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from tqdm import tqdm


def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc, lambda_l1):

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    #loss = F.nll_loss(y_pred, target)
    loss = F.nll_loss(y_pred, target, reduction='sum')

    if(lambda_l1 > 0):
      l1 = 0
      for p in model.parameters():
        l1 = l1 + p.abs().sum()
      loss = loss + lambda_l1*l1


    train_loss += loss.item()
    

    # Backpropagation
    loss.backward()
    optimizer.step()
    #scheduler.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    #processed += len(data)

  #pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  
  train_losses.append(train_loss/len(train_loader.dataset))
  train_acc.append(100*correct/len(train_loader.dataset))

  print(f'\n Average Training Loss={train_loss/len(train_loader.dataset)}, Accuracy={100*correct/len(train_loader.dataset)}')

def test(model, device, test_loader,test_losses, test_acc,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy_epoch = 100. * correct / len(test_loader.dataset)
    if(accuracy_epoch > 85):
      model_name_file = "Session7_assignment_epoch_" + str(epoch) + "_acc_" + str(round(accuracy_epoch,2)) + ".pth"
      path = "/content/drive/MyDrive/EVA7/Session_7/" + model_name_file
      torch.save(model.state_dict(), path)
      print(f'Saved Model weights in file:  {model_name_file}')
    test_acc.append(100. * correct / len(test_loader.dataset))
    return accuracy_epoch
  
def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc, lambda_l1):

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    #loss = F.nll_loss(y_pred, target)
    loss = F.nll_loss(y_pred, target, reduction='sum')

    if(lambda_l1 > 0):
      l1 = 0
      for p in model.parameters():
        l1 = l1 + p.abs().sum()
      loss = loss + lambda_l1*l1


    train_loss += loss.item()
    

    # Backpropagation
    loss.backward()
    optimizer.step()
    #scheduler.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    #processed += len(data)

  #pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  
  train_losses.append(train_loss/len(train_loader.dataset))
  train_acc.append(100*correct/len(train_loader.dataset))

  print(f'\n Average Training Loss={train_loss/len(train_loader.dataset)}, Accuracy={100*correct/len(train_loader.dataset)}')
