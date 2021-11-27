
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


def train(oclr, model, device, train_loader, optimizer, epoch, scheduler, train_losses, train_acc, lambda_l1):

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
    if(oclr==True):
      scheduler.step()

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
  
def train_test_model(optim, oclr, model, trainloader, testloader, norm_type='BN', EPOCHS=20, max_epoch, lr=0.001, lrmax=0.5, device='cpu'):

  train_losses_BN = []
  test_losses_BN = []
  train_acc_BN = []
  test_acc_BN = []
  wrong_predictions_BN = []
  scheduler = None

  torch.manual_seed(42)
  lambda_l1 = 0
  model =  model.to(device)
  print(model)
  if(optim=='Adam'):
    optimizer = Adam(model.parameters(), lr=lr)
  else:
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.90)
  if(oclr==True):
    scheduler = OneCycleLR(optimizer=optimizer, max_lr=lrmax, epochs=EPOCHS, steps_per_epoch=len(trainloader), pct_start=max_epoch/EPOCHS, div_factor=10)
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
    
    train(oclr, model, device, trainloader, optimizer, epoch, scheduler,train_losses, train_acc, lambda_l1)
    #train(model, device, trainloader, optimizer, epoch, train_losses, train_acc, lambda_l1)
    eval_test_acc = test(model, device, testloader, test_losses, test_acc, epoch)
  
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
  return model, train_losses,test_losses,wrong_prediction_list
