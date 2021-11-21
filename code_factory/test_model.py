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
