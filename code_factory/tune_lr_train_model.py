import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch_lr_finder import LRFinder
from deep_dream_code.models.custom_resnet import Custom_ResNet
from deep_dream_code.code_factory.OCLR_lr_finder import find_lr

def pipeline_tune_lr_train_model(list_end_lr, trainloader_mod, testloader_mod, total_epoch, max_epoch,  opt, oclr, device):
  for lr in list_end_lr:
    print(f'#################Running find_lr for lr {lr}')
    test_model = Custom_ResNet()
    start_lr = 1e-3
    end_lr = lr
    lrmax= find_lr('Adm',test_model,trainloader_mod, testloader_mod, start_lr, end_lr,total_epoch)
    print(f'######Value of lrmax is {lrmax}')
    model = Custom_ResNet()
    train_losses = []
    test_losses  = []
    wrong_prediction_list = []
    net, train_losses,test_losses,wrong_prediction_list,eval_test_acc = train_test_model(opt, oclr, model, trainloader_mod, testloader_mod,'BN', total_epoch, max_epoch, lrmax, device)
    if(eval_test_acc >=93):
      break
  return net, train_losses,test_losses,wrong_prediction_list
