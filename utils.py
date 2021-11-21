import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as  np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def default_DL():
  transform = transforms.Compose(
    [transforms.ToTensor()])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
  return trainloader, trainset



class C_10_DS(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def set_compose_params(mean, std):
  horizontalflip_prob= 0.2
  rotate_limit= 15
  shiftscalerotate_prob= 0.25
  num_holes= 1
  cutout_prob= 0.5

  transform_train = A.Compose(
    [#A.HorizontalFlip(p=horizontalflip_prob),
     #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=rotate_limit, p=shiftscalerotate_prob),
     A.RandomCrop(width=32, height=32, p=4)
     A.CoarseDropout(max_holes=num_holes,min_holes = 1, max_height=16, max_width=16, 
     p=cutout_prob,fill_value=tuple([x * 255.0 for x in mean]),
     min_height=16, min_width=16, mask_fill_value = None),
     A.Normalize(mean = mean, std = std, max_pixel_value=255, always_apply = True),
     ToTensorV2()
    ])
  
  transform_valid = A.Compose(
    [
     A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255,
        ),
     ToTensorV2()
    ])
  return transform_train, transform_valid


def tl_ts_mod(transform_train,transform_valid):
  trainset = C_10_DS(root='./data', train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
  testset = C_10_DS(root='./data', train=False, download=True, transform=transform_valid)
  testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
  return trainset,trainloader,testset,testloader
