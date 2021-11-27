import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom_ResNet(nn.Module):
  def __init__(self,dropout=0.0):
    super().__init__()
    
    self.prep = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.BatchNorm2d(64),
                              nn.ReLU(inplace=True),
                              nn.Dropout2d(dropout)
                              )
    
    self.layer1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(2,2),  #16X16
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d(dropout) 
                                )
    
    self.res1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(inplace=True) 
                              )
    
    self.layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(2,2), #8X8
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d(dropout)
                                )
    
    self.layer3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(2,2), #4X4
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d(dropout)
                                )
    
    self.res2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(inplace=True) 
                              )
    
    self.pool = nn.MaxPool2d(4,4) #1X1
    
    self.fc =   nn.Linear(512,10, bias=False) 
    
  def forward(self, x):
    out = self.prep(x)
    
    out = self.layer1(out)
	r1  = self.res1(out)
	out = out + r1
    
    out = self.layer2(out)
    
    out = self.layer3(out)
    r2 = self.res2(out)
    out = out + r2
    
    out = self.pool(out)
	
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    
    return F.log_softmax(out,dim=1)
