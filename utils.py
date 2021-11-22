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
    [A.RandomCrop(width=32, height=32, p=4),
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


'''
Below code gradcam code is been referenced from https://github.com/kazuto1011/grad-cam-pytorch/blob/fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py
'''

from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np



class GradCAM:
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers 
    target_layers = list of convolution layer index as shown in summary
    """
    def __init__(self, model, candidate_layers=None):
        def save_fmaps(key):
          def forward_hook(module, input, output):
              self.fmap_pool[key] = output.detach()

          return forward_hook

        def save_grads(key):
          def backward_hook(module, grad_in, grad_out):
              self.grad_pool[key] = grad_out[0].detach()

          return backward_hook

        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.nll).to(self.device)
        print(one_hot.shape)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:] # HxW
        self.nll = self.model(image)
        #self.probs = F.softmax(self.logits, dim=1)
        return self.nll.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.nll.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # need to capture image size duign forward pass
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # scale output between 0,1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

def generate_gradcam(misclassified_images, model, target_layers,device):
    images=[]
    labels=[]
    for i, (img, pred, correct) in enumerate(misclassified_images):
        images.append(img)
        labels.append(correct)
    
    model.eval()
    
    # map input to device
    images = torch.stack(images).to(device)
    
    # set up grad cam
    gcam = GradCAM(model, target_layers)
    
    # forward pass
    probs, ids = gcam.forward(images)
    
    # outputs agaist which to compute gradients
    ids_ = torch.LongTensor(labels).view(len(images),-1).to(device)
    
    # backward pass
    gcam.backward(ids=ids_)
    layers = []
    for i in range(len(target_layers)):
        target_layer = target_layers[i]
        print("Generating Grad-CAM @{}".format(target_layer))
        # Grad-CAM
        layers.append(gcam.generate(target_layer=target_layer))
        
    # remove hooks when done
    gcam.remove_hook()
    return layers, probs, ids

def plot_gradcam(gcam_layers, target_layers, class_names, image_size,predicted, misclassified_images, mean,std):
    
    images=[]
    labels=[]
    for i, (img, pred, correct) in enumerate(misclassified_images):
      images.append(img)
      labels.append(correct)

    c = len(images)+1
    r = len(target_layers)+2
    fig = plt.figure(figsize=(30,14))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    ax = plt.subplot(r, c, 1)
    ax.text(0.3,-0.5, "INPUT", fontsize=14)
    plt.axis('off')
    for i in range(len(target_layers)):
      target_layer = target_layers[i]
      ax = plt.subplot(r, c, c*(i+1)+1)
      ax.text(0.3,-0.5, target_layer, fontsize=14)
      plt.axis('off')

      for j in range(len(images)):
        image_cpu = images[j].cpu().numpy().astype(dtype=np.float32)
        for k in range(image_cpu.shape[0]):
          image_cpu[k] = image_cpu[k] * std[k] + mean[k]
        image_cpu = np.transpose(image_cpu, (1,2,0))
        img = np.uint8(255*image_cpu)
        if i==0:
          ax = plt.subplot(r, c, j+2)
          ax.text(0, 0.2, f"actual: {class_names[labels[j]]} \npredicted: {class_names[predicted[j][0]]}", fontsize=12)
          plt.axis('off')
          plt.subplot(r, c, c+j+2)
          plt.imshow(img)
          plt.axis('off')
          
        
        heatmap = 1-gcam_layers[i][j].cpu().numpy()[0] # reverse the color map
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (128,128))
        plt.subplot(r, c, (i+2)*c+j+2)
        plt.imshow(superimposed_img, interpolation='bilinear')
        
        plt.axis('off')
    plt.show()

def show_sample_img(loader, classes, n):
  import matplotlib.pyplot as plt
  import numpy as np

# functions to show an image


  def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
  dataiter = iter(loader)
  images, labels = dataiter.next()
  p = images.shape[0]
  if(n<p):
    p=n

# show images
  imshow(torchvision.utils.make_grid(images[:p]))
# print labels
  print(' '.join('%5s' % classes[labels[j]] for j in range(p)))
  
  
def torch_device(dev_stat):
  device = "cuda" if dev_stat else "cpu"
  print(device)
  return device

def view_model_summary(test_model,device):
  from torchsummary import summary
  test_model = test_model.to(device)
  summary(test_model, input_size=(3, 32, 32))
  print(test_model)
  
def display_incorrect_images(ll,std, mean, classes,n=10 ):
  import matplotlib.pyplot as plt
  display_images = ll[:n]
  index = 0
  fig = plt.figure(figsize=(10,14))
  for img in display_images:
    image = img[0].squeeze().to('cpu').numpy()
    for i in range(image.shape[0]):
      image[i] = image[i]*std[i] + mean[i]
    pred = classes[img[1]]
    actual = classes[img[2]]
    ax = fig.add_subplot(5, 2, index+1)
    ax.axis('off')
    ax.set_title(f'\n Predicted Label {pred} \n Actual Label : {actual}',fontsize=10) 
    ax.imshow(np.transpose(image,(1,2,0)), cmap='gray_r')
    index = index + 1
  plt.show()
  
def show_plots(train_losses, test_losses):
  import matplotlib.pyplot as plt
  fig, axs = plt.subplots(1,2,figsize=(15,5))

  axs[0].plot(train_losses, label='Train_losses')
  axs[0].legend(loc='upper right')
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Train_loss')
  axs[0].set_title("Training Loss")

  axs[1].plot(test_losses, label='Test_losses')
  axs[1].legend(loc='upper right')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Test Loss')
  axs[1].set_title("Test Loss")

  plt.show()
  
