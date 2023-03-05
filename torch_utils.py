import os
import pandas as pd
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision.transforms as tt
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
        
# Moves the model/data to device    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
          
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class  ImageClassificationBase(nn.Module):

    def __init__(self):
        super().__init__()
    def training_step(self, batch):
        images, labels = batch
        out = self(images) 
                       # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss#tensor
        acc=accuracy(out,labels)

        return {'train_loss':loss,'train_acc':acc}
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)         
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        # batch_prec=[x['prec'] for x in outputs]
        # epoch_prec=torch.stack(batch_prec).mean()
        # batch_recall=[x['rec'] for x in outputs]
        # epoch_recall=torch.stack(batch_recall).mean()
        # batch_f1=[x['f'] for x in outputs]
        # epoch_f1=torch.stack(batch_f1).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f},train_acc:{:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_acc'],result['train_loss'], result['val_loss'],result['val_acc']))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace = True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class net(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)          #64 x 150 x 150
        self.conv2 = conv_block(64, 128, pool=True)       # 128 x 75 x 75
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))   # 128 x 75 x 75
        
        self.conv3 = conv_block(128, 256, pool=True)     #256 x 37 x 37
        self.conv4 = conv_block(256, 512, pool=True)     # 512 x 18 x 18
        self.res2 = nn.Sequential(conv_block(512, 512),
                                  conv_block(512, 512))  # 512 x 18 x 18
        
        self.classifier = nn.Sequential(nn.MaxPool2d(18), # 512 x 1 x 1
                                        nn.Flatten(),     #512
                                        nn.Dropout(0.0),  
                                        nn.Linear(512, num_classes))  # 512--> 6
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
# device=get_default_device()
model=to_device(net(3,3),torch.device('cpu'))
PATH='app/model.pth'
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

def predict_image(img, model=model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), torch.device('cpu'))
    # Get predictions from model
    yb = model(xb)
    print(yb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    return preds
def transform_image(image_bytes):
    import io
    transform = tt.Compose([tt.Resize((150,150)),
                                    tt.ToTensor(),])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image)
    # transform=tt.Compose([tt.ToTensor()])
    # img_t=transform(image)
    # tr=tt.Compose([tt.Resize((150,150))])