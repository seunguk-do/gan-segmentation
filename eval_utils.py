from torch import nn
import torch


class mIoU(nn.Module):
    def __init__(self, n_classes=91):
        super(mIoULoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        ones = torch.ones_like(inputs)
        
        sum_IoU = 0
        
        for i in range(1, self.n_classes):
            a = torch.zeros_like(inputs)
            b = torch.zeros_like(inputs)
            #import pdb; pdb.set_trace()

            a[torch.where(inputs == i)] = 1.
            b[torch.where(targets == i)] = 1.
        
            intersection = (a * b).sum()
            total = (a + b).sum()
            union = total - intersection 
        
            IoU = (intersection + smooth)/(union + smooth)
            sum_IoU += IoU
                
        return sum_IoU/(self.n_classes-1)
