from torch import nn
import torch


class mIoUEstimator(nn.Module):
    def __init__(self, n_classes=91):
        super(mIoUEstimator, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        ones = torch.ones_like(inputs)
        
        sum_IoU = 0
        num_classes = 0
        
        for i in range(0, self.n_classes):
            a = torch.zeros_like(inputs)
            b = torch.zeros_like(inputs)

            a[torch.where(inputs == i)] = 1.
            b[torch.where(targets == i)] = 1.
        
            intersection = (a * b).sum()
            total = (a + b).sum()
            union = total - intersection 
        
            if union != 0:
                IoU = (intersection + smooth)/(union + smooth)
                sum_IoU += IoU
                num_classes += 1
                
                
        return sum_IoU/(num_classes)
