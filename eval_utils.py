from torch import nn
import torch


class mIoUEstimator(nn.Module):
    def __init__(self, n_classes=91):
        super(mIoUEstimator, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, targets, smooth=1):
        """
        Args: 
            inputs: torch tensor [B, H, W] or [B, 1, H, W]
            targets: torch tensor [B, H, W] or [B, 1, H, W]
        """
        B = inputs.shape[0]
        inputs = inputs.view(B, -1)
        targets = targets.view(B, -1)
                
        sum_IoU = 0
        
        for i in range(self.n_classes):
            a = torch.zeros_like(inputs)
            b = torch.zeros_like(inputs)

            a[torch.where(inputs == i)] = 1.
            b[torch.where(targets == i)] = 1.
        
            intersection = (a * b).sum(dim=1)
            total = (a + b).sum(dim=1)
            union = total - intersection 
        
            IoU = (intersection + smooth)/(union + smooth)
            sum_IoU += IoU
                
        return sum_IoU/(self.n_classes)