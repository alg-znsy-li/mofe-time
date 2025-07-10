import torch.nn as nn
import torch

class CauchyMseLoss(nn.Module):
    def __init__(self, delta, reduce="none"):
        super().__init__()
        self.delta = delta
        self.reduce = reduce

    def forward(self, y_preds, y_labels, loss_mask=None):
        residual = y_labels - y_preds
        losses = torch.log2(
                1.0 + (residual/self.delta)**2
            )

        if loss_mask:
            losses *= loss_mask
            
        if self.reduce == 'sum':
            return torch.sum(
                input=losses, dtype=torch.float32
            )
        elif self.reduce == 'mean':
            if loss_mask is None:
                return torch.mean(
                    input=losses, dtype=torch.float32
                )
            else:
                return torch.sum(input=losses, dtype=torch.float32) / (torch.sum(loss_mask) + 0.001)

        return losses