import torch
import torch.nn as nn

class l2norm(nn.Module):


    def __init__(self):
        super(l2norm, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([50.]))


    def forward(self, x):
        x = x/x.norm(p=2, dim=1, keepdim=True)
        out = self.alpha * x
        return out