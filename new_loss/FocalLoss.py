import torch

def FocalLoss(x, mask, gamma=2):
    # (N,C) = x.size()
    # mask = torch.zeros(N, C).scatter_(1, target_var.data.view(-1,1), 1)# convert labels to one hot matrix
    # mask = mask.cuda()
    x = torch.exp(x)
    x = x / x.sum(dim=1, keepdim=True)  # normalize x to be softmax
    x = mask * x
    x = x.sum(dim=1, keepdim=True)
    x = - torch.pow(1-x, gamma) * torch.log(x)
    out = x.mean()
    return out
