import torch
import torch.nn as nn


class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            
            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)
            if x.is_cuda:
                alpha = alpha.cuda()
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x


class ContentRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            
            idx_swap = torch.randperm(N)
            x = x[idx_swap].detach()

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x
