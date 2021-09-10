import torch
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        """
        An implementation of a Batch Normalization over a mini-batch of 2D inputs.

        The mean and standard-deviation are calculated per-dimension over the
        mini-batches and gamma and beta are learnable parameter vectors of
        size num_features.

        Parameters:
        - num_features: C from an expected input of size (N, C, H, W).
        - eps: a value added to the denominator for numerical stability. Default: 1e-5
        - momentum: momentum – the value used for the running_mean and running_var
        computation. Default: 0.1
        - gamma: the learnable weights of shape (num_features).
        - beta: the learnable bias of the module of shape (num_features).
        """

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(num_features)
        self.beta =  torch.zeros(num_features)

        # Register buffer here for purpose of using cuda
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))


    def forward(self, x):
        """
        During training this layer keeps running estimates of its computed mean and
        variance, which are then used for normalization during evaluation.
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data of shape (N, C, H, W) (same shape as input)
        """

        if self.training:
          mean_batch = x.mean([0,2,3])
          var_batch = x.var([0,2,3], unbiased=False)

          m = x.shape[0] * x.shape[2] * x.shape[3]

          with torch.no_grad():
            self.running_mean = (1-self.momentum) * self.running_mean + (self.momentum) * mean_batch
            unbiased_var = m / (m-1) * var_batch
            self.running_var = (1-self.momentum) * self.running_var + (self.momentum) * unbiased_var

          diff = x.sub(mean_batch[None,:,None,None])

          x = self.gamma[None,:,None,None] * ((diff)/ torch.sqrt(var_batch[None,:,None,None] + self.eps)) + self.beta[None,:,None,None]
        else:
          diff = x.sub(self.running_mean[None,:,None,None])
          x = self.gamma[None,:,None,None] * ((diff)/ torch.sqrt(self.running_var[None,:,None,None] + self.eps)) + self.beta[None,:,None,None]

        return x