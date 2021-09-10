import torch.nn as nn

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        """
        An implementation of a max-pooling layer.

        Parameters:
        - kernel_size: the size of the window to take a max over
        """

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, C, H', W').
        """

        windows = x.unfold(2, self.kernel_size[0], self.kernel_size[0]).unfold(3, self.kernel_size[1],
                                                                               self.kernel_size[1])
        out = windows.max(4)[0].max(4)[0]

        return out