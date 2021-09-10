import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()

        """
        An implementation of a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Parameters:
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - kernel_size: Size of the convolving kernel
        - stride: The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - padding: The number of pixels that will be used to zero-pad the input.
        """

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        self.w = torch.zeros(out_channel, in_channel, *kernel_size)
        self.b = torch.zeros(out_channel)

        if not bias:
            self.b = 0

        self.F = out_channel
        self.C = in_channel
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """

        H_prime = int(((x.shape[2] - self.kernel_size[0] + 2 * self.padding) / self.stride) + 1)
        W_prime = int(((x.shape[3] - self.kernel_size[1] + 2 * self.padding) / self.stride) + 1)
        windows = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).transpose(1, 2)

        processed_windows = windows.matmul(self.w.view(self.w.size(0), -1).t()).add(self.b).transpose(1, 2)

        out = F.fold(processed_windows, (H_prime, W_prime), kernel_size=1)

        return out