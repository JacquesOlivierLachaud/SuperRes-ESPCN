import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        # ReLU is so much better than Tanh !
        # and better than GELU
        self.mode  = "relu"
        self.act   = nn.Tanh() if self.mode == "tanh" else nn.ReLU()
        self.conv1 = nn.Conv2d(1,  128, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain( self.mode ))
        init.orthogonal_(self.conv2.weight, init.calculate_gain( self.mode ))
        init.orthogonal_(self.conv3.weight, init.calculate_gain( self.mode ))
        init.orthogonal_(self.conv4.weight)
