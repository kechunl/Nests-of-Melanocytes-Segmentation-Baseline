from torch import nn
from torch.nn.init import xavier_uniform_ as xavier
from torch.nn.init import zeros_


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Input N x 3 x 128 x 128
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # N, 32, 128, 128
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # N, 64, 128, 128
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # N, 64, 64, 64

            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # N, 64, 64, 64
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # N, 128, 64, 64
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # N, 128, 32, 32

            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # N, 128, 32, 32
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # N, 64, 32, 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # N, 64, 16, 16

            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # N, 64, 16, 16
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # N, 32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # N, 32, 16, 16
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # N, 64, 16, 16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),    # N, 64, 32, 32

            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # N, 64, 32, 32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # N, 128, 32, 32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # N, 128, 64, 64

            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # N, 128, 64, 64
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # N, 64, 64, 64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # N, 64, 128, 128

            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # N, 64, 128, 128
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # N, 32, 128, 128
            nn.ReLU(True)
        )
        # Last layer will be changed in 2nd stage training. nn.Conv2d(32, 1, 3, stride=1, padding=1) + nn.Sigmoid()
        self.last_layer = nn.Sequential(
            nn.Conv2d(32, 3, 3, stride=1, padding=1),  # N, 3, 128, 128
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.last_layer(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        zeros_(m.bias.data)


def weights_init_seg(m):
    classname = m.__class__.__name__
    if ("decoder" in classname or "last_layer" in classname) and isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        zeros_(m.bias.data)