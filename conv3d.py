import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoConv3D(nn.Module):
    """
    CNN block for UNet with 3D convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(TwoConv3D, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq_block(x)

class UNet3D(nn.Module):
    def __init__(self, num_frames, in_channels=3, out_channels=1):
        super(UNet3D, self).__init__()

        self.encoder = nn.ModuleList([
            TwoConv3D(in_channels, 64),   # in_channels = 3 (RGB)
            TwoConv3D(64, 128),
            TwoConv3D(128, 256),
            TwoConv3D(256, 512),
            TwoConv3D(512, 1024),
        ])

        self.decoder = nn.ModuleList([
            nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=(1,2,2)),  # Upsample spatially
            TwoConv3D(1024, 512),
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=(1,2,2)),
            TwoConv3D(512, 256),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=(1,2,2)),
            TwoConv3D(256, 128),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=(1,2,2)),
            TwoConv3D(128, 64)
        ])

        # Final 3D conv to collapse the time dimension (num_frames -> 1)
        self.output = nn.Sequential(
            nn.Conv3d(64, out_channels, kernel_size=(num_frames, 1, 1)),  # Collapse frames
            nn.Squeeze(2)  # Removes the single frame dimension
        )

    def forward(self, x):
        enc_outputs = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x if i == 0 else F.max_pool3d(enc_outputs[-1], (1,2,2)))  # Pool only spatially
            enc_outputs.append(x)

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            x = self.decoder[i + 1](torch.cat([enc_outputs[-(i//2 + 2)], x], dim=1))

        x = self.output(x)  # Reduce the frame dimension (D) -> (batch, out_channels, H, W)
        return x

# Example usage
num_frames = 5  # Example: 5 frames
model = UNet3D(num_frames)
dummy_input = torch.randn(1, 3, num_frames, 256, 256)  # (batch, channels, frames, H, W)
output = model(dummy_input)
print(output.shape)  # Expected: (1, 1, 256, 256)
