'Module for the class unet that can also be executed.'
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    'Neural network that follows the Unet architecture.'

    def __init__(
        self, input_channels=3, output_channels=1,
        features=[32, 64, 128, 256, 512], conv_kernel_size=3,
        final_conv_kernel_size=1, pool_kernel_size=2, stride=2,
    ):
        super(Unet, self).__init__()
        self.architecture_blocks(
            input_channels, output_channels, features, conv_kernel_size,
            final_conv_kernel_size, pool_kernel_size, stride,
        )

    def forward(self, input):
        'Forward of the network'
        down_block_1 = self.encoder_1(input)
        down_block_2 = self.encoder_2(self.pool(down_block_1))
        down_block_3 = self.encoder_3(self.pool(down_block_2))
        down_block_4 = self.encoder_4(self.pool(down_block_3))

        bottleneck_block = self.bottleneck_1(self.pool(down_block_4))

        up_block_1 = self.up_conv_1(bottleneck_block)
        up_block_1 = self.decoder_1(
            torch.cat((up_block_1, down_block_4), dim=1)
        )
        up_block_2 = self.up_conv_2(up_block_1)
        up_block_2 = self.decoder_2(
            torch.cat((up_block_2, down_block_3), dim=1)
        )
        up_block_3 = self.up_conv_3(up_block_2)
        up_block_3 = self.decoder_3(
            torch.cat((up_block_3, down_block_2), dim=1)
        )
        up_block_4 = self.up_conv_4(up_block_3)
        up_block_4 = self.decoder_4(
            torch.cat((up_block_4, down_block_1), dim=1)
        )

        final_block = self.final_conv(up_block_4)

        return final_block

    def architecture_blocks(
        self, input_channels: int, output_channels: int,
        features: List[int], conv_kernel_size: int,
        final_conv_kernel_size: int, pool_kernel_size: int, stride: int,
    ) -> None:

        self.encoder_1 = Unet.block(
            input_channels, features[0], conv_kernel_size,
        )
        self.encoder_2 = Unet.block(
            features[0], features[1], conv_kernel_size,
        )
        self.encoder_3 = Unet.block(
            features[1], features[2], conv_kernel_size,
        )
        self.encoder_4 = Unet.block(
            features[2], features[3], conv_kernel_size,
        )
        self.pool = nn.MaxPool2d(pool_kernel_size, stride)

        self.bottleneck_1 = Unet.block(
            features[3], features[4], conv_kernel_size,
        )
        self.up_conv_1 = nn.ConvTranspose2d(
            features[4], features[3], pool_kernel_size, stride,
        )
        self.up_conv_2 = nn.ConvTranspose2d(
            features[3], features[2], pool_kernel_size, stride,
        )
        self.up_conv_3 = nn.ConvTranspose2d(
            features[2], features[1], pool_kernel_size, stride,
        )
        self.up_conv_4 = nn.ConvTranspose2d(
            features[1], features[0], pool_kernel_size, stride,
        )
        self.decoder_1 = Unet.block(
            features[4], features[3], conv_kernel_size,
        )
        self.decoder_2 = Unet.block(
            features[3], features[2], conv_kernel_size,
        )
        self.decoder_3 = Unet.block(
            features[2], features[1], conv_kernel_size,
        )
        self.decoder_4 = Unet.block(
            features[1], features[0], conv_kernel_size,
        )
        self.final_conv = nn.Conv2d(
            features[0], output_channels, final_conv_kernel_size,
        )

    @staticmethod
    def block(
        input_channels: int, output_channels: int, conv_kernel_size: int,
    ) -> nn.Sequential:
        """Build a block for the Unet. It works for up blocks and down blocks,
        as well as the bottleneck.

        Arguments:
            input_channels {int} -- Input number of channels.
            output_channels {int} -- Output number of channels.
            conv_kernel_size {int} -- Kernel size for the convolution.

        Returns:
            nn.Sequential -- Sequence of the steps that the block follows.
        """
        conv_1, conv_2 = Unet.normal_cell_convolutions(
            input_channels,
            output_channels,
            conv_kernel_size,
        )
        block = nn.Sequential(
            conv_1, nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=output_channels),
            conv_2, nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=output_channels),
        )
        return block

    @staticmethod
    def up_conv(
        input_channels: int, output_channels: int,
        conv_kernel_size: int, stride: int,
    ) -> nn.ConvTranspose2d:
        """Create 2D up convolution to be used before up blocks.

        Arguments:
            input_channels {int} -- Input number of channels.
            output_channels {int} -- Output number of channels.
            conv_kernel_size {int} -- Kernel size for the convolution.
            stride {int} -- Stride for the convolution

        Returns:
            nn.ConvTranspose2d -- 2D Convolution.
        """
        up_conv = nn.ConvTranspose2d(
            input_channels, output_channels, conv_kernel_size, stride,
        )
        return up_conv

    @staticmethod
    def normal_cell_convolutions(
        input_channels: int, output_channels: int, conv_kernel_size: int = 3,
    ) -> nn.Conv2d:
        """Create normal cell convolutions 1 and 2, to be used inside each
        block, followed by the corresponding ReLUs and batch normalisation.

        Arguments:
            input_channels {int} -- Input number of channels.
            output_channels {int} -- Output number of channels.

        Keyword Arguments:
            conv_kernel_size {int} -- [description] (default: {3})

        Returns:
            nn.Conv2d -- 2D Convolution.
        """
        conv_1 = nn.Conv2d(
            input_channels, output_channels,
            conv_kernel_size, padding=1, bias=False
        )
        conv_2 = nn.Conv2d(
            output_channels, output_channels,
            conv_kernel_size, padding=1, bias=False
        )
        return conv_1, conv_2


if __name__ == "__main__":
    'Test execution of the net if the module is run.'
    net = Unet()
    print(net)
    input_values = torch.randn(1, 3, 256, 256)
    output = net(input_values)
    print(output)
