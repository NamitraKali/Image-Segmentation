import torch
import torch.nn.functional as F
from torch import nn


class convBlock(nn.Module):
    """Basic convolutional layer to be used throughout the model

    Args:
    in_channels (int): The number of input channels
    out_channels (int): The number of output channels
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, padding=1, rezero=False):
        super(convBlock, self).__init__()
        self.rezero = rezero

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True, padding=padding)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.batch = nn.BatchNorm2d(out_channels)
        
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        output = self.conv(x)
        output = self.act(output)
        if self.rezero:
            output = x + self.resweight*output
        output = self.dropout(output)
        output = self.batch(output)

        return output
    

class deconvBlock(nn.Module):
    """Basic convolutional layer to be used throughout the model

    Args:
    in_channels (int): The number of input channels
    out_channels (int): The number of output channels
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, padding=1):
        super(deconvBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.batch = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv(x)
        output = self.act(output)
        output = self.dropout(output)
        output = self.batch(output)

        return output


class Uencoder(nn.Module):
    """The first half of the UNet model with 2 layers of compression

    Args:
    in_channels (int): The number of input channels
    mid_channels (int): The number of channels in the intermediate layers
    """
    def __init__(self, in_channels, mid_channels, dropout_rate=0.2):
        super(Uencoder, self).__init__()
        
        self.block1_1 = convBlock(in_channels, mid_channels, dropout_rate=0.2)
        self.block1_2 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block1 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, return_indices=True)

        self.block2_1 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block2_2 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block2 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, return_indices=True)

        self.block3_1 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block3_2 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block3 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)

    def forward(self, x):
        output = self.block1_1(x)
        output = self.block1_2(output)
        block1 = self.block1(output)

        size1 = block1.size()
        pool1, indices1 = self.pool1(block1)
        
        output = self.block2_1(pool1)
        output = self.block2_2(output)
        block2 = self.block2(output)

        size2 = block2.size()
        pool2, indices2 = self.pool2(block2)

        output = self.block3_1(pool2)
        output = self.block3_2(output)
        block3 = self.block3(output)

        return block1, block2, block3, indices1, indices2, size1, size2


class Udecoder(nn.Module):
    """The second half of the UNet model with two layers of decompression

    Args:
    mid_channels (int): The number of intermediate channels
    out_channels (int): The number of output channels
    """
    def __init__(self, mid_channels, out_channels, dropout_rate=0.2):
        super(Udecoder, self).__init__()

        self.block1 = deconvBlock(mid_channels*2, mid_channels, dropout_rate=0.2)
        self.block1_1 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block1_2 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        
        # self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=1)

        self.block2 = deconvBlock(mid_channels*2, mid_channels, dropout_rate=0.2)
        self.block2_1 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block2_2 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        
        # self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=1)

        self.block3 = deconvBlock(mid_channels*2, mid_channels, dropout_rate=0.2)
        self.block3_1 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        self.block3_2 = convBlock(mid_channels, mid_channels, dropout_rate=0.2, rezero=True)
        
        self.output = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x, block1, block2, block3, indices1, indices2, size1, size2):
        output = torch.cat((x, block3), 1)
        output = self.block1(output)
        output = self.block1_1(output)
        output = self.block1_2(output)
        #print(output.size())
        #output = self.unpool1(output, indices2, output_size=size2)

        output = torch.cat((output, block2), 1)
        output = self.block2(output)
        output = self.block2_1(output)
        output = self.block2_2(output)
        #output = self.unpool2(output, indices1, output_size=size1)

        output = torch.cat((output, block1), 1)
        output = self.block3(output)
        output = self.block3_1(output)
        output = self.block3_2(output)
        
        output = self.output(output)

        return output


class UNet(nn.Module):
    """A UNet style implementation model for image segmentation

    Args:
    encoder (Uencoder): The encoder half of the model
    decoder (Udecoder): The decoder half of the model
    """
    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.2):
        super(UNet, self).__init__()
        
        self.encoder = Uencoder(in_channels, mid_channels, dropout_rate=dropout_rate)
        self.middle = convBlock(mid_channels, mid_channels, dropout_rate=dropout_rate, rezero=True)
        self.decoder = Udecoder(mid_channels, out_channels, dropout_rate=dropout_rate)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        block1, block2, block3, indices1, indices2, size1, size2 = self.encoder(x)
        output = self.middle(block3)
        output = self.decoder(output, block1, block2, block3, indices1, indices2, size1, size2)
        return self.act(output)