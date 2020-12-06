import torch
import torch.nn.functional as F
from torch import nn


class cnn_encoder(nn.Module):
    """A Convolutional Encoder with 2 levels of compression
    
    Args:
        in_channels (int): The number of channels found the input image/matrix
        mid_channels (int): Number of intermediate feature maps
        out_channels (int): Number of feature maps for the bottleneck
    """
    def __init__(self, in_channels=3, mid_channels=32, out_channels=64):
        super(cnn_encoder, self).__init__()
        
        # Convolutional Block #1
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.batch1 = nn.BatchNorm2d(mid_channels)

        # Max Pooling Layer #1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True) 

        # Convolutional Block #2 
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.batch2 = nn.BatchNorm2d(mid_channels)

        # Max Pooling Layer #2 
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True) 

        # Convolutional Layer #3
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, bias=True)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.batch3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        """The forward pass of the Convolutional Encoder
        
        Args:
            x (torch.Tensor): The input image/matrix
        
        Returns:
            torch.Tensor : Compressed matrix
            index1 : Max indices for first compression (for cnn_decoder)
            index2 : Max indices for second compression (for cnn_decoder)
        """
        # Convolutional Block #1
        out = self.conv1(x).clamp(min=0)
        out = self.dropout1(out)
        out = self.batch1(out)

        # Max Pooling Layer #1
        size1 = out.size()
        out, indices1 = self.pool1(out)

        # Convolutional Block #2
        out = self.conv2(out).clamp(min=0)
        out = self.dropout2(out)
        out = self.batch2(out)

        # Max Pooling Layer #2
        size2 = out.size()
        out, indices2 = self.pool2(out)
        
        # Convolutional Block #3
        out = self.conv3(out).clamp(min=0)
        out = self.dropout3(out)
        out = self.batch3(out)
          
        return out, indices1, indices2, size1, size2
    

class cnn_decoder(nn.Module):
    """A Convolutional Decoder with 2 levels of decompression
    
    Args:
        in_channels (int): The number of channels found in the input image/matrix
        mid_channels (int): Number of intermediate feature maps
        out_channels (int): Number of feature maps for the output image/matrix
    """
    def __init__(self, in_channels=64, mid_channels=32, out_channels=3):
        super(cnn_decoder, self).__init__()
        # DECODING
        # Deconvolutional Layer #1
        self.deconv1 = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=1, bias=True)
        nn.init.kaiming_normal_(self.deconv1.weight)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.batch1 = nn.BatchNorm2d(mid_channels)

        # Max Unpooling Layer #1 
        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2) 

        # Deconvolutional Layer #2
        self.deconv2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=3, stride=1, bias=True)
        nn.init.kaiming_normal_(self.deconv2.weight)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.batch2 = nn.BatchNorm2d(mid_channels)

        # Max Unpooling Layer #2 
        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2) 

        # Deconvolutional Layer #3
        self.deconv3 = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=1, bias=True)
        nn.init.kaiming_normal_(self.deconv3.weight)
    
    def forward(self, x, indices1, indices2, size1, size2):
        """The forward pass of the Convolutional Decoder
        
        Args:
            x (torch.Tensor): The input image/matrix
            index1: index1 from cnn_encoder.forward()
            index2: index2 from cnn_encoder.forward()
        
        Returns:
            torch.Tensor : Decompressed image/matrix
        """
        # Deconvolutional Block #1
        out = self.deconv1(x).clamp(min=0)
        out = self.dropout1(out)
        out = self.batch1(out)

        # Max Unpooling Layer #1
        out = self.unpool1(out, indices2, output_size=size2)

        # Deconvolutional Block 5
        out = self.deconv2(out).clamp(min=0)
        out = self.dropout2(out)
        out = self.batch2(out)

        # Max Unpooling Layer #2
        out = self.unpool2(out, indices1, output_size=size1)

        # Deconvolutional Block 6
        out = self.deconv3(out)
        
        return out

    

class cnn_autoencoder(nn.Module):
    """A Convolutional Autoencoder with symmetrical encoding and decoding halves
        Can be trained for a variety of tasks such as...
            - Denoising Images
            - Image Reconstruction
            - Image Segmentation
    
    Args:
        in_channels (int): The number of channels found the input image/matrix
        mid_channels (int): Number of feature maps for the bottleneck
        out_channels (int): Number of channels for the output image/matrix
    """
    def __init__(self, in_channels=3, bottleneck_channels=64, out_channels=3):
        super(cnn_autoencoder, self).__init__()

        self.encoder = cnn_encoder(
            in_channels=in_channels,
            mid_channels=(bottleneck_channels // 2),
            out_channels=bottleneck_channels
        )
        self.decoder = cnn_decoder(
            in_channels=bottleneck_channels,
            mid_channels=(bottleneck_channels // 2),
            out_channels=out_channels
        )


    def forward(self, x):
        """The forward pass of the Convolutional Autoencoder
        
        Args:
            x (torch.Tensor): The input image/matrix
        
        Returns:
            torch.Tensor : Output image/matrix
        """
        out, indices1, indices2, size1, size2 = self.encoder(x)
        out = self.decoder(out, indices1, indices2, size1, size2)
        return out