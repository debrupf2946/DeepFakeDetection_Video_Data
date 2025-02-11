import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pywt
import cv2
import matplotlib.pyplot as plt


class Gated_convolution(nn.Module):
  def __init__(self, in_channels,kernel_size):
    super(Gated_convolution,self).__init__()
    self.out_channels=2*in_channels
    self.feature_projector = nn.Conv2d(in_channels,self.out_channels,kernel_size, padding=1)
    self.relu=nn.ReLU()
    self.sigmoid=nn.Sigmoid()

  def forward(self,x):
    x=self.feature_projector(x)
    x=self.relu(x)
    x1, x2 = torch.chunk(x, 2, dim=1)
    x1 = self.sigmoid(x1)
    x2 = self.relu(x2)
    output = x1 * x2
    return output


class Residual_feature(nn.Module):
    def __init__(self, in_channels):
        super(Residual_feature, self).__init__()
        # Create filters for each channel
        self.filter_1 = torch.tensor([
            [0,  0,  0,  0,  0],
            [0, -1,  2, -1,  0],
            [0,  2,  4,  2,  0],
            [0, -1,  2, -1,  0],
            [0,  0,  0,  0,  0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1).to(device)  # Match in_channels

        self.filter_2 = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1).to(device)

        self.filter_3 = torch.tensor([
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [0,  1, -2,  1,  0],
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1).to(device)

    def forward(self, x):
        edge_features = F.conv2d(x, self.filter_1, stride=1, padding=2, groups=x.size(1))
        texture_features = F.conv2d(edge_features, self.filter_2, stride=1, padding=2, groups=x.size(1))
        residual_features = F.conv2d(texture_features, self.filter_3, stride=1, padding=2, groups=x.size(1))
        return residual_features


class WaveletDomainFeature(nn.Module):
    def __init__(self, wavelet="db4"):
        super(WaveletDomainFeature, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        batch, channels, h, w = x.shape
        freq_images = []

        for i in range(batch):
            per_channel_features = []
            for c in range(channels):

                img_numpy = x[i, c].cpu().numpy()
                coeffs2 = pywt.dwt2(img_numpy, self.wavelet)
                LL, (LH, HL, HH) = coeffs2

                freq_image = np.abs(LH) + np.abs(HL) + np.abs(HH)
                freq_image = torch.tensor(freq_image, device=x.device, dtype=torch.float32)

                freq_image = freq_image.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

                freq_image = F.interpolate(freq_image, size=(h, w), mode='bilinear', align_corners=False)

                freq_image = freq_image.squeeze(0)
                per_channel_features.append(freq_image)

            per_channel_features = torch.cat(per_channel_features, dim=0)  # Shape: [3, 244, 224]
            freq_images.append(per_channel_features)


        freq_images = torch.stack(freq_images, dim=0)  # Shape: [batch, 3, 244, 224]
        return freq_images

class composite_feature_extractor(nn.Module):
    def __init__(self, in_channels):
        super(composite_feature_extractor, self).__init__()
        self.wavelet_extractor = WaveletDomainFeature()
        self.residual_network = Residual_feature(in_channels)

    def forward(self, x):
        wavelet_features = self.wavelet_extractor(x)
        residual_features = self.residual_network(x)
        print("wavelet_feature",wavelet_features.shape)
        print("residual_feature",residual_features.shape)
        composite_features = wavelet_features + residual_features
        return composite_features



class rgb_feature(nn.Module):
  def __init__(self):
    super(rgb_feature, self).__init__()
    self.feature_extractor=models.resnet34(pretrained=True)
    self.feature_extractor.eval()
    self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-5])

  def forward(self,x):
    with torch.no_grad():
      features=self.feature_extractor(x)
      return features


class gated_composite_feature_fusion(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(gated_composite_feature_fusion, self).__init__()
    self.composite_feature_extractor=composite_feature_extractor(in_channels=3)
    self.convolution = nn.Sequential(
            # First conv: 224 -> 112
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),

            # Second conv: 112 -> 56
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
            # Gated conv with same padding to maintain spatial dimensions
    self.gated_convolution=Gated_convolution(out_channels, kernel_size=3)  # Added padding=1

  def forward(self,x):
    composite_logit=self.composite_feature_extractor(x)
    logit=self.convolution(composite_logit)
    gated_logit=self.gated_convolution(logit)
    return gated_logit,logit


class adaptive_feature_map(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(adaptive_feature_map, self).__init__()
    self.rgb_feature_extractor=rgb_feature()
    self.gated_composite_feature_fusion=gated_composite_feature_fusion(in_channels,out_channels)
    self.gated_convolution=Gated_convolution(out_channels,kernel_size=3)

  def forward(self,x):
    composite_logit,_=self.gated_composite_feature_fusion(x)
    rgb_logit=self.rgb_feature_extractor(x)
    fusion_logit=torch.add(rgb_logit,composite_logit)
    gated_fusion_logit=self.gated_convolution(fusion_logit)
    return gated_fusion_logit


class adaptive_feature_fusion(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(adaptive_feature_fusion, self).__init__()
    self.adaptive_feature_map=adaptive_feature_map(in_channels, out_channels)
    self.gated_composite_feature_fusion=gated_composite_feature_fusion(in_channels,out_channels)
    self.channel_reducer = nn.Sequential(
    # First reduction with spatial preservation
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),

    # Second reduction maintaining spatial info
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),

    # Final pointwise reduction to target channels
    nn.Conv2d(32, 3, kernel_size=1),
    nn.BatchNorm2d(3)

)
    self.backbone=models.resnet34(pretrained=True)
    self.backbone.fc = nn.Linear(self.backbone.fc.in_features,2)


  def forward(self,x):
    adaptive_feature_map_logit=self.adaptive_feature_map(x)
    composite_logit,conv_composite_feature_map=self.gated_composite_feature_fusion(x)
    fusion_logit=torch.mul(adaptive_feature_map_logit,composite_logit)
    fusion_logit=torch.add(fusion_logit,conv_composite_feature_map)
    fusion_logit = self.channel_reducer(fusion_logit)
    fusion_logit=self.backbone(fusion_logit)

    return fusion_logit
