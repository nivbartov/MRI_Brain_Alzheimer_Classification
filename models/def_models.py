import sys
import os
import torch
import torch.nn as nn
import copy
    
class DINOv2(nn.Module):
    def __init__(self, DINOv2_backbone, output_channels):
        super(DINOv2, self).__init__()
        
        self.DINOv2_backbone = copy.deepcopy(DINOv2_backbone)

        in_features = 384 # Retrieve in_features from the original head
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1000, output_channels)
        )
    
    def forward(self, x):
        x = self.DINOv2_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    def __init__(self, ResNet_backbone, output_channels):
        super(ResNet, self).__init__()
        
        self.ResNet_backbone = copy.deepcopy(ResNet_backbone)
        
        in_features = 1000 # Retrieve in_features from the original head
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1000, output_channels)
        )
    
    def forward(self, x):
        x = self.ResNet_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self, EfficientNet_backbone, output_channels):
        super(EfficientNet, self).__init__()

        self.EfficientNet_backbone = copy.deepcopy(EfficientNet_backbone)

        # EfficientNet-B4 typically outputs features of size 1792
        in_features = 1000
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, output_channels)
        )
    
    def forward(self, x):
        x = self.EfficientNet_backbone(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
class NaiveModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=4):
        super(NaiveModel, self).__init__()

        self.conv_layer = nn.Sequential(
            # 1st main conv layer
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd main conv layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # 3rd main conv layer
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adjusting the input size for the fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(14 , 2048),  # Updated to 32768 input size
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, output_channels)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x