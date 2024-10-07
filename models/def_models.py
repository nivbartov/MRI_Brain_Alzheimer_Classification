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
            nn.Dropout(0.1),
            
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, output_channels)
        )
    
    def forward(self, x):
        x = self.EfficientNet_backbone(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x