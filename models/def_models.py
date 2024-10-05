import sys
import os
import torch
import torch.nn as nn
import copy
    
class DINO_v2_FT(nn.Module):
    def __init__(self, dino_backbone, output_channels):
        super(DINO_v2_FT, self).__init__()
        
        self.dino_backbone = copy.deepcopy(dino_backbone)

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
    
        x = self.dino_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
class Efficientnet_B4_FT(nn.Module):
    def __init__(self, efficientnet_backbone, output_channels):
        super(Efficientnet_B4_FT, self).__init__()

        self.efficientnet_backbone = copy.deepcopy(efficientnet_backbone)

        # EfficientNet-B4 typically outputs features of size 1792
        in_features = 1000

        self.fc = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        x = self.efficientnet_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    
    
    
    