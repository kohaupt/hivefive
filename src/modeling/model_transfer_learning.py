
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.densenet import DenseNet121_Weights
from torchinfo import summary

class DenseNetTransferModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_classes = 4

        self.transfer_model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Disable gradient calculation for all parameters (no backpropagation needed)
        # (Freeze the feature parameters)
        for param in self.transfer_model.parameters():
            param.requires_grad = False
        
        # Change the output layer to match the number of classes in our dataset
        num_input_features = self.transfer_model.classifier.in_features
        self.transfer_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_classes),
        )

        print(summary(self.transfer_model))


    def forward(self, x):
        return self.transfer_model(x)