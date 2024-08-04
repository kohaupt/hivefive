
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_classes = 1
        # Size of mel spec: (128, 400)

        # Add all network layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        # After the first convolutional layer, the size of the feature map is 126x398
        # After the first pooling layer, the size of the feature map is 63x199
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        # After the second convolutional layer, the size of the feature map is 61x197
        # After the second pooling layer, the size of the feature map is 30x98
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        # After the third convolutional layer, the size of the feature map is 28x96
        # After the third pooling layer, the size of the feature map is 14x48
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        # After the fourth convolutional layer, the size of the feature map is 12x46
        # After the fourth pooling layer, the size of the feature map is 6x23
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
    
        self.fc1 = nn.Linear(128 * 6 * 23, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x