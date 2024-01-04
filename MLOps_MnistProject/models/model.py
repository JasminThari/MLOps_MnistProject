import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # MNIST images are 28x28 after two pooling layers
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits (0-9)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
