import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ImageEncoder(nn.Module):
    """
    Encodes an input image (4 x 96 x 96) into a 4-dimensional latent vector.
    """
    def __init__(self):
        super(ImageEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)   # -> (8, 48, 48)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1)  # -> (16, 24, 24)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1) # -> (32, 12, 12)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1) # -> (64, 6, 6)
        
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 4) 

    def forward(self, x):
        # Convolutional downsampling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten
        x = x.view(x.size(0), -1)  
        
        # Fully connected compression
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        
        return x


class ImageDecoder(nn.Module):

    
    def __init__(self, input_dim=4, output_dim=4, output_size=(96, 96)):
        super(ImageDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_size = output_size

        # Sequential FC layers to expand the input dimension
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.fc5 = nn.Linear(4096, 8192)
        self.fc6 = nn.Linear(8192, output_dim * output_size[0] * output_size[1])

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        # Reshape to (batch_size, output_dim, height, width)
        x = x.view(-1, self.output_dim, self.output_size[0], self.output_size[1])
        x = torch.sigmoid(x)
        return x
