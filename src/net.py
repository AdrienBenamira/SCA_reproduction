import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module): ##noConv1 ascad_desync 50

    def __init__(self, classes=256):
        super(Net,self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size=25) #note: no. of filter = no. out channel
        self.bn1 = nn.BatchNorm1d(num_features = 64)
        self.conv2 = nn.Conv1d(in_channels = 64,out_channels = 128,kernel_size = 3)
        self.bn2 = nn.BatchNorm1d(num_features=128)



        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features = 256, out_features = 20)
        self.fc2 = nn.Linear(in_features = 20, out_features = 20)
        self.fc3 = nn.Linear(in_features = 20, out_features = classes)

    def forward(self, x):
        # Averagepool first
        x = x.unsqueeze(1) #turning (Batch x length sequence) -> (Batch x channel = 1 x length sequence)
        x = F.avg_pool1d(x, kernel_size = 2, stride=2)
        # first convolutional layer
        x = F.avg_pool1d(self.bn1(F.selu(self.conv1(x))), kernel_size = 25, stride =25)
        # second convolutional
        x = F.avg_pool1d(self.bn2(F.selu(self.conv2(x))), kernel_size=4, stride=4)
        x = x.view(x.size()[0], -1) ## Flattening

        # Fully Connected layer
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.softmax(x)
        #print(F.softmax(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


