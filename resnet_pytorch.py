import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.new_block(16, 16, 1)
        self.layer2 = self.new_block(16, 32, 2)
        self.reslayer2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.layer3 = self.new_block(32, 64, 2)
        self.reslayer3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(4)
        )

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = self.bn1(output)
        output = self.layer1(output)
        output = self.layer2(output) + self.reslayer2(output)
        output = self.layer3(output) + self.reslayer3(output)
        output = self.avgpool(output)
        output = self.fc(output)
        return output

    @staticmethod
    def new_block(in_planes, out_planes, stride):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, 3, padding=1),
            nn.ReLU()
        )
