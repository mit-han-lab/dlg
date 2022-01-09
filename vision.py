import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from utils import label_to_onehot


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
        
class LeNet(nn.Module):
    BATCH_SIZE = 64
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            # nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # act(),
            # nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)
        )

        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


    def test_nn(self, test_loader,criterion):
        self.eval()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        correct = 0
        test_loss = 0
        accuracy_list = []

        for batch_idx, (data, label) in enumerate(test_loader):
            # send to device
            data, label = data.to(device), label.to(device)
            try:
                onehot_label = label_to_onehot(label, num_classes=10)
            except:
                continue

            output = self(data)
            test_loss = + criterion(output, onehot_label)

            pred = output.data.max(1, keepdim=True)[1]
            try:
                correct += pred.eq(label.view(LeNet.BATCH_SIZE, 1).data).cpu().sum().item()
            except:
                continue

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        accuracy_list.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                     len(test_loader.dataset),
                                                                                     accuracy))

    def train_nn(self, train_loader, optimizer, criterion,test_loader,  epoch_num=3):
        self.train()
        # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.0001)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        for epoch in range(epoch_num):
            correct = 0
            n = 0
            train_loss = 0
            accuracy_list = []

            for batch_idx, (data, label) in enumerate(train_loader):
                # send to device
                data, label = data.to(device), label.to(device)
                try:
                    onehot_label = label_to_onehot(label, num_classes=10)
                except:
                    continue

                optimizer.zero_grad()
                output = self(data)
                train_loss = + criterion(output, onehot_label)
                train_loss.backward()
                optimizer.step()

                pred = output.data.max(1, keepdim=True)[1]
                try:
                    correct += pred.eq(label.view(LeNet.BATCH_SIZE, 1).data).cpu().sum().item()
                except:
                    continue

                if batch_idx % 100 == 0:
                    print(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader),
                            train_loss.item()))

            train_loss /= len(train_loader.dataset)
            accuracy = 100. * correct / len(train_loader.dataset)
            accuracy_list.append(accuracy)
            self.test_nn(test_loader,criterion)

            print(
                '\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    train_loss, correct, len(train_loader.dataset), accuracy))


'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias") and m.bias != None:
        m.bias.data.uniform_(-0.5, 0.5)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.sigmoid(self.bn1(self.conv1(x)))
        out = F.sigmoid(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# class CNN(nn.Module):
#     def __init__(self, input_size, n_feature, output_size):
#         super(CNN, self).__init__()
#         self.n_feature = n_feature
#         self.input_size = input_size
#         self.fc_size = int((((self.input_size-4)/2)-4)/2)
#
#
#         self.output_size = output_size
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=5)
#         self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
#         self.fc1 = nn.Linear(n_feature * self.fc_size * self.fc_size, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x, verbose=False):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#         x = x.view(-1, self.n_feature * self.fc_size * self.fc_size)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.log_softmax(x, dim=1)
#         return x

class CNN(nn.Module):
    def __init__(self, n=4,kernel_size=5):
        super(CNN, self).__init__()
        # TODO: complete this method
        self.cnn1 = nn.Sequential( # 3x32x32
            nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=(kernel_size-1)//2),
            # nn.BatchNorm2d(n),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn2 = nn.Sequential(# 3x16x16
            nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=(kernel_size-1)//2),
            # nn.BatchNorm2d(2*n),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn3 = nn.Sequential(# 3x8x8
            nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=(kernel_size-1)//2),
            # nn.BatchNorm2d(4*n),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn4 = nn.Sequential(# 3x4x4
            nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=(kernel_size-1)//2),
            # nn.BatchNorm2d(8*n),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2)
        )# 3x28x14
        self.fc1 = nn.Linear(8*n*2*2,100)
        self.fc2 = nn.Linear(100,10)
    def forward(self, inp):
        out = self.cnn1(inp)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = nn.Flatten()(out)
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out