import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import cv2
from PIL import Image


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


# 数据转换
transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
batch_size = 32
train_dataset = torchvision.datasets.ImageFolder(
    root=r'data\datasets\train', transform=transform)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(
    root=r'data\datasets\test', transform=transform)
test_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 模型训练函数


def model_train(num_classes: int, lr: float, num_epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    max_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            f'[eopch:{epoch}/{num_epochs}]:Accuracy:{(100 * correct / total):.2f} %')
        if max_acc < (correct / total):
            torch.save(model.state_dict(),
                       r'Learning\models\resnet18_best.pth')
            print("Model saved to Learning\models")
        max_acc = correct / total

    torch.save(model.state_dict(), r'Learning\models\resnet18.pth')
    print("Model saved to 'resnet18.pth'")
# 加载预训练权重


def load_pretrained_weights(model, state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# 定义超参数
learning_rate = 0.001
num_epochs = 20
num_classes = 3
# 训练模型
# model_train(num_classes, learning_rate, num_epochs)


with open(r"Learning\data\label.txt", 'r') as file:
    labells = []
    for line in file:
        # delete the blank of the str's right and left and the ',' of the str's right.
        # finally split the str by ':' once
        index, label = line.strip().rstrip(',').split(':', 1)
        # print(label.lstrip("'"))
        labells.append(label.replace("'", ''))


num_classes = 3  # 根据你的实际情况设置分类数
model = ResNet18(num_classes=num_classes)
# print(model)
# 加载预训练的 ResNet-18 权重
pretrained_weights = torch.load(r'Learning\models\resnet18.pth')  # 权重文件路径
load_pretrained_weights(model, pretrained_weights)
# 测试一个随机输入
# img = cv2.imread(r"downloaded_images\car\image_1.jpg")
img = Image.open(r"downloaded_images\car\image_1.jpg").convert("RGB")
# the image of opencv is bgr, will turn it into RGB
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)
output = model(img_tensor)
print(labells[torch.argmax(output[0]).tolist()])
