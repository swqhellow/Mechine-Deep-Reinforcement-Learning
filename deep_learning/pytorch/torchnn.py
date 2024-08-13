import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class CustomConvNet(nn.Module):
    def __init__(self):
        super(CustomConvNet, self).__init__()
        # 定义一个卷积层，输出通道设置为 3 以匹配 RGB 图像
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        # 初始化卷积核为随机值
        ker = torch.stack([torch.ones([3, 3, 3], dtype=torch.float) / 9.0 for _ in range(3)])
        self.conv1.weight = nn.Parameter(ker)
        #nn.init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
        #nn.init.constant(self.conv1.weight,1/9.0)
        self.conv1.bias.data.zero_()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # 应用卷积层
        x = self.relu(x)
        return x

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor()
])
toimg = transforms.Compose([
    transforms.ToPILImage()
])

# 读取图片并进行预处理
image_path = r"Learning\data\bus.jpg"
image = Image.open(image_path).convert("RGB")  # 使用 PIL 打开图片并确保为 RGB 格式
input_tensor = transform(image).unsqueeze(0)  # 转换为张量并增加一个 batch 维度

# 打印输入张量的形状
print("Input tensor shape:", input_tensor.shape)

# 创建一个模型实例
model = CustomConvNet()

# 执行前向传播
output_tensor = model(input_tensor)

# 确保输出值在 [0, 1] 范围内
output_tensor = torch.clamp(output_tensor, 0, 1)
output_tensor = output_tensor.squeeze(0)  # 去掉 batch 维度
print(output_tensor.shape)
# 将输出张量转换为 PIL 图像并显示
output_img = toimg(output_tensor)
output_img.show()
