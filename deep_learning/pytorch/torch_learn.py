import torch as torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
# Torch. FloatTensor is used to generate a floating-point Tensor,
# and the argument passed to the torch.
# FloatTensor can be a list or a dimension value.
th = torch
a = th.FloatTensor(2, 3)
b = th.FloatTensor([2, 3, 4, 5])
print(a, b)
# torch.randn
print(f"torch.randn can generate floating_point number\
 \nsatisfied a normal ditribution with a mean of 0 and a variance of 1")
print(f"such as torch.randn(2,3):{torch.randn(2,3)}")
# torch.zeros/ones/empty
print("these function is used to generate a tensor with all zero,one or None")
print("a = torch.zeros(2,3):{}".format(torch.zeros(2, 3)))
# torch.abs
print("torch.abs can return the absolute value with the same dim")
a = th.randn(3, 2)
print(f"{a} abs->\n{a.abs()}")
# torch.add
print(f"torch.add(a,b),can add two tensor one by one \
 n\and use broadcast mechanism when the dim is different")
a = torch.randn(3, 2)
b = torch.randn_like(a)
print(f'a:{a}\nb:{b}')
print(f"torch.add(a,b):{torch.add(a,b)}")
# torch.clamp
print(f"torch.clamp(tensor,down,up) can change the elements of a tensor to the range up and down and \
\nthe elements beyond up will equel to up. down is the same")
a = torch.randn(2, 3)
print(a)
print(torch.clamp(a, -1, 1))

# 自定义卷积层类
class CustomConvNet(nn.Module):
    def __init__(self):
        super(CustomConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        # 手动初始化卷积核
        with torch.no_grad():
            # 使用正态分布的随机数初始化卷积核
            ker = torch.ones([3,3,3,3],dtype=torch.float)
            self.conv1.weight = nn.Parameter(ker)
            self.conv1.bias = nn.Parameter(torch.tensor([0.0,0.0,0.0], dtype=torch.float32))

    def forward(self, x):
        x = self.conv1(x)
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
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # 转换为张量并增加一个 batch 维度

# 打印输入张量的形状
print("Input tensor shape:", input_tensor.shape)

# 创建模型实例
model = CustomConvNet()

# 打印模型结构和卷积核
print("Model structure:")
print(model)
print("\nCustom convolution kernel:")
print(model.conv1.weight)

# 执行前向传播
output_tensor = model(input_tensor)

# 移除 batch 维度，恢复到 [1, H, W] 形状
output_tensor = output_tensor.squeeze(0)

# 确保数据在 [0, 255] 范围内
output_tensor = (output_tensor - output_tensor.min()) / (output_tensor.max() - output_tensor.min()) * 255.0

# 将张量转换为 NumPy 数组
output_numpy = output_tensor.byte().cpu().numpy()  # 转换为 8-bit 图像

# 转换为 PIL 图像
output_img = Image.fromarray(output_numpy[0])

# 打印图像对象
print(output_img)

# 显示图像
output_img.show()
