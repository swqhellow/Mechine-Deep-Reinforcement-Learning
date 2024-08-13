import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import Image
# 数据转换
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[1e-4])
])
class self_dataloader(Dataset):
    def __init__(self,data_path,transform = None) -> None:
        self.data_path = data_path
        self.transform = transform
        self.images_path = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if fname.endswith('.jpg') or fname.endswith('.png')]
        self.labels = [self.get_label_from_filename(fname) for fname in os.listdir(data_path) if fname.endswith('.jpg') or fname.endswith('.png')]
    def get_label_from_filename(self,filename):
        return int(filename.split('_')[0])
    

data_path = r"downloaded_images"

batch_size = 64
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 查看 train_loader 的结构和数据类型
for i, (images, labels) in enumerate(train_loader):
    print(f"Batch {i+1}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    break  # 只查看第一个批次
