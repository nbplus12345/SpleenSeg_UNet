import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SpleenDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: 2D切片存放的路径 (例如 './dataset/train_2d')
        """
        self.img_dir = os.path.join(data_dir, "images")
        self.lab_dir = os.path.join(data_dir, "labels")

        # 获取所有切片文件名并排序，确保图像和标签一一对应
        self.file_names = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # 1. 获取文件名
        file_name = self.file_names[idx]

        # 2. 读取 numpy 数据
        img = np.load(os.path.join(self.img_dir, file_name))
        lab = np.load(os.path.join(self.lab_dir, file_name))

        # 3. 维度增加：从 [512, 512] 变成 [1, 512, 512]
        # 因为 PyTorch 的卷积层要求输入必须有通道维度 (C, H, W)
        img = img[np.newaxis, :, :]
        lab = lab[np.newaxis, :, :]

        # 4. 强行转换为 Tensor
        img_tensor = torch.from_numpy(img).float()
        lab_tensor = torch.from_numpy(lab).float()

        return img_tensor, lab_tensor
