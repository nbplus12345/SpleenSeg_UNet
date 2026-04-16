import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: 模型输出，经过 Sigmoid 后的概率 (0~1 之间)
        target: 真实的标签 (0 或 1)
        """
        # 1. 确保预测值在 0-1 之间 (如果你在模型最后没加 Sigmoid，这里要加)
        # pred = torch.sigmoid(pred)

        # 2. 将数据拉平成一维，就是将矩阵变成数列
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        # 3. 计算交集 (只有两者都为 1 时，乘积才为 1)
        # 就是数列的乘积，只是target数列里面只有0和1
        intersection = (pred_flat * target_flat).sum()

        # 4. 计算 Dice 系数
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        # 5. 返回 Loss (1 减去得分，得分越高 Loss 越小)
        return 1 - dice_score
