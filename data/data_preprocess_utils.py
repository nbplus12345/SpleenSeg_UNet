import os
import sys

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path


def windowing(img, wl, ww):
    # 窗宽窗位处理
    lower = wl - ww // 2
    upper = wl + ww // 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img.astype(np.float32)


def prepare_2d_slices(split_name):
    # 将3D卷切分为2D切片
    src_img_dir = f"../data/{split_name}/images"
    src_lab_dir = f"../data/{split_name}/labels"

    # 创建存放2D切片的目录
    save_dir = Path(f"../data/{split_name}_2d")
    (save_dir / "images").mkdir(parents=True, exist_ok=True)
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(src_img_dir) if f.endswith(".nii.gz")]
    print(f"\n[INFO] Preprocessing {split_name} set...")

    for f in tqdm(files, file=sys.stdout):
        # 1. 读取 3D 数据
        img_itk = sitk.ReadImage(os.path.join(src_img_dir, f))
        lab_itk = sitk.ReadImage(os.path.join(src_lab_dir, f))

        # 转换成 Numpy (z, y, x)
        img_array = sitk.GetArrayFromImage(img_itk)
        lab_array = sitk.GetArrayFromImage(lab_itk)

        # 2. 窗宽窗位处理
        img_array = windowing(img_array, wl=40, ww=400)

        # 3. 逐层切片并保存
        for z in range(img_array.shape[0]):
            slice_img = img_array[z, :, :]
            slice_lab = lab_array[z, :, :]

            # 优化策略：如果这一层完全没有脾脏(像素全为0)，我们可以跳过它，
            # 或者只保留一部分背景，这样可以防止模型被过多的负样本淹没。
            if np.sum(slice_lab) == 0:
                continue  # 这里我们先激进点：只练有脾脏的切片

            # 保存为 numpy 文件，读取速度极快
            case_name = f.replace(".nii.gz", "")
            np.save(save_dir / "images" / f"{case_name}_slice{z:03d}.npy", slice_img)
            np.save(save_dir / "labels" / f"{case_name}_slice{z:03d}.npy", slice_lab)
