import torch
import SimpleITK as sitk  # 医疗影像界的最强搬运工
import numpy as np
import os
import time
from utils.config_utils import load_config, get_args
from utils.logger_utils import Logger
from model.unet import UNet
from scipy.ndimage import label
from data.data_preprocess_utils import windowing


def calculate_3d_dice(pred_mask, gt_mask):
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    if union == 0: # 如果两边都没画，说明全是背景，完美重合
        return 1.0
    return 2.0 * intersection / union


args = get_args()
config = load_config(args.config)
current_time = time.strftime("%Y%m%d_%H%M")
log_manager = Logger(logger_name="SpleenSeg-Unet", log_file=f"output/logs/SpleenSeg_UNet_evaluate_{current_time}.log")
logger = log_manager.get_logger()
logger.info("")
logger.info("=== Pre-flight Checklist ===")

# 1. 硬件配置
try:
    import torch_directml
    device = torch_directml.device()
except ImportError as e:
    torch_directml = None
    logger.error(f"[ERROR] Failed to load DirectML! Reason: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[INFO] Device set to: {device}")

# 2. 加载模型与权重
model = UNet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(config.paths.weight_path, map_location=device))
logger.info(f"[INFO] Model weight path is {config.paths.weight_path}")

# 3. 进入考试模式
model.eval()

# 4. 读取病人数据
patient_files = [f for f in os.listdir(config.paths.test_image_dir) if f.endswith('.nii.gz')]
logger.info(f"[INFO] Dataset loaded. Test: {len(patient_files)} files")
logger.info("============================")

total_dice = 0.0
start_time = time.time()
logger.info("")
logger.info("==== Evaluation Started ===✈")
with torch.no_grad():
    for idx, filename in enumerate(patient_files):
        img_path = os.path.join(config.paths.test_image_dir, filename)
        lbl_path = os.path.join(config.paths.test_label_dir, filename)

        # 读取原图和真实标签
        img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))  # 医生的标准答案

        z_slices, h, w = img_array.shape
        pred_3d_mask = np.zeros_like(img_array, dtype=np.uint8)

        # --- 逐层预测 ---
        for i in range(z_slices):
            slice_2d = img_array[i, :, :]

            # 归一化
            slice_2d = windowing(slice_2d, wl=40, ww=400)

            tensor_2d = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor_2d)
            prob_numpy = torch.sigmoid(output).squeeze().cpu().numpy()

            pred_3d_mask[i, :, :] = (prob_numpy > 0.5).astype(np.uint8)

        # --- 3D 后处理 ---
        labeled_array, num_features = label(pred_3d_mask)
        if num_features > 0:
            volumes = np.bincount(labeled_array.ravel())[1:]
            biggest_label = volumes.argmax() + 1
            pred_3d_mask = (labeled_array == biggest_label).astype(np.uint8)

        # --- 计算当前病人的 Dice ---
        patient_dice = calculate_3d_dice(pred_3d_mask, gt_mask)
        total_dice += patient_dice
        fragments_count = max(0, num_features - 1)  # 防bug代码
        logger.debug(f"[{idx + 1}/{len(patient_files)}] {filename} | Dice: {patient_dice * 100:.2f}% | find fragments: {num_features - 1} ")

logger.info("")
logger.info("== Evaluation Summary ==")
avg_dice = total_dice / len(patient_files)
total_time = time.time() - start_time
logger.info(f"Average Dice : {avg_dice * 100:.2f}% | Time : {int(total_time // 60)}m {int(total_time % 60):02d}s")
logger.info("========================")
