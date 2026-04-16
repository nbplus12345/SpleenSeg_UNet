import torch
import SimpleITK as sitk  # 医疗影像界的最强搬运工
import numpy as np
import time
from utils.config_utils import load_config, get_args
from data.data_preprocess_utils import windowing
from model.unet import UNet
from scipy.ndimage import label

args = get_args()
config = load_config(args.config)
print("=== Pre-flight Checklist ===")
# 1. 硬件配置 (继续使用你打通的 DirectML)
try:
    import torch_directml
    device = torch_directml.device()
except ImportError as e:
    torch_directml = None
    print(f"[ERROR] Failed to load DirectML! Reason: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device set to: {device}")

# 2. 加载模型与权重
model = UNet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(config.paths.weight_path, map_location=device))
print(f"[INFO] Model weight path is {config.paths.weight_path}")

# 3. 进入考试模式
model.eval()

# 4. 读取病人数据
# 使用 SimpleITK 读取整张 3D 影像
original_image = sitk.ReadImage(config.paths.CT_dir)
# 把它转换成 NumPy 矩阵，方便我们用 Python 切片
# 形状会变成 (Z, Y, X)，Z 就是层数（比如 120 层，Y 和 X 是 512x512）
img_array = sitk.GetArrayFromImage(original_image)
z_slices, h, w = img_array.shape
print(f"[INFO] Load CT: {config.paths.CT_dir} | Size : [ {z_slices} , {h} , {w} ]")

# 准备一个全黑的、一模一样大小的 3D 盒子，用来装我们预测出来的脾脏
pred_3d_mask = np.zeros_like(img_array, dtype=np.uint8)

print("============================")
start_time = time.time()
print("==== Segmentation Started ===✈")
with torch.no_grad():  # 绝对不能记录梯度，不然显存瞬间爆炸
    for i in range(z_slices):
        # 抽出一张 2D 切片
        slice_2d = img_array[i, :, :]

        # 归一化处理
        slice_2d = windowing(slice_2d, wl=40, ww=400)

        tensor_2d = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).to(device)
        output = model(tensor_2d)
        prob_numpy = torch.sigmoid(output).squeeze().cpu().numpy()

        pred_3d_mask[i, :, :] = (prob_numpy > 0.5).astype(np.uint8)

        # 打印进度条
        if (i + 1) % 20 == 0:
            print(f"  -> Scanned {i + 1}/{z_slices} slices")

print(f"[INFO] Scan complete! Time : {time.time() - start_time:.2f} seconds")
print("[INFO] Starting 3D post-processing: removing isolated fragments...")
# 1. 寻找 3D 空间里所有连在一起的独立区块
labeled_array, num_features = label(pred_3d_mask)
if num_features > 0:
    # 2. 计算每个区块的体积（像素数量）
    # bincount 可以快速统计每个标签出现的次数，[1:] 是为了排除背景(标签0)
    volumes = np.bincount(labeled_array.ravel())[1:]
    # 3. 找到体积最大的那个区块的标签号
    biggest_label = volumes.argmax() + 1
    # 4. 魔法时刻：只保留最大的区块，其他全部变成 0
    pred_3d_mask = (labeled_array == biggest_label).astype(np.uint8)
    print(f"[INFO] Cleanup complete! Clean {num_features - 1} fragments.")
else:
    print("[WARN] Model predicted nothing, mask is completely blank!")

# 把 Numpy 矩阵变回医疗影像格式
predicted_sitk_img = sitk.GetImageFromArray(pred_3d_mask)

# 原样复制物理信息！
# 这一步极其关键。它把原图的“方向、像素间距、三维坐标原点”原封不动地复制给预测结果。
predicted_sitk_img.CopyInformation(original_image)

# 保存文件
sitk.WriteImage(predicted_sitk_img, config.paths.output_path)
print(f"[INFO] Success! 3D spleen model saved to: {config.paths.output_path}")