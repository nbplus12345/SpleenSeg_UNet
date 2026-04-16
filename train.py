import os
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.dataset_utils import SpleenDataset
from model.unet import UNet
from utils.loss import DiceLoss
from utils.logger_utils import Logger
from utils.config_utils import load_config, get_args
import time


args = get_args()
config = load_config(args.config)
current_time = time.strftime("%Y%m%d_%H%M")
log_manager = Logger(logger_name="SpleenSeg_UNet", log_file=f"output/logs/SpleenSeg_UNet_train_{current_time}.log")
logger = log_manager.get_logger()
TQDM_BASE_CONFIG = {
    "file": sys.stdout,
    "colour": "white",
    "disable": not sys.stdout.isatty(),
    "leave": False,           # 跑完自动消失，保持屏幕整洁
    "dynamic_ncols": True     # 自动适应终端窗口宽度
}
logger.info("")
logger.info("=== Pre-flight Checklist ==✈")

# 1. 基础配置
try:
    import torch_directml
    device = torch_directml.device()
except ImportError as e:
    torch_directml = None
    logger.error(f"[ERROR] Failed to load DirectML! Reason: {e}")
    # 如果没有安装 directml，则回退到标准的 CUDA (Nvidia GPU) 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[INFO] Device set to: {device}")
batch_size = config.train.batch_size    # 512x512的图比较大，显存有限的话建议先设小点（2或4）

# 2. 实例化数据集、开启装载机
train_dataset = SpleenDataset(config.paths.train_dir)
val_dataset = SpleenDataset(config.paths.val_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
logger.info(f"[INFO] Dataset loaded. Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
logger.info(f"[INFO] Dataloaders ready. Batch size: {train_loader.batch_size} | Total train steps per epoch: {len(train_loader)}")

# 3. 初始化
model = UNet(n_channels=1, n_classes=1).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"[INFO] Model initialized: 【{model.__class__.__name__}】 | Total trainable parameters: {total_params:,}")
criterion = DiceLoss()
logger.info(f"[INFO] Criterion: {criterion.__class__.__name__}")
optimizer = optim.SGD(model.parameters(), lr=config.train.lr, momentum=config.train.momentum, weight_decay=1e-4, foreach=False)
logger.info(f"[INFO] Optimizer configured: SGD (lr={config.train.lr}, momentum={config.train.momentum})")

# 可视化
# 实例化 TensorBoard 画笔
tb_log_dir = os.path.join('./output', f"tensorboard/SpleenSeg_UNet_Board_{current_time}")
board_writer = SummaryWriter(log_dir=tb_log_dir)
logger.info(f"[INFO] TensorBoard monitoring activated. Logs saved at: {tb_log_dir}")
# 全局计步器：用来记录模型一共吃了多少个 Batch (画图的横坐标 X 轴)
global_step = 0

# 早停参数
lowest_val_loss = float('inf')  # 初始设为无穷大
patience = config.train.patience
max_epochs = config.train.epochs
logger.info(f"[INFO] Early Stopping configured. Patience: {patience} epochs, max_epochs: {max_epochs}")
counter = 0  # 计数器

# 断点续训
start_epoch = 0
if config.train.resume_training and os.path.exists(config.paths.checkpoint_path):
    # 1. 把包裹取回来
    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    # 2. 依次把记忆注入到对应的身体里
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从断电的下一轮开始
    lowest_val_loss = checkpoint['lowest_val_loss']
    counter = checkpoint['counter']
    logger.info(f"[INFO] Find Checkpoint {config.paths.checkpoint_path}, Continue training from epoch {start_epoch + 1} .")

logger.info("============================")
logger.info("")
logger.info("===== Training Started ====✈")

# 4. 开始正式循环
for epoch in range(start_epoch, max_epochs):

    # --- 训练阶段 ---
    model.train()  # 告诉模型：现在是学习状态
    logger.info("")
    logger.info(f"[Epoch {epoch + 1:03d}/{max_epochs:03d}]")
    epoch_start_time = time.time()
    batch_total_loss = 0.0
    epoch_total_loss = 0.0

    train_pbar = tqdm(train_loader, desc=f"[Train]", **TQDM_BASE_CONFIG)
    for images, labels in train_pbar:
        global_step += 1
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)    # 二分类的激活函数
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 画瞬时 Loss 曲线 (横坐标是步数 global_step)
        board_writer.add_scalar("Train/Step_Loss", loss.item(), global_step)
        train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        batch_total_loss += loss.item()
        epoch_total_loss += loss.item()

    epoch_avg_train_loss = epoch_total_loss / len(train_loader)
    # 画 Epoch 平均 Loss 曲线 (横坐标是当前的 epoch)
    board_writer.add_scalar("Train/Epoch_Loss", epoch_avg_train_loss, epoch)
    tqdm.write("[INFO] Training completed")

    # --- 验证阶段 ---
    model.eval()
    val_total_dice = 0.0

    with torch.no_grad():  # 考试时不准改答案，所以关掉梯度计算，省内存
        val_pbar = tqdm(val_loader, desc=f"[Val]", **TQDM_BASE_CONFIG)
        for images, labels in val_pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.sigmoid(model(images))
            # 计算这一波的 Dice 得分（注意：Dice 得分 = 1 - Dice Loss）
            # 我们之前写的 loss.py 返回的是 1 - dice，所以这里反向算回来
            val_batch_loss = criterion(outputs, labels)
            val_total_dice += (1 - val_batch_loss.item())

    # 计算平均成绩
    val_avg_dice = val_total_dice / len(val_loader)
    val_avg_loss = 1 - val_avg_dice
    # 画验证集的loss (横坐标是当前的 epoch)
    board_writer.add_scalar("Val/Epoch_Loss", val_avg_loss, epoch)
    epoch_time = time.time() - epoch_start_time
    # 打印当前 Epoch 的综合成绩单
    logger.info("========================================================")
    logger.info(f"[Epoch Summary] | Train Loss: {epoch_avg_train_loss:.4f}  | Val Loss: {val_avg_loss:.4f}")
    logger.info(f"                | Val Dice  : {val_avg_dice:.4f}  | Time    : {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s")
    logger.info("========================================================")

    # 断点续训
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lowest_val_loss': lowest_val_loss,
        'counter': counter
    }
    torch.save(checkpoint, "latest_checkpoint.pth")  # 覆盖保存最新的快照

    if val_avg_loss < lowest_val_loss:
        # 情况 A：模型进步了！
        lowest_val_loss = val_avg_loss
        counter = 0  # 重置耐心计数器
        torch.save(model.state_dict(), config.paths.weight_path)
        logger.info(f"[SAVE] New best record! Model saved!")

    else:
        # 情况 B：模型没进步（甚至反弹了）
        counter += 1
        logger.info(f"[WARN] No improvement. Patience: {counter}/{patience}")
        if counter >= patience:
            board_writer.close()
            logger.info("[STOP] Early stopping triggered! Training halted, TensorBoard writer closed.")
            break  # 跳出大循环，提前结束训练
