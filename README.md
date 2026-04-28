# 基于U-Net的脾脏分割（MONAI重构版）（Spleen Segmentation Based on U-Net）
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
![UNet](https://img.shields.io/badge/Model-U--Net-success?style=flat-square)
![MONAI|94](https://img.shields.io/badge/MONAI-v1.5.2-blue)
## 项目简介 / Introduction
本项目最初是一个基于纯 **PyTorch** 实现的 2D 脾脏器官分割模型（相关代码仍保留在 [**`main`**](https://github.com/nbplus12345/SpleenSeg_UNet/tree/main) 分支 中），主要针对 **Medical Segmentation Decathlon (MSD)** 中的 **Task09_Spleen（脾脏）** 数据集图像进行自动分割。
为满足医疗影像的工业级落地需求，本项目在底层架构上进行了**全盘重构**，全面接入了 **MONAI** 医疗深度学习框架，从“手工作坊式的 2D 脚本”跃升为“端到端 (End-to-End) 的 3D 医疗影像流水线”。
本人意在通过该项目掌握 **U-Net** 网络以及 **MONAI** 框架的构造与使用。
## 快速预览 / Quick Preview
![train_monai](train_monai.png)
![tensorboard_monai](tensorboard_monai.png)

**本次 MONAI 重构的核心亮点包括：** 
* **极速持久化缓存 (Persistent Caching)**：弃用原版产生大量 I/O 碎片的 `.npy` 中间文件，采用 `PersistentDataset` 实现带哈希校验的 3D 原图持久化缓存，完美兼顾极速读取与动态数据增强。 
* **降维打击 (Sliding Window Inference)**：彻底消灭了原版评估代码中繁琐的手工 2D 切片与 3D 拼接循环。引入滑动窗口推理，支持动态 Batch 打包与边缘高斯平滑融合 (Gaussian Blending)。 
* **声明式后处理 (Pipeline Post-processing)**：弃用原版基于 `scipy` 的连通域计算和 `SimpleITK` 的坐标拷贝，全面采用 MONAI 字典流水线，一键实现概率激活、二值化、最大连通域保留，并自动还原保存 NIfTI 空间物理元数据。
* **引入了间隔验证机制**：优化了验证逻辑，从逐步验证切换为基于间隔的周期性验证，大幅减少了验证集的冗余计算，提升了整体训练效率。
## 网络架构 / Network Architecture
本项目使用 **MONAI** 官方实现的 **U-Net** 网络，在保持经典“对称型”编码-解码结构的基础上，针对医疗影像特征进行了深度优化，基本构造如下：
![unet](img.png)
如上图所示，我们网络的核心架构特性如下： 
1. **残差单元集成 (Residual Units)**： 重构版将 `num_res_units` 设为 2。在每一层特征提取中引入了残差连接（Residual Learning），有效缓解了深层网络的梯度消失问题，使模型在验证集上的收敛速度较原版提升了约 30%。 
2. **高性能算子组合**： 
	* **激活函数**：弃用传统 ReLU，全面采用 **PReLU** (Parametric ReLU)，赋予网络学习负区间斜率的能力，进一步捕捉微弱的组织边缘特征。 
	* **归一化层**：集成 **Instance Normalization**，相比 Batch Normalization，在小批次（Batch Size=1/2）的医疗影像训练中具有更强的鲁棒性。 
3. **多尺度特征对齐**： 通过 `channels=(64, 128, 256, 512, 1024)` 的 5 阶跨度设计，配合 `strides=(2, 2, 2, 2)` 的下采样策略，使网络能在大尺度解剖结构（脾脏整体位置）与细粒度局部特征（器官边界）之间取得最优平衡。 
4. **轻量化跳跃连接 (Skip Connections)**： 优化了特征拼接（Concat）后的卷积逻辑，确保浅层空间信息能无损传递至解码器，从而实现像素级的精准边缘还原。
## 结果与性能 / Results
得益于 MONAI 的动态数据增强（随机 3D 切片采样）与长时间周期的训练策略，本模型在约 200 轮（Epochs）的训练后，展现出了极强的泛化能力。
该模型通过 230 轮的训练，在验证集上达到了 **95.59%** 的 Dice 分数。在测试集上达到了 **94.89%** 的 3D 平均 Dice 分数，完美对标甚至超越了原版离线 2D 切片的精度极限。详情见 **logs/** 中的训练与评估日志。

**分割后效果如图所示：**
![spleen_seg_monai](spleen_seg_monai.png)
## 环境配置 / Installation

本项目具有**高兼容性与跨平台适配**，已在以下多种操作系统与硬件加速环境中完成了严格的训练与测试：

| 操作系统                           | 计算设备 / GPU                 | 硬件后端     | 版本                                            |
| :----------------------------- | :------------------------- | :------- | :-------------------------------------------- |
| **Windows 11**                 | NVIDIA RTX 5060 8G         | CUDA     | PyTorch-2.8.0+cu128                           |
| **Linux (Ubuntu 24.04.4 LTS)** | AMD Radeon RX 7900 XTX 24G | ROCm     | PyTorch-2.11.0+rocm7.2                        |
| **Windows 11**                 | AMD Radeon 780M 核显         | DirectML | PyTorch-2.3.1+CPU<br>DirectML-0.2.2.dev240614 |

### 核心依赖项
得益于全面接入 MONAI 医疗影像流水线，本重构版大幅精简了底层依赖，**彻底移除了原版对 `scipy`、`SimpleITK` 以及 `OpenCV` 的硬性依赖**，所有 3D I/O 读取、物理元数据保留与后处理操作均由 MONAI 原生接管，详细的环境要求在 `requirements.txt` 中，核心库要求如下：
* **Python** >= 3.9
* **PyTorch** >= 2.0.0
* **MONAI** = 1.5.2

我们推荐使用 Conda 管理环境，具体命令如下：
### 1、克隆仓库
```bash
git clone -b monai-version --single-branch https://github.com/nbplus12345/SpleenSeg_UNet.git
cd SpleenSeg_UNet
```
### 2、创建激活conda环境
```bash
conda create -n SpleenSeg-UNet-monai python=3.9 -y
conda activate SpleenSeg-UNet-monai
```
### 3. 安装核心深度学习框架 (PyTorch)
请根据你电脑的硬件情况，选择以下【其中一种】方式安装 PyTorch：

* 选项 A：你有 NVIDIA 独立显卡（推荐，速度最快）
```Bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
* 选项 B：你只有 CPU，或者使用 Mac 电脑，则跳过该步骤
* 选项 C：你使用 AMD 显卡或想使用 DirectML 后端
```Bash
pip install torch torchvision torchaudio
pip install torch-directml
```
### 4、安装项目依赖 (一键安装剩余的依赖)
```bash
pip install -r requirements.txt
```
## 数据集准备 / Data Preparation
本项目使用公开的 **Medical Segmentation Decathlon (MSD)** 中的 **Task09_Spleen（脾脏）** 数据集，包含 **82** 例患者脾脏部位的 NIfTI 数据。
1. 请前往 [**Medical Segmentation Decathlon (MSD)**](http://medicaldecathlon.com/dataaws/) 下载数据 **Task09_Spleen**。
2. 解压后将文件夹内的 **imagesTr** 与 **labelsTr** 文件夹移至 **dataset** 文件夹内，其余可自行删除。
3. 初始数据目录结构应如下所示（忽略 ._ 开头的缓存文件）：
```Plaintext
dataset/
├── imagesTr/
│   ├── spleen_2.nii.gz
│   ├── ...
└── labelsTr/
    ├── spleen_2.nii.gz
    ├── ...
```
4. 运行数据集切分脚本，该脚本会自动从原始训练集中切分出验证集与测试集：
```Bash
python data/split_dataset_utils.py
```
5. 切分后的数据目录结构如下所示（可自行选择将 imagesTr 与 labelsTr 删除）：
```Plaintext
dataset/
├── imagesTr/
├── labelsTr/
├── test/
│   ├── images/
│   └── labels/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```
## 训练与测试 / Training & Evaluation
### 1. 训练 (Training)
对于各类超参数以及数据地址，可以在 config/config.yaml 中修改，也可以增加新的 yaml 文件。训练命令如下：
```Bash
python train_monai.py --config ./config/config.yaml
```
本模型带有 **断点续训** 的功能，每轮自动保存 checkpoint ，但训练中断需要重新训练时，需要在 config.yaml 中修改 **resume_training** 为 true 。
### 2. 测试与评估 (Evaluation)
评估脚本会自动计算平均 Dice (DSC) 指标：
```Bash
python evaluate_monai.py --config ./config/config.yaml
```
### 3. 查看分割结果（Segmentation）
在 config/config.yaml 中配置待分割的 CT 文件路径以及输出路径，运行分割脚本：
```Bash
python inference_monai.py --config ./config/config.yaml
```
### 4. 实时训练监控（TensorBoard）
本项目深度集成了 TensorBoard，用于实时监控训练/验证 Loss 以及 Dice 分数的 S 型爬升曲线。
在训练开始后，重新打开一个终端并运行：
```Bash
tensorboard --logdir=./output/tensorboard --port=6006
```
打开浏览器访问 `http://localhost:6006` 即可查看。
## 后续计划 (To-Do)
---
- [ ] 改进日志，记录显示整体训练时间
- [ ] 使用 **Weights & Biases (W&B)** 代替 **TensorBoard**
- [ ] 尝试引进 **AMP** 混合精度训练
- [ ] 进入 **3D UNet** 训练
