    # 基于U-Net的脾脏分割（Spleen Segmentation Based on U-Net）
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
## 项目简介 / Introduction/Abstract
本项目是一个基于 **U-Net** 的脾脏器官分割模型，主要针对 **Medical Segmentation Decathlon (MSD)** 中的 **Task09_Spleen（脾脏）** 数据集图像进行自动分割。本人意在通过该项目掌握 **U-Net** 网络的构造与使用。
## 网络架构 / Network Architecture
本项目使用经典的 **U-Net** 网络，基本构造如下。
![img.png](img.png)
如上图所示，我们的网络主要由以下几个核心组件构成： 
1. **[双层卷积基础块 (DoubleConv)]**：由连续的两次 `Conv2d -> BatchNorm2d -> ReLU` 堆叠而成。作为贯穿整个网络的核心特征提取单元，它负责提取图像的局部纹理与结构特征，并借助批量归一化 (Batch Normalization) 显著缓解内部协变量偏移，加速模型收敛并提升在医疗影像上的训练稳定性。
2. **[全卷积编码器 (Encoder)]**：采用 4 阶下采样结构。通过最大池化层 (Max Pooling) 逐层将图像空间尺寸减半，同时利用双层卷积将特征通道数从 64 逐级扩展至网络底部的 1024。该模块负责逐步扩大感受野，提取从浅层边缘轮廓到深层抽象语义的多尺度特征。 
3. **[跳跃连接机制 (Skip Connections)]**：在 U 型结构的同一层级水平搭建桥梁，将编码器中保留了丰富高分辨率空间细节的浅层特征图，直接传递并拼接 (Concatenation) 到解码器对应的特征图中。这一机制极大地弥补了下采样过程中不可逆的空间信息丢失，使模型能够精准还原脾脏的复杂边缘形态。
4. **[上采样解码器 (Decoder)]**：采用转置卷积 (ConvTranspose2d) 作为上采样算子，逐层将深层语义特征图的空间分辨率放大 2 倍并减半通道数。在融合了跳跃连接传来的细粒度特征后，再经过卷积层进行特征解码，最后通过 $1 \times 1$ 卷积层将通道数降维至类别数 (单通道)，输出脾脏的 2D 分割掩膜 (Mask) 概率图。
## 结果与性能 / Results
该模型通过 10 轮的训练，在验证集上达到了 **88.98%** 的 Dice 分数。在测试集上达到了 **93.07%** 的 Dice 分数。



## 环境配置 / Installation
本项目在以下环境中进行了严格的测试和验证： 
* Python == 3.9.25 
* PyTorch == 2.3.1+CPU
* torchaudio == 2.3.1+CPU
* torchvision == 0.18.1+cpu
* Torch-Directml == 0.2.2.dev240614
* numpy == 1.26.4
* OpenCV-python == 4.13.0.92
* MONAI == 1.5.2
* SimpleITK == 2.5.3
* Nibabel == 5.3.3
* scipy == 1.13.1
* pyyaml == 6.0.3
* tensorboard == 2.2

我们推荐使用 Conda 管理环境，具体命令如下：
### 1、克隆仓库
```bash
git clone https://github.com/nbplus12345/SpleenSeg_UNet.git
cd SpleenSeg_UNet
```
### 2、创建激活conda环境
```bash
conda create -n SpleenSeg-UNet python=3.9 -y
conda activate SpleenSeg-UNet
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
### 3、安装项目依赖 (一键安装剩余的依赖)
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
6. 由于本项目采用的是 2D U-Net，我们需要将 3D 的 `.nii.gz` 数据在 Z 轴上逐层切分为 2D 的 `.npy` 数组，并进行窗宽窗位（Windowing）归一化以及滤除无效的空白背景切片。请运行以下预处理脚本：
```Bash
python data/data_preprocess_utils.py
```
## 训练与测试 / Training & Evaluation
### 1. 训练 (Training)
对于各类超参数以及数据地址，可以在 config/config.yaml 中修改，也可以增加新的 yaml 文件。训练命令如下：
```Bash
python train.py --config ./config/config.yaml
```
本模型带有 **断点续训** 的功能，每轮自动保存 checkpoint ，但训练中断需要重新训练时，需要在 config.yaml 中修改 **resume_training** 为 true 。
### 2. 测试与评估 (Evaluation)
评估脚本会自动计算平均 Dice (DSC) 指标：
```Bash
python evaluate.py --config ./config/config.yaml
```
### 3. 查看分割结果（Segmentation）
在 config/config.yaml 中配置待分割的 CT 文件路径以及输出路径，运行分割脚本：
```Bash
python inference.py --config ./config/config.yaml
```