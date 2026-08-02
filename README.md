# U-Net-Based Spleen Segmentation (MONAI Refactored Version)
[English](./README.md) | [简体中文](./README_zh-CN.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
![UNet](https://img.shields.io/badge/Model-U--Net-success?style=flat-square)
![MONAI|94](https://img.shields.io/badge/MONAI-v1.5.2-blue)
## Project Overview / Introduction
This project was originally a 2D spleen segmentation model implemented in pure **PyTorch** (the relevant code remains available on the [**`main`**](https://github.com/nbplus12345/SpleenSeg_UNet/tree/main) branch), designed primarily to automatically segment images from the **Task09_Spleen** dataset of the **Medical Segmentation Decathlon (MSD)**.
To meet production-grade medical imaging requirements, the project underwent a **complete architectural refactor** and fully adopted the **MONAI** medical deep learning framework, evolving from a collection of handcrafted 2D scripts into an end-to-end 3D medical imaging pipeline.
I created this project to learn how to construct and use the **U-Net** network and the **MONAI** framework.

I later wrote a fairly comprehensive retrospective on the project. It mainly documents my learning journey through the 2D U-Net spleen CT segmentation project, including NIfTI data processing, case-level data splitting, window width and window level normalization, slicing 3D volumes into 2D slices, blank-slice handling, the U-Net architecture, DiceLoss, reconstructing 2D predictions into 3D masks during inference, and a comparison between the MONAI refactored and handwritten workflows.

Blog: [From Cat-and-Dog Classification to Medical Image Segmentation: A Retrospective on 2D U-Net Spleen CT Segmentation](https://blog.csdn.net/weixin_53384391/article/details/161930030)

## Quick Preview / Quick Preview
![train_monai](train_monai.png)
![tensorboard_monai](tensorboard_monai.png)

**Core highlights of the MONAI refactor include:** 
* **High-Speed Persistent Caching (Persistent Caching)**: Replaces the original `.npy` intermediate files, which produced substantial I/O fragmentation, with `PersistentDataset` for hash-verified persistent caching of original 3D images, combining fast reads with dynamic data augmentation. 
* **Streamlined Sliding Window Inference**: Completely removes the cumbersome manual 2D slicing and 3D stitching loops from the original evaluation code. Sliding-window inference supports dynamic batch packing and Gaussian blending at the edges. 
* **Declarative Pipeline Post-processing**: Replaces the original `scipy`-based connected-component calculation and `SimpleITK` coordinate copying with a MONAI dictionary pipeline that performs probability activation, binarization, largest-connected-component retention, and automatic restoration and saving of NIfTI spatial metadata.
* **Interval-Based Validation**: Optimizes validation by switching from step-by-step validation to periodic interval-based validation, greatly reducing redundant computation on the validation set and improving overall training efficiency.
## Network Architecture / Network Architecture
This project uses MONAI's official **U-Net** implementation. It retains the classic symmetric encoder-decoder structure while being deeply optimized for medical imaging features. The basic architecture is shown below:
![unet](img.png)
As shown above, the network has the following core architectural features: 
1. **Residual Unit Integration (Residual Units)**: The refactored version sets `num_res_units` to 2. Residual connections are introduced at every feature extraction level, effectively mitigating vanishing gradients in deep networks and improving convergence speed on the validation set by approximately 30% compared with the original version. 
2. **High-Performance Operator Combination**: 
	* **Activation Function**: Replaces conventional ReLU with **PReLU** (Parametric ReLU), enabling the network to learn slopes in the negative region and better capture subtle tissue boundary features. 
	* **Normalization Layer**: Integrates **Instance Normalization**, which is more robust than Batch Normalization for medical image training with small batches (Batch Size=1/2). 
3. **Multi-Scale Feature Alignment**: A five-stage design with `channels=(64, 128, 256, 512, 1024)`, combined with the `strides=(2, 2, 2, 2)` downsampling strategy, balances large-scale anatomical structure (overall spleen location) with fine-grained local features (organ boundaries). 
4. **Lightweight Skip Connections**: Optimizes convolution after feature concatenation to ensure that shallow spatial information reaches the decoder without loss, enabling accurate pixel-level boundary reconstruction.
## Results and Performance / Results
Thanks to MONAI's dynamic data augmentation (random 3D patch sampling) and long training schedule, the model demonstrates strong generalization after approximately 200 epochs.
After 230 epochs, the model achieved a **95.59%** Dice score on the validation set and a **94.89%** mean 3D Dice score on the test set, matching or even exceeding the accuracy ceiling of the original offline 2D slicing approach. See the training and evaluation logs in **logs/** for details.

**The segmentation result is shown below:**
![spleen_seg_monai](spleen_seg_monai.png)
## Environment Setup / Installation

This project offers **high compatibility and cross-platform support**. It has been thoroughly trained and tested on the following operating systems and hardware acceleration environments:

| Operating System | Compute Device / GPU | Hardware Backend | Version |
| :----------------------------- | :------------------------- | :------- | :-------------------------------------------- |
| **Windows 11**                 | NVIDIA RTX 5060 8G         | CUDA     | PyTorch-2.8.0+cu128                           |
| **Linux (Ubuntu 24.04.4 LTS)** | AMD Radeon RX 7900 XTX 24G | ROCm     | PyTorch-2.11.0+rocm7.2                        |
| **Windows 11**                 | AMD Radeon 780M integrated GPU | DirectML | PyTorch-2.3.1+CPU<br>DirectML-0.2.2.dev240614 |

### Core Dependencies
By fully adopting the MONAI medical imaging pipeline, this refactored version greatly simplifies the underlying dependencies and **completely removes the original hard dependencies on `scipy`, `SimpleITK`, and `OpenCV`**. MONAI now handles all 3D I/O, preservation of physical metadata, and post-processing natively. Detailed environment requirements are provided in `requirements.txt`; the core requirements are:
* **Python** >= 3.9
* **PyTorch** >= 2.0.0
* **MONAI** = 1.5.2

We recommend using Conda to manage the environment. The commands are as follows:
### 1. Clone the Repository
```bash
git clone -b monai-version --single-branch https://github.com/nbplus12345/SpleenSeg_UNet.git
cd SpleenSeg_UNet
```
### 2. Create and Activate the Conda Environment
```bash
conda create -n SpleenSeg-UNet-monai python=3.9 -y
conda activate SpleenSeg-UNet-monai
```
### 3. Install the Core Deep Learning Framework (PyTorch)
Choose **one** of the following PyTorch installation methods based on your computer hardware:

* Option A: You have a dedicated NVIDIA GPU (recommended and fastest)
```Bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
* Option B: You only have a CPU or use a Mac; skip this step
* Option C: You use an AMD GPU or want to use the DirectML backend
```Bash
pip install torch torchvision torchaudio
pip install torch-directml
```
### 4. Install Project Dependencies (install all remaining dependencies at once)
```bash
pip install -r requirements.txt
```
## Data Preparation / Data Preparation
This project uses the public **Task09_Spleen** dataset from the **Medical Segmentation Decathlon (MSD)**, containing NIfTI data from **41** patients.
1. Visit [**Medical Segmentation Decathlon (MSD)**](http://medicaldecathlon.com/dataaws/) and download **Task09_Spleen**.
2. After extraction, move the **imagesTr** and **labelsTr** folders into the **dataset** folder. The remaining files may be deleted.
3. The initial data directory should have the following structure (ignore cache files beginning with ._):
```Plaintext
dataset/
├── imagesTr/
│   ├── spleen_2.nii.gz
│   ├── ...
└── labelsTr/
    ├── spleen_2.nii.gz
    ├── ...
```
4. Run the dataset splitting script, which automatically creates validation and test sets from the original training set:
```Bash
python data/split_dataset_utils.py
```
5. The split data directory has the following structure (you may delete imagesTr and labelsTr if desired):
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
## Training and Testing / Training & Evaluation
### 1. Training (Training)
Hyperparameters and data paths can be modified in config/config.yaml, and additional YAML files may also be added. Run training with:
```Bash
python train_monai.py --config ./config/config.yaml
```
This model supports **resuming interrupted training** and automatically saves a checkpoint after every epoch. To resume after an interruption, set **resume_training** to true in config.yaml.
### 2. Testing and Evaluation (Evaluation)
The evaluation script automatically calculates the mean Dice (DSC):
```Bash
python evaluate_monai.py --config ./config/config.yaml
```
### 3. View Segmentation Results (Segmentation)
Configure the input CT path and output path in config/config.yaml, then run the segmentation script:
```Bash
python inference_monai.py --config ./config/config.yaml
```
### 4. Real-Time Training Monitoring (TensorBoard)
This project deeply integrates TensorBoard to monitor training/validation Loss and the S-shaped rise of the Dice score in real time.
After training begins, open another terminal and run:
```Bash
tensorboard --logdir=./output/tensorboard --port=6006
```
Open `http://localhost:6006` in a browser to view it.
## License

This project is open-sourced under the MIT License and may be freely used, modified, and distributed. See the [LICENSE](./LICENSE) file for details.
