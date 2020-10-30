# PaddleSeg

## 简介

PaddleSeg是基于[PaddlePaddle](https://www.paddlepaddle.org.cn)开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。

- [特点](#特点)
- [安装](#安装)
- [使用教程](#使用教程)
  - [快速入门](#快速入门)
  - [基础功能](#基础功能)
  - [预测部署](#预测部署)
  - [高级功能](#高级功能)
- [在线体验](#在线体验)
- [FAQ](#FAQ)
- [交流与反馈](#交流与反馈)
- [更新日志](#更新日志)
- [贡献代码](#贡献代码)

## 特点

- **丰富的数据增强**

基于百度视觉技术部的实际业务经验，内置10+种数据增强策略，可结合实际业务场景进行定制组合，提升模型泛化能力和鲁棒性。

- **模块化设计**

支持U-Net, DeepLabv3+, ICNet, PSPNet, HRNet, Fast-SCNN六种主流分割网络，结合预训练模型和可调节的骨干网络，满足不同性能和精度的要求；选择不同的损失函数如Dice Loss, Lovasz Loss等方式可以强化小目标和不均衡样本场景下的分割精度。

- **高性能**

PaddleSeg支持多进程I/O、多卡并行等训练加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的显存开销，让开发者更低成本、更高效地完成图像分割训练。

- **工业级部署**

全面提供**服务端**和**移动端**的工业级部署能力，依托飞桨高性能推理引擎和高性能图像处理实现，开发者可以轻松完成高性能的分割模型部署和集成。通过[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)，可以在移动设备或者嵌入式设备上完成轻量级、高性能的人像分割模型部署。

- **产业实践案例**

PaddleSeg提供丰富地产业实践案例，如[人像分割](./contrib/HumanSeg)、[工业表计检测](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib#%E5%B7%A5%E4%B8%9A%E8%A1%A8%E7%9B%98%E5%88%86%E5%89%B2)、[遥感分割](./contrib/RemoteSensing)、[人体解析](contrib/ACE2P)，[工业质检](https://aistudio.baidu.com/aistudio/projectdetail/184392)等产业实践案例，助力开发者更便捷地落地图像分割技术。

## 安装

### 1. 安装PaddlePaddle

版本要求
* PaddlePaddle >= 1.7.0
* Python >= 3.5+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg.
```
pip install -U paddlepaddle-gpu
```
同时请保证您参考NVIDIA官网，已经正确配置和安装了显卡驱动，CUDA 9，cuDNN 7.3，NCCL2等依赖，其他更加详细的安装信息请参考：[PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/install/doc/index)。

### 2. 下载PaddleSeg代码

```
git clone https://github.com/PaddlePaddle/PaddleSeg
```

### 3. 安装PaddleSeg依赖
通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令：
```
cd PaddleSeg
pip install -r requirements.txt
```

## 使用教程

我们提供了一系列的使用教程，来说明如何使用PaddleSeg完成语义分割模型的训练、评估、部署。

这一系列的文档被分为**快速入门**、**基础功能**、**预测部署**、**高级功能**四个部分，四个教程由浅至深地介绍PaddleSeg的设计思路和使用方法。

### 快速入门

* [PaddleSeg快速入门](./docs/usage.md)

### 基础功能

* [自定义数据的标注与准备](./docs/data_prepare.md)
* [脚本使用和配置说明](./docs/config.md)
* [数据和配置校验](./docs/check.md)
* [分割模型介绍](./docs/models.md)
* [预训练模型下载](./docs/model_zoo.md)
* [DeepLabv3+模型使用教程](./tutorial/finetune_deeplabv3plus.md)
* [U-Net模型使用教程](./tutorial/finetune_unet.md)
* [ICNet模型使用教程](./tutorial/finetune_icnet.md)
* [PSPNet模型使用教程](./tutorial/finetune_pspnet.md)
* [HRNet模型使用教程](./tutorial/finetune_hrnet.md)
* [Fast-SCNN模型使用教程](./tutorial/finetune_fast_scnn.md)
* [OCRNet模型使用教程](./tutorial/finetune_ocrnet.md)

### 预测部署

* [模型导出](./docs/model_export.md)
* [Python预测](./deploy/python/)
* [C++预测](./deploy/cpp/)
* [Paddle-Lite移动端预测部署](./deploy/lite/)


### 高级功能

* [PaddleSeg的数据增强](./docs/data_aug.md)
* [PaddleSeg的loss选择](./docs/loss_select.md)
* [PaddleSeg产业实践](./contrib)
* [多进程训练和混合精度训练](./docs/multiple_gpus_train_and_mixed_precision_train.md)
* 使用PaddleSlim进行分割模型压缩([量化](./slim/quantization/README.md), [蒸馏](./slim/distillation/README.md), [剪枝](./slim/prune/README.md), [搜索](./slim/nas/README.md))

## FAQ

#### Q: 安装requirements.txt指定的依赖包时，部分包提示找不到？

A: 可能是pip源的问题，这种情况下建议切换为官方源，或者通过`pip install -r requirements.txt -i `指定其他源地址。

#### Q:图像分割的数据增强如何配置，Unpadding, StepScaling, RangeScaling的原理是什么？

A: 更详细数据增强文档可以参考[数据增强](./docs/data_aug.md)

#### Q: 训练时因为某些原因中断了，如何恢复训练？

A: 启动训练脚本时通过命令行覆盖TRAIN.RESUME_MODEL_DIR配置为模型checkpoint目录即可, 以下代码示例第100轮重新恢复训练：
```
python pdseg/train.py --cfg xxx.yaml TRAIN.RESUME_MODEL_DIR /PATH/TO/MODEL_CKPT/100
```

#### Q: 预测时图片过大，导致显存不足如何处理？

A: 降低Batch size，使用Group Norm策略；请注意训练过程中当`DEFAULT_NORM_TYPE`选择`bn`时，为了Batch Norm计算稳定性，batch size需要满足>=2
