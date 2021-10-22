# HyperIQA
⌚️: 2021年10月13日

📚参考
- [Blindly Assess Image Quality in the Wild Guided by A Self-Adaptive Hyper Network](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf)
---
## 1. 环境/Dependencies
- Python 3.6+
- PyTorch 0.4+
- TorchVision
- scipy
(optional for loading specific IQA Datasets)
- csv (KonIQ-10k Dataset)
- openpyxl (BID Dataset)

## 2. 代码使用/Usages

###  2.1 文件介绍
```
-- data ：数据集处理、获取
———— img-select-tools.py :评分小工具
———— get_dataset.py ：评分小工具评分后， 使用本代码获取代码可以使用的数据集

-- dataloader ：数据集加载
———— screenCheck.py ：dataset、dataloader类

-- models：模型类
———— models.py:    模型定义部分，定义了IQA网络
———— models2.py:   模型定义部分，定义了IQA网络，但与models.py的区别是此文件把targetnet融合到了hypernet当中，使得两个网络变成了一个整体

-- pretrained：模型存储
———— ScreenCheck_20211014184110_172.pth
———— ...

-- train.py ：训练的具体过程，train函数是训练的主函数（1、数据加载；2、模型定义；3、迁移学习；4、损失函数；5、优化器和部分冻结；6、训练和测试过程）
-- score_imgs.py ：评价文件夹中所有组文件
-- export_onnx.py ：将modes.py中模型导出, 并使用onnx模型运行
```
