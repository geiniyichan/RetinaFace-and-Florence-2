# geiniyichan-RetinaFace-and-Florence-2
该项目使用开源的人脸检测模型RetinaFace与语义描述模型Florence-2相结合，对大规模人群进行人脸检测并对其进行详细描述

项目思路：通过训练大规模人脸检测模型RetinaFace在复杂环境下的多人脸并行检测能力，并在检测出人脸与切割的基础上，继续使用微软Azure AI 团队开源的Florence-2模型中的图像描述任务，对人脸的细节与表情情绪进行更深入的描述，从而得到大规模人群画像的表达。

本项目对RetinaFace还进行了多网络架构的对比，包括了resnet50、mobilenet0.25、shufflenetv2_Final，并使用yolov8作为基线模型进行深入对比，验证其在复杂环境下大规模人脸检测的优越性


# 安装
本项目所需环境

PyTorch  2.5.1

Python  3.12(ubuntu22.04)

CUDA  12.4

（没提到的自行安装🤭）

并可以使用```git clone https://github.com/geiniyichan/RetinaFace-and-Florence-2.git```进行克隆项目

# 数据集

需下载widerface数据集进行模型训练与测试

数据集目录格式：
text
 ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```

百度网盘链接https://pan.baidu.com/s/1kPN-A-jyjTwXn62OPSW_fA 提取码: GNYC

# 模型参数的下载

本项目使用RetinaFace大规模人脸检测模型进行训练实验与测试对比，其架构如下：


<img width="2048" height="489" alt="image" src="https://github.com/user-attachments/assets/a1c37760-d8bb-4fc4-a743-5d2e5b674ecd" />



本项目所有已训练好的模型参数均提供下载网盘，其中，关于RetinaFace的模型参数，直接存放入weights文件夹即可使用

放置格式为：
```text
./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
      shufflenetv2_Final.pth
      ghostnet_1x.pth
```

百度网盘链接: https://pan.baidu.com/s/1DJbCn0THTYMZYU0wx1O-iQ 提取码: GNYC


Florence-2 由 Microsoft 于 2024 年 6 月发布，是在 MIT 许可下开源的高级、轻量级基础视觉语言模型。该模型非常有吸引力，因为它体积小（0.2B 和 0.7B）并且在各种计算机视觉和视觉语言任务上具有强大的性能。 尽管体积小，但它的效果可与 Kosmos-2 等更大的型号相媲美。该模型的优势不在于复杂的架构，而在于大规模的 FLD-5B 数据集，该数据集由 1.26 亿张图像和 54 亿条综合视觉注释组成。

该模型支持多种任务：
- Caption,
- Detailed Caption,
- More Detailed Caption,
- Dense Region Caption,
- Object Detection,
- OCR,
- Caption to Phrase Grounding,
- segmentation,
- Region proposal,
- OCR,
- OCR with Region.

关于Florence-2模型参数为开源模型，此提供其base-ft与large-ft供选择

模型性能与其大小有关，根据自身需求进行选择不同的模型参数

官方Florence-2模型参数链接为：

https://huggingface.co/microsoft/Florence-2-base-ft

https://huggingface.co/microsoft/Florence-2-large-ft

模型下载完整文件夹后放在root根目录即可，命名为```Florence-2-base-ft```与```Florence-2-base-ft```

# RetinaFace性能结果
### 网络架构对比
三种不同网络架构评估结果对比：
| 网络架构       | 难易程度 (Eazy) | 难易程度 (Medium) | 难易程度 (Hard) |
|----------------|-----------------|-------------------|-----------------|
| MobileNet0.25  | 0.907           | 0.881             | 0.738           |
| Resnet50       | 0.954           | 0.940             | 0.844           |
| ShuffleNetV2   | 0.877           | 0.847             | 0.679           |

### PR 曲线对比  
<table align="center">
  <tr>
    <!-- MobileNet0.25 结果列 -->
    <td style="text-align: center;">
      <img src="detection_results/MobileNet0.25.png" alt="MobileNet0.25 Result" width="300">  
      <p style="text-align: center; font-weight: bold;">MobileNet0.25</p>
    </td>
    <!-- ResNet50 结果列 -->
    <td style="text-align: center;">
      <img src="detection_results/ResNet50.png" alt="ResNet50 Result" width="300">  
      <p style="text-align: center; font-weight: bold;">ResNet50</p>
    </td>
    <!-- ShuffleNetV2 结果列 -->
    <td style="text-align: center;">
      <img src="detection_results/ShuffleNetV2.png" alt="ShuffleNetV2 Result" width="300">  
      <p style="text-align: center; font-weight: bold;">ShuffleNetV2</p>
    </td>
  </tr>
</table>

### 效果展示

![单场景检测](detection_results/detection_1.jpg)  

  
![多场景检测](detection_results/detection_2.jpg)  


# 模型训练

本项目提供RetinaFace三种网络架构resnet50、mobilenet0.25、shufflenetv2_Final进行对比实验

训练命令：
```text
python Pytorch_Retinaface/train.py --network resnet50

python Pytorch_Retinaface/train.py --network mobile0.25

python Pytorch_Retinaface/train.py --network shufflenetv2
```

yolov8基线模型对比实验命令：

```text
python yolo111111.py
```

# 测试评估

一、RetinaFace模型评估 widerface val：

1.需要先生成相应的文本文件

```text
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50 or shufflenetv2
```

2.在widerface_evaluate中进行评估
```text
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```

二、RetinaFace模型评估FDDB：

数据集目录应为：
```text
./data/FDDB/images/
```
测试命令：
```text
python test_fddb.py --trained_model weight_file --network mobile0.25 or resnet50 or shufflenetv2
```
其中：

- --trained_model 指定训练好的模型权重文件路径
- --network 指定 backbone 网络，可选 mobile0.25 或 resnet50 或 shufflenetv2

三、基线模型yolov8评估widerface val：
1.需要先生成相应的文本文件
```text
python yolo_test.py
```

2.在widerface_evaluate中进行评估
```text
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
（与RetinaFace测试相同）
# 大规模人群画像实现流程

1.RetinaFace模型进行人脸检测

使用 detect_single_image.py 脚本对单张图片进行人脸检测

示例命令
```text
python Pytorch_Retinaface/detect_single_image.py --trained_model ./weights/Resnet50_Final.pth --image_path ./test.jpg
```
2.得到大规模人群的人脸检测图像



3.进行切割的基础上，使用Florence-2模型进行语义描述

对单一人脸图像进行详细描述任务的命令：
```text
python xiazai.py
```

其中，xiazai.py为Florence-2的执行py，其中```task_prompt = "<DETAILED_CAPTION>"```可进行需求修改

如需要更详细的描述：<MORE_DETAILED_CAPTION>，可以进行替换。执行后会对单一图像进行详细的英文人脸描述
