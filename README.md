# geiniyichan-RetinaFace-and-Florence-2
该项目使用开源的人脸检测模型RetinaFace与语义描述模型Florence-2相结合，对大规模人群进行人脸检测并对其进行详细描述

项目思路：通过训练大规模人脸检测模型RetinaFace在复杂环境下的多人脸并行检测能力，并在检测出人脸与切割的基础上，继续使用谷歌开源的Florence-2模型中的图像描述任务，对人脸的细节与表情情绪进行更深入的描述，从而得到大规模人群画像的表达。

本项目对RetinaFace还进行了多网络架构的对比，包括了resnet50、mobilenet0.25、shufflenetv2_Final，并使用yolov8作为基线模型进行深入对比，验证其在复杂环境下大规模人脸检测的优越性


# 安装
本项目所需环境

PyTorch  2.5.1

Python  3.12(ubuntu22.04)

CUDA  12.4

（没提到的自行安装🤭）

并可以使用git clone https://github.com/geiniyichan/RetinaFace-and-Florence-2.git 进行克隆项目
# 数据集
需下载widerface数据集进行模型训练与测试

数据集目录格式：
```text
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
