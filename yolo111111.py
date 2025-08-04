from ultralytics import YOLO
import os
import torch

torch.cuda.empty_cache()
# 确保模型文件存在
model_path = 'models/yolov8n.pt'
if not os.path.exists(model_path):
    print(f"模型文件 {model_path} 不存在，请检查路径或手动下载模型文件！")
    exit(1)

# 加载预训练模型
model = YOLO(model_path)

# 训练模型
results = model.train(
    data='widerface.yaml',  # 数据集配置文件
    epochs=100,              # 训练轮次
    imgsz=640,              # 输入图像尺寸
    batch=32,               # 批次大小
    name='yolov8n_widerface'  # 训练结果保存名称
)