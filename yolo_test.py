from ultralytics import YOLO
import os
import cv2
import numpy as np

# 加载训练好的YOLOv8模型
model = YOLO('weights/yolov8n_widerface19/weights/best.pt')

# WiderFace验证集图片路径
val_image_dir = './data/widerface/val/images/'
# 预测结果保存路径
prediction_dir = './widerface_txt/'

# 确保预测结果保存目录存在
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

# 获取所有事件目录
events = os.listdir(val_image_dir)
for event in events:
    event_dir = os.path.join(val_image_dir, event)
    if os.path.isdir(event_dir):
        event_pred_dir = os.path.join(prediction_dir, event)
        if not os.path.exists(event_pred_dir):
            os.makedirs(event_pred_dir)
        
        # 获取当前事件下的所有图片
        images = os.listdir(event_dir)
        for img_name in images:
            img_path = os.path.join(event_dir, img_name)
            img = cv2.imread(img_path)
            results = model(img)
            
            # 保存预测结果到TXT文件
            txt_path = os.path.join(event_pred_dir, img_name.replace('.jpg', '.txt'))
            with open(txt_path, 'w') as f:
                f.write(img_path + '\n')
                f.write(str(len(results[0].boxes)) + '\n')
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    line = f'{xyxy[0]} {xyxy[1]} {xyxy[2]-xyxy[0]} {xyxy[3]-xyxy[1]} {conf}\n'
                    f.write(line)