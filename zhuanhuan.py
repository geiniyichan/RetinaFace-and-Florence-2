import os
import cv2

def convert_widerface_to_yolov8(input_file, image_dir, output_dir):
    """
    将 WiderFace 标注文件转换为 YOLO 格式。

    :param input_file: WiderFace 标注文件路径
    :param image_dir: 图像文件夹路径
    :param output_dir: 输出 YOLO 标注文件的目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # 读取图像文件名
        image_name = lines[i].strip()
        i += 1

        # 读取人脸数量
        num_faces = int(lines[i].strip())
        i += 1

        # 读取图像尺寸
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            i += num_faces
            continue

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 准备 YOLO 格式的标注
        yolo_lines = []
        for _ in range(num_faces):
            parts = lines[i].strip().split()
            x1, y1, w, h = map(float, parts[:4])
            x_center = (x1 + w / 2) / width
            y_center = (y1 + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)
            i += 1

        # 保存 YOLO 格式的标注文件
        output_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.txt')
        output_subdir = os.path.dirname(output_file)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)  # 确保目标文件夹存在

        with open(output_file, 'w') as out_f:
            out_f.write('\n'.join(yolo_lines))

# 使用示例
input_file = './data/widerface/val/wider_face_val_bbx_gt.txt'
image_dir = './data/widerface/val/images'
output_dir = './data/widerface/val/labels'
convert_widerface_to_yolov8(input_file, image_dir, output_dir)