# EBiM Task 3 YOLOv8 训练指南

## 概述

本目录包含 Task 3 (辅助生活喂饭) 的 YOLOv8 目标检测训练流水线。

## 10 类物体定义

| ID | 类别     | 说明       |
| -- | ------ | -------- |
| 0  | plate  | 餐盘       |
| 1  | cup    | 杯子       |
| 2  | bowl   | 碗        |
| 3  | spoon  | 勺子       |
| 4  | head   | 头部/人脸安全区域 |
| 5  | sink   | 水槽区域 |
| 6  | recycling_bin | 豆子回收容器 |
| 7  | seat_area | 用餐座位/摆桌区域 |
| 8  | simple_tray | 托盘 |
| 9  | bean | 豆子 |

## 目录结构

```
submission/
├── train_yolo.py              # 训练脚本
├── yolo_detector.py           # 推理模块
├── extract_frames_for_labeling.py  # 从视频提取帧
├── models/
│   └── ebim_task3.pt          # 训练好的模型（需训练生成）
└── data/
    └── task3_yolo/
        ├── task3.yaml         # 数据集配置
        ├── images/
        │   ├── train/         # 训练图片
        │   └── val/           # 验证图片
        └── labels/
            ├── train/         # 训练标注 (YOLO格式)
            └── val/           # 验证标注 (YOLO格式)
```

## 训练流程

### 步骤 1: 提取视频帧

从实机视频数据中提取帧用于标注：

```bash
# 从头部摄像头提取帧（全局视角，适合检测桌面物体）
python3 extract_frames_for_labeling.py \
    --cameras head \
    --interval 30 \
    --episodes 000,001,002,003,005,010,020,030 \
    --output submission/data/task3_yolo/images/train
```

### 步骤 2: 标注数据

使用 LabelImg 或任何支持 YOLO 格式的标注工具：

```bash
pip3 install labelImg
labelImg
```

标注规则：

- 每个物体一个边界框

- 标注文件与图片同名，`.txt` 后缀

- 格式: `class_id cx cy w h` (归一化坐标)

### 步骤 3: 划分数据集

将数据按 80/20 划分为训练集和验证集：

```bash
# 移动 20% 图片到 val
cd submission/data/task3_yolo/images/train
files=(*.jpg)
val_count=$(( ${#files[@]} / 5 ))
for f in "${files[@]:0:$val_count}"; do
    mv "$f" ../val/
    mv "../../labels/train/${f%.jpg}.txt" "../../labels/val/" 2>/dev/null
done
```

### 步骤 4: 开始训练

```bash
cd submission

# 小模型快速训练（推荐在 CPU/Mac 上调试）
python3 train_yolo.py --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640

# 大模型高精度训练（推荐在 GPU 服务器上）
python3 train_yolo.py --model yolov8m.pt --epochs 150 --batch 32 --imgsz 640 --device 0
```

### 步骤 5: 验证模型

训练完成后，模型自动保存到 `models/ebim_task3.pt`，同时生成 SHA256 校验文件。

## 在 RTX 4090 服务器上训练

```bash
# 1. 上传标注好的数据到服务器
scp -r submission/data/task3_yolo root@server:/root/gpufree-data/

# 2. 在服务器上训练
cd /root/gpufree-data/
python3 train_yolo.py --model yolov8m.pt --epochs 150 --batch 64 --device 0

# 3. 下载训练好的模型
scp root@server:/root/gpufree-data/models/ebim_task3.pt submission/models/
```

## 模型推理

```python
from yolo_detector import detect_objects, select_best_detection, get_object_3d_position
import cv2

# 加载图像
image = cv2.imread("test.jpg")

# 检测物体
detections = detect_objects(image, conf_threshold=0.35)

# 找餐盘
plate = select_best_detection(detections, "plate", image_center=(320, 240))

# 3D 定位（需要深度图）
if plate and depth_image is not None:
    pos_3d = get_object_3d_position(plate, depth_image, camera_info)
```

## 实机部署注意事项

1. **摄像头选择**: 头部摄像头（全局视角）用于桌面物体检测，腕部摄像头用于精细操作
2. **分辨率**: 头部 336x188，腕部 640x480
3. **帧率**: 头部 \~10fps，腕部 \~30fps
4. **深度图**: ZED 头部摄像头提供深度，腕部可能没有
5. **推理速度**: yolov8n 在 CPU 上 \~50ms，在 GPU 上 \~5ms

## 紧急方案（现场来不及训练）

如果现场没有时间训练完整模型，可以：

1. 不要把通用 COCO YOLOv8 权重当作 Task 3 检测器；运行时会拒绝类别不匹配的 checkpoint。
2. 缺少经过验证的 `ebim_task3.pt` 时保持 fail-closed，不发布伪造的 3D 目标位姿。
3. `head`、`seat_area` 和 `recycling_bin` 必须在目标场地数据中标注并验证后才可用于策略。
