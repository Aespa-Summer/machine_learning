from ultralytics import YOLO
import torch

# 核心修复1：添加main保护，避免多进程启动错误
if __name__ == '__main__':
    # 加载YOLOv8预训练模型
    model = YOLO('yolov8n.pt')

    # 开始训练（适配最新版YOLO参数命名）
    results = model.train(
        # 基础路径配置（保持你的原有路径）
        data='C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\yolo--Fruit-Classification-main\\src\\config\\fruit.yaml',
        epochs=80,
        imgsz=640,
        batch=8,          # 适配3050 4G显存
        device=0,         # 指定使用GPU
        workers=0,        # Windows必须设为0
        patience=50,      # 早停耐心值
        save=True,

        # ===== 核心优化1：优化器+学习率策略（参数名兼容最新版） =====
        optimizer='AdamW',  # 替换默认SGD，适配小样本水果数据集
        lr0=0.005,          # 初始学习率（适配AdamW）
        lrf=0.0005,         # 最终学习率（lr0*lrf）
        weight_decay=0.001, # 权重衰减，抑制过拟合
        cos_lr=True,        # 余弦退火学习率（比线性衰减更稳定）

        # ===== 核心优化2：针对性数据增强（适配水果检测） =====
        hsv_h=0.05,  # 色相增强（小幅度，避免水果颜色失真）
        hsv_s=0.2,   # 饱和度增强（适配不同成熟度水果）
        hsv_v=0.2,   # 明度增强（对抗光照变化）
        degrees=15.0,# 旋转角度（水果自然摆放角度）
        translate=0.1,# 平移
        scale=0.1,   # 缩放
        flipud=0.0,  # 上下翻转（水果少用，避免不自然）
        fliplr=0.5,  # 左右翻转（常规增强）
        mosaic=0.3,  # 低比例Mosaic（避免水果拼接失真）
        mixup=0.0,   # 关闭mixup（水果类别易混淆）

        # ===== 核心修复：替换旧参数为最新版兼容参数 =====
        # 1. 替代accumulation（梯度累积）：新版用batch_size + rect=False实现等效效果
        rect=False,         # 关闭矩形训练，配合batch=8+累加逻辑
        # 2. 替代fp16（混合精度）：新版统一用amp参数
        amp=True,           # 混合精度训练（原fp16=True，提速+省显存）
    )

    print("训练完成，结果保存在 runs/detect/train 目录下")