from ultralytics import YOLO
import torch
import json          
import os           
# 1. 加载优化前后模型并获取验证结果
def get_metrics(model_path):
    model = YOLO(model_path)
    val_results = model.val(
        # 替换为test.py中一致的fruit.yaml绝对路径
        data='C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\yolo--Fruit-Classification-main\\src\\config\\fruit.yaml',
        split='test',
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=False,
        batch=8,  # 适配3050 4G显存（与test.py保持一致）
        imgsz=640 # 与训练/测试参数对齐
    )
    return {
        'mAP@0.5': round(val_results.box.map50, 4),
        'mAP@0.5:0.95': round(val_results.box.map, 4),
        'Precision': round(val_results.box.mp, 4),
        'Recall': round(val_results.box.mr, 4)
    }

# 2. 获取优化前后指标（替换为实际模型绝对路径）
before_metrics = get_metrics('yolov8n.pt')  # 原始预训练模型（自动下载到本地）
# 替换为test.py中训练好的best.pt绝对路径
after_metrics = get_metrics('C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\runs\\detect\\train2\\weights\\best.pt')

# 3. 计算变化幅度
change_metrics = {}
for k in before_metrics.keys():
    change = (after_metrics[k] - before_metrics[k]) * 100
    change_metrics[k] = f"+{change:.2f} 个百分点" if change >=0 else f"{change:.2f} 个百分点"

# 4. 生成对比表格
df = pd.DataFrame({
    '评估指标': list(before_metrics.keys()),
    '优化前': list(before_metrics.values()),
    '优化后': list(after_metrics.values()),
    '变化幅度': list(change_metrics.values())
})

# 5. 保存表格（指定绝对路径，避免路径错误）
save_csv_path = 'C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\runs\\metrics_comparison.csv'
# 确保保存目录存在
os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
df.to_csv(save_csv_path, index=False, encoding='utf-8')
print("指标对比表格：")
print(df)

# 6. 提取各类别AP值（使用优化后模型路径）
model = YOLO('C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\runs\\detect\\train2\\weights\\best.pt')
val_results = model.val(
    data='C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\yolo--Fruit-Classification-main\\src\\config\\fruit.yaml',
    split='test',
    verbose=False,
    batch=8,
    imgsz=640
)
class_ap = {}
for i, name in enumerate(val_results.names):
    class_ap[name] = round(val_results.box.ap50[i], 3)
print("\n各类别AP值：")
for k, v in class_ap.items():
    print(f"{k}: {v}")