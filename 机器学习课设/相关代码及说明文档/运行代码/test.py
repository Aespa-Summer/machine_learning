from ultralytics import YOLO
import warnings
import json
import os
warnings.filterwarnings('ignore')  # 忽略无关警告

def evaluate_test_set():
    # 1. 加载训练好的最佳模型（建议用best.pt，比last.pt效果好）
    model = YOLO(
        'C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\runs\\detect\\train2\\weights\\best.pt'
    )
    
    # 2. 执行测试集评估（关键参数：split='test'）
    test_results = model.val(
        data='C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\yolo--Fruit-Classification-main\\src\\config\\fruit.yaml',
        split='test',          # 明确指定评估测试集
        project='C:\\Users\\w\\Desktop\\yolo--Fruit-Classification-main\\runs\\detect',
        name='test_set_results',  # 测试集结果保存文件夹
        exist_ok=False,        # 防止覆盖历史结果
        save=True,             # 保存可视化图、混淆矩阵、PR曲线
        save_json=True,        # 保存详细指标到JSON文件
        imgsz=640,             # 与训练时一致
        batch=8,               # 适配3050 4G显存
        device=0,              # 使用GPU加速
        conf=0.001,            # 低置信度阈值
        iou=0.5                # IoU阈值0.5
    )

    # 3. 打印整体核心指标
    print("=" * 60)
    print("YOLOv8n 水果检测 - 测试集评估结果（整体）")
    print(f"mAP@0.5（核心指标）: {test_results.box.map50:.4f}")
    print(f"mAP@0.5:0.95（综合指标）: {test_results.box.map:.4f}")
    print(f"精确率（Precision）: {test_results.box.mp:.4f}")
    print(f"召回率（Recall）: {test_results.box.mr:.4f}")
    print("=" * 60)

    # 4. 新增：提取各类别指标（适配报告分析）
    class_names = ['apple', 'pear', 'banana', 'peach', 'carambola']
    print("\n各类别详细指标（mAP@0.5）：")
    for i, cls_name in enumerate(class_names):
        cls_ap50 = test_results.box.ap50[i]  # 单个类别的mAP@0.5
        print(f"{cls_name}（{i}）: {cls_ap50:.4f}")

    # 5. 保存指标到JSON文件（方便报告引用）
    metrics = {
        "整体指标": {
            "mAP@0.5": float(test_results.box.map50),
            "mAP@0.5:0.95": float(test_results.box.map),
            "Precision": float(test_results.box.mp),
            "Recall": float(test_results.box.mr)
        },
        "类别指标": {
            cls: float(test_results.box.ap50[i]) for i, cls in enumerate(class_names)
        }
    }
    json_path = os.path.join(test_results.save_dir, 'metrics_detail.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # 告知结果保存路径
    print(f"\n测试集结果文件保存至：{test_results.save_dir}")
    print("包含：混淆矩阵图、PR曲线、各类别指标JSON文件（直接用于报告）")

if __name__ == '__main__':
    evaluate_test_set()