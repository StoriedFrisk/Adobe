from ultralytics import YOLO

# 加载模型
model = YOLO('D:\\Documaents\\Adobe\\runs\\detect\\bamboo_exp7\\weights\\best.pt')

# 在验证集上评估
# split='val': 指定使用验证集
metrics = model.val(data='bamboo.yaml', split='val')

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")