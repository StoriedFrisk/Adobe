import os
import shutil
from ultralytics import YOLO
from tqdm import tqdm

# ================= 配置 =================
# 1. 刚才训练好的模型路径
model_path = 'runs/classify/my_feature_cls/weights/best.pt'

# 2. 待分类的图片文件夹
source_dir = 'unlabeled_images'

# 3. 分类结果输出到哪里
output_dir = 'sorted_results'

# 4. 置信度阈值 (0-1)
# 只有当模型非常确信(比如>0.8)时才自动归类，
# 如果不确信(比如0.4)，最好放到一个 'unsure' 文件夹人工检查
conf_threshold = 0.7 
# =======================================

def main():
    # 加载模型
    model = YOLO(model_path)
    
    # 获取类别名称字典 {0: 'feature_A', 1: 'feature_B', ...}
    class_names = model.names

    # 预先创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name in class_names.values():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'unsure'), exist_ok=True)

    # 获取所有图片
    # 支持 jpg, png, bmp, jpeg
    imgs = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    print(f"准备处理 {len(imgs)} 张图片...")

    # 批量推理 (stream=True 可以防止内存溢出)
    # 这里的 batch=16 可以加快速度
    results = model.predict(source=source_dir, stream=True, conf=0.5)

    for result in results:
        # 获取当前处理的图片路径
        img_path = result.path
        file_name = os.path.basename(img_path)
        
        # 获取预测概率最高的类别索引
        probs = result.probs
        top1_index = probs.top1
        top1_conf = probs.top1conf.item() # 最高置信度
        
        # 获取类别名
        class_name = class_names[top1_index]

        # 决定目标文件夹
        if top1_conf >= conf_threshold:
            target_folder = os.path.join(output_dir, class_name)
            print(f"[{class_name}] {top1_conf:.2f} -> {file_name}")
        else:
            target_folder = os.path.join(output_dir, 'unsure')
            print(f"[不确定] {top1_conf:.2f} -> {file_name}")

        # 移动文件 (如果你只想复制，把 move 改成 copy)
        target_path = os.path.join(target_folder, file_name)
        shutil.move(img_path, target_path)

    print("分类完成！请检查 sorted_results 文件夹。")

if __name__ == '__main__':
    main()