import os
import random
import shutil
from pathlib import Path

# ================= 配置区域 =================
SOURCE_DIR = r"D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\images"  # 你的40000张图片所在的文件夹路径
TARGET_DIR = r"D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\Sample"            # 抽取出来的图片存放路径
SAMPLE_SIZE = 1000                        # 想要抽取的数量
MODE = "random"                           # 模式: "random" (随机) 或 "interval" (等间距/视频帧)
# ===========================================

def sample_images():
    # 1. 准备路径
    source_path = Path(SOURCE_DIR)
    target_path = Path(TARGET_DIR)
    
    if not source_path.exists():
        print(f"错误：源文件夹 {SOURCE_DIR} 不存在")
        return

    # 创建目标文件夹
    target_path.mkdir(parents=True, exist_ok=True)

    # 2. 获取所有图片文件
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = [f for f in source_path.iterdir() if f.suffix.lower() in valid_extensions and f.is_file()]
    
    total_images = len(all_images)
    print(f"图片总数: {total_images} 张")

    if total_images < SAMPLE_SIZE:
        print("图片总数少于目标抽取数，将复制所有图片...")
        selected_images = all_images
    else:
        # 3. 根据模式抽取
        if MODE == "random":
            print("正在进行随机抽样...")
            selected_images = random.sample(all_images, SAMPLE_SIZE)
        elif MODE == "interval":
            print("正在进行等间距抽样..")
            # 按文件名排序，保证顺序
            all_images.sort(key=lambda x: x.name)
            step = total_images // SAMPLE_SIZE
            selected_images = [all_images[i] for i in range(0, total_images, step)][:SAMPLE_SIZE]

    # 4. 复制文件
    print(f"复制 {len(selected_images)} 张图片到 {TARGET_DIR} ...")
    
    count = 0
    for img in selected_images:
        try:
            shutil.copy2(img, target_path / img.name)
            count += 1
            if count % 100 == 0:
                print(f"已复制 {count} 张...")
        except Exception as e:
            print(f"复制 {img.name} 失败: {e}")

    print(f"成功抽取 {count} 张图片到 '{TARGET_DIR}' 文件夹...")

if __name__ == "__main__":
    sample_images()