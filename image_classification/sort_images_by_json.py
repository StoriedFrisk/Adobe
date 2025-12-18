import os
import shutil
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 那个混乱的、存放所有原图的文件夹路径
source_images_dir = r'D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\images' 

# 2. 你辛苦建立的四个放 JSON 的文件夹路径
# 格式：'类别名': 'JSON文件夹路径'
# 注意：类别名要用英文 (最好和 YOLO 训练时的名字一致)
json_folders = {
    'mold':     r'D:\Downloads\\Compressed\\bamboo_saw\\labeled\\霉变',
    'crack':    r'D:\Downloads\\Compressed\\bamboo_saw\\labeled\\严重开裂',
    'wormhole': r'D:\Downloads\\Compressed\\bamboo_saw\\labeled\\虫眼',
    'missing':  r'D:\Downloads\\Compressed\\bamboo_saw\\labeled\\边壁缺失'
}

# 3. 整理后的图片想放在哪里 (会自动创建)
target_dataset_dir = r'D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\Final'
# ===========================================

def main():
    # 预先支持的图片后缀
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 统计数据
    moved_count = 0
    missing_count = 0

    print("开始根据JSON整理图片...")

    # 遍历你在配置里写的每一个类别
    for class_name, json_dir in json_folders.items():
        print(f"\n正在处理类别: {class_name} ...")
        
        # 目标文件夹D:\Downloads\\Compressed\\bamboo_saw\\labeled\\霉变
        # 我们把所有整理好的图都先放进train里，后面让YOLO自己分验证集
        target_dir = os.path.join(target_dataset_dir, 'train', class_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 获取该文件夹下所有的 JSON 文件
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"文件夹 {json_dir} 里没有发现JSON文件，跳过此文件夹...")
            continue

        for json_file in tqdm(json_files):
            # 假设 json 文件名是 "123.json"，那图片名应该是 "123"
            file_base_name = os.path.splitext(json_file)[0]
            
            # 去图片仓库里找对应的图
            found_img = None
            for ext in valid_exts:
                # 尝试 123.jpg, 123.png 等
                possible_path = os.path.join(source_images_dir, file_base_name + ext)
                if os.path.exists(possible_path):
                    found_img = possible_path
                    break
            
            if found_img:
                # 复制图片到新家
                # 目标路径D:\Downloads\\Compressed\\bamboo_saw\\labeled\\霉变
                target_path = os.path.join(target_dir, os.path.basename(found_img))
                shutil.copy2(found_img, target_path) # copy2 保留文件修改时间
                moved_count += 1
            else:
                # 如果只有 JSON 但找不到图 (可能名字对不上)
                missing_count += 1
                print(f"找不到对应的图片: {file_base_name}，检查文件夹...?(在 {source_images_dir})")

    print(f"\n整理完成！")
    print(f"成功归类图片: {moved_count} 张")
    print(f"未找到原图: {missing_count} 张")
    print(f"数据集已准备好: {target_dataset_dir}")
    print("所有图片位于Final文件夹中")

if __name__ == '__main__':
    main()