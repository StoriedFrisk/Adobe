from ultralytics import YOLO

def main():
    # 1. 加载预训练的分类模型
    # yolov8n-cls.pt 是专门用于分类的最小模型
    model = YOLO('yolov8n-cls.pt') 
    # 2. 开始训练
    # data: 指向包含 train/ 和 val/ 的文件夹路径
    model.train(
        data='D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\Final',    # 你的数据文件夹路径
        epochs=50,         # 200张图跑50轮很快
        imgsz=224,         # 分类任务标准尺寸是 224x224
        batch=16,
        name='my_feature_cls'
    )

if __name__ == '__main__':
    main()