from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

def main():
    model = YOLO('yolov8m.pt') 
    # test yolov8m
    # yolov8m.pt,yolov8l.pt,yolov8x.pt
    model.train(
        data='bamboo.yaml', 
        epochs=100,
        imgsz=640, 
        batch=8, 
        workers=2,
        device=0,
        cls=4.0,
        name='bamboo_exp'
    )
    model.export(format='onnx')
if __name__ == '__main__':
    main()