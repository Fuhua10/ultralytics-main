from ultralytics import YOLO




def main():

    # 加载模型
    model = YOLO('yolo12n.pt')  # 或你选择的模型

    # 训练模型
    results = model.train(
        data="VisDrone.yaml",  # 数据配置文件路径
        epochs=300,  # 训练轮数
        imgsz=640,  # 输入图像大小
        batch=64,  # 批次大小
        workers=16,
        device=0,
        patience=300,
        optimizer='auto',
    )



if __name__ == '__main__':
    main()