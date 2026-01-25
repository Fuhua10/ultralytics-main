from ultralytics import YOLO

# 自动下载 + 加载 yolo26n.pt（如果本地没有）
model = YOLO('yolo12n.pt')

# 打印模型信息
#print(model)

# 查看任务类型（检测/分割/姿态等）
print(f"Model task: {model.task}")

# 查看配置
print(f"Model config: {model.cfg}")

#查看参数
total = sum(p.numel() for p in model.parameters())
print(f"Params: {total / 1e6:.2f}M") #yolo26n=2.57m