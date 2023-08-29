from PIL import Image
import torch
from mylib.model.model import YOLOv3
from mylib.model.preprocessing import Preprocessor
import yaml

# 加载模型和预处理器的配置
with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

with open('config/preprocessing_config.yaml', 'r') as f:
    preprocessing_config = yaml.load(f, Loader=yaml.FullLoader)

# 初始化模型和预处理器
model = YOLOv3(model_config['model'])
preprocessor = Preprocessor(preprocessing_config['preprocessing'])

# 加载训练好的模型权重
model.load_state_dict(torch.load('path_to_saved_model.pth'))
model.eval()

# 处理输入图像
input_image = Image.open('input_image.jpg')
input_tensor = preprocessor.preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度

# 进行推理
with torch.no_grad():
    detections = model(input_tensor)
    # 在这里根据模型输出获取目标检测结果

# 处理检测结果，绘制边界框等


# 显示结果图像或保存结果

