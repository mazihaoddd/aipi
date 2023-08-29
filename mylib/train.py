import torch
import torch.optim as optim
from model.model import YOLOv3
from model.preprocessing import Preprocessor
import yaml

# 加载模型和预处理器的配置
with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

with open('config/preprocessing_config.yaml', 'r') as f:
    preprocessing_config = yaml.load(f, Loader=yaml.FullLoader)

# 初始化模型和预处理器
model = YOLOv3(model_config['model'])
preprocessor = Preprocessor(preprocessing_config['preprocessing'])

# 设置损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加载训练数据集，这里假设您有一个自定义的数据加载器
train_loader = ...

# 开始训练
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        data = preprocessor.preprocess(data)
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
