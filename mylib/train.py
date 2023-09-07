import argparse
import torch
import torch.optim as optim
from model.yolo import YOLO
from model.preprocessing import Preprocessor
from utils.utils import init_seeds
from utils import torch_utils
from utils.parse_config import *
from torch.utils.data import DataLoader
import yaml
def train(
        net_config_path,
        data_config_path,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        ):
    num_epochs = epochs
    device = torch_utils.select_device()
    data_config = parse_data_config(data_config_path)
    # 加载模型和预处理器的配置
    with open(net_config_path, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    with open('config/preprocessing_config.yaml', 'r') as f:
        preprocessing_config = yaml.load(f, Loader=yaml.FullLoader)

    # 初始化模型和预处理器
    model = YOLO(model_config['model'], img_size)
    preprocessor = Preprocessor(preprocessing_config['preprocessing'])

    # 设置损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载训练数据集，这里假设您有一个自定义的数据加载器
    train_loader = DataLoader(data_config, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--data-config', type=str, default='cfg/coco.data', help='path to data config file')
    parser.add_argument('--cfg', type=str, default='config/model_config.yaml', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--weights-path', type=str, default='weights', help='path to store weights')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--report', action='store_true', help='report TP, FP, FN, P and R per batch (slower)')
    parser.add_argument('--freeze', action='store_true', help='freeze darknet53.conv.74 layers for first epoche')
    parser.add_argument('--var', type=float, default=0, help='optional test variable')
    args = parser.parse_args()
    init_seeds()

    torch.cuda.empty_cache()
    train(
        args.cfg,
        args.data_config,
        img_size=args.img_size,
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
