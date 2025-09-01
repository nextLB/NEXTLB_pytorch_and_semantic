import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.cm as cm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_folder(folderPath: str):
    """
    清空文件夹内的所有内容，但保留文件夹本身

    参数:
        folder_path: 要清空的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folderPath):
        print(f"错误: 文件夹 '{folderPath}' 不存在")
        return

    # 检查路径是否指向一个文件夹
    if not os.path.isdir(folderPath):
        print(f"错误: '{folderPath}' 不是一个文件夹")
        return

    # 遍历文件夹内的所有内容
    for item in os.listdir(folderPath):
        item_path = os.path.join(folderPath, item)

        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                print(f"已删除文件: {item_path}")
            # 如果是文件夹，递归删除整个文件夹
            elif os.path.isdir(item_path):
                # 使用shutil.rmtree删除文件夹及其内容
                import shutil
                shutil.rmtree(item_path)
                print(f"已删除文件夹: {item_path}")
        except Exception as e:
            print(f"删除 {item_path} 时出错: {e}")

    print(f"文件夹 '{folderPath}' 内容已清空")


saveFeaturePicturePath = './results/feature_picture'
os.makedirs(saveFeaturePicturePath, exist_ok=True)
clear_folder(saveFeaturePicturePath)


def visualize_feature_maps(feature_maps, layer_name):
    """可视化特征图，使用PIL保存图像"""
    savePath = ''
    maxCount = 5

    if feature_maps is None:
        print(f"No feature maps")
        return

    # 创建保存目录（如果不存在）
    os.makedirs(saveFeaturePicturePath, exist_ok=True)

    # 将特征图从GPU移动到CPU并转换为numpy数组
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.cpu().numpy()

    # 获取viridis色彩映射
    viridis_cmap = cm.get_cmap('viridis')
    if len(feature_maps.shape) == 4:
        for i in range(min(feature_maps.shape[0], maxCount)):
            for j in range(min(feature_maps[i].shape[0], maxCount)):
                # 获取第j个通道的特征图
                feature_map = feature_maps[i][j]
                # 归一化特征图以便更好地显示
                if feature_map.max() - feature_map.min() > 0:
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

                # 应用viridis色彩映射
                # 将特征图应用色彩映射，得到RGBA图像
                colored_feature = viridis_cmap(feature_map)
                print(colored_feature.shape)

                # 转换为RGB格式（去掉alpha通道）并转为0-255范围
                colored_feature_rgb = (colored_feature[:, :, :3] * 255).astype(np.uint8)

                # 创建PIL图像
                img = Image.fromarray(colored_feature_rgb)

                # 保存图像
                savePath = os.path.join(saveFeaturePicturePath, f"{layer_name}_feature_map_batch_{i}_channel_{j}.png")
                img.save(savePath)

    elif len(feature_maps.shape) == 2:
        for i in range(min(feature_maps.shape[0], maxCount)):
            # 将1D向量重塑为2D图像
            # 找到最接近平方数的尺寸
            vector_length = feature_maps[i].shape[0]
            side_length = int(np.sqrt(vector_length))

            # 如果向量长度不是完全平方数，使用零填充
            if side_length * side_length < vector_length:
                side_length += 1
                padded_vector = np.zeros(side_length * side_length)
                padded_vector[:vector_length] = feature_maps[i]
                feature_map_2d = padded_vector.reshape(side_length, side_length)
            else:
                feature_map_2d = feature_maps[i].reshape(side_length, side_length)

            # 归一化特征图以便更好地显示
            if feature_map_2d.max() - feature_map_2d.min() > 0:
                feature_map_2d = (feature_map_2d - feature_map_2d.min()) / (
                        feature_map_2d.max() - feature_map_2d.min() + 1e-8)

            # 应用viridis色彩映射
            colored_feature = viridis_cmap(feature_map_2d)
            print(colored_feature.shape)

            colored_feature_rgb = (colored_feature[:, :, :3] * 255).astype(np.uint8)

            # 创建PIL图像
            img = Image.fromarray(colored_feature_rgb)

            # 保存图像
            savePath = os.path.join(saveFeaturePicturePath, f"{layer_name}_batch_{i}.png")
            img.save(savePath)


transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    transforms.Normalize((0.5, ), (0.5, ))  # 归一化到[-1, 1]   由于MNIST的是单通道图像，所以只需要指定一个维度即可
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)





class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层：输入1通道，输出32通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 定义卷积层：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入大小 = 特征图大小 * 通道数
        self.fc2 = nn.Linear(128, 10)  # 10 个类别

        self.feature_maps_before_conv1 = None
        self.feature_maps_after_conv1 = None
        self.feature_maps_before_conv2 = None
        self.feature_maps_after_conv2 = None

        self.feature_maps_before_fc1 = None
        self.feature_maps_after_fc1 = None
        self.feature_maps_before_fc2 = None
        self.feture_maps_after_fc2 = None

        # 注册钩子
        self._register_hooks()

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = x.view(-1, 64 * 7 * 7) # 展平操作

        # 保存展平后的特征图（fc1的输入）
        self.feature_maps_before_fc1 = x.detach().clone()

        x = F.relu(self.fc1(x))  # 全连接层 + ReLU

        # 保存fc1后的特征图
        self.feature_maps_after_fc1 = x.detach().clone()

        # 保存fc2前的特征图（与fc1后相同）
        self.feature_maps_before_fc2 = x.detach().clone()

        x = self.fc2(x)  # 全连接层输出

        # 保存fc2后的特征图
        self.feature_maps_after_fc2 = x.detach().clone()

        return x
    def _register_hooks(self):
        def forward_hook_before_conv1(module, input, output):
            self.feature_maps_before_conv1 = input[0].detach().clone()
            return None

        def forward_hook_after_conv1(module, input, output):
            self.feature_maps_after_conv1 = output.detach().clone()
            return None

        def forward_hook_before_conv2(module, input, output):
            self.feature_maps_before_conv2 = input[0].detach().clone()
            return None

        def forward_hook_after_conv2(module, input, output):
            self.feature_maps_after_conv2 = output.detach().clone()
            return None

        self.conv1.register_forward_hook(forward_hook_before_conv1)
        self.conv1.register_forward_hook(forward_hook_after_conv1)
        self.conv2.register_forward_hook(forward_hook_before_conv2)
        self.conv2.register_forward_hook(forward_hook_after_conv2)




# 创建模型实例
model = SimpleCNN()
model = model.to(device)  # 将模型移动到GPU上

criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 学习率和动量

num_epochs = 5
model.train()  # 设为训练模式

batchIndex = 0
for epoch in range(num_epochs):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)  # 将图像数据移动到GPU上
        labels = labels.to(device)  # 将标签数据移动到GPU上
        # 前向传播
        outputs = model(images)
        if batchIndex == 0:
        # 可视化并保存不同阶段的特征图
            visualize_feature_maps(
                model.feature_maps_before_conv1,
                'conv1_before'
            )

            visualize_feature_maps(
                model.feature_maps_after_conv1,
                'conv1_after'
            )

            visualize_feature_maps(
                model.feature_maps_before_conv2,
                'conv2_before'
            )

            visualize_feature_maps(
                model.feature_maps_after_conv2,
                'conv2_after'
            )

            visualize_feature_maps(
                model.feature_maps_before_fc1,
                'fc1_before'
            )

            visualize_feature_maps(
                model.feature_maps_after_fc1,
                'fc1_after'
            )

            visualize_feature_maps(
                model.feature_maps_before_fc2,
                'fc2_before'
            )

            visualize_feature_maps(
                model.feature_maps_after_fc2,
                'fc2_after'
            )

        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batchIndex += 1
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


model.eval()  # 设置为评估模式
correct = 0
total = 0

with torch.no_grad():  # 评估时不需要计算梯度
    for images, labels in test_loader:
        images = images.to(device)  # 将图像数据移动到GPU上
        labels = labels.to(device)  # 将标签数据移动到GPU上
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



