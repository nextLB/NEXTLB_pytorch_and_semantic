

"""
    本文件程序，主要是实现DINOV2过程中所需要用到的自定义工具函数
"""
import os
import re
import shutil
import matplotlib.pyplot as plt
import config_path

# The minimum value is selected and its subscript in the original array is returned
def get_min_value(Array):
    length = len(Array)
    if length == 0:
        # print('It is not valid for the array length to be empty.')
        return -1
    elif length == 1:
        return 0
    else:
        min_value = float('inf')
        min_index = -1
        for i in range(length):
            if Array[i] <= min_value:
                min_value = Array[i]
                min_index = i
        return min_index

# The maximum value is selected and its subscript in the original array is returned
def get_max_value(Array):
    length = len(Array)
    if length == 0:
        # print('It is not valid for the array length to be empty.')
        return -1
    elif length == 1:
        return 0
    else:
        max_value = float('inf')
        max_index = -1
        for i in range(length):
            if Array[i] >= max_value:
                max_value = Array[i]
                max_index = i
        return max_index


# 清空文件夹内的所有内容
def clear_folder(folderPath: str):

    # 遍历文件夹内的所有内容
    for item in os.listdir(folderPath):
        itemPath = os.path.join(folderPath, item)
        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(itemPath) or os.path.islink(itemPath):
                os.unlink(itemPath)
                print(f'已删除文件夹: {itemPath}')
            elif os.path.isdir(itemPath):
                # 使用 shutil.rmtree 删除文件夹及其内容
                shutil.rmtree(itemPath)
                print(f"已删除文件夹： {itemPath}")

        except Exception as e:
            print(f"删除 {itemPath} 时出错： {e}")

    print(f"文件夹 {folderPath} 内容已清空")


def save_S_S_transformer_datasets(batchIndex, index, image, mask):
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示图像      顺序： CHW -> HWC
    ax1.imshow(image.permute(1, 2, 0).cpu.numpy())  # 将 torch.Tensor张量格式的图像[C, H, W]转换为[H, W, C]格式的图像进行显示
    ax1.set_title('Image')
    ax1.axis('off')

    # 显示掩码 (使用不同颜色表示不同类别)
    # 若 mask 的形状为 (1, H, W)（可能包含一个冗余的通道维度），squeeze() 后会变为 (H, W)；
    # 若 mask 的形状为 (B, 1, H, W)（批量处理的掩码），squeeze() 后会变为 (B, H, W)（仅移除维度为 1 的通道维）。
    ax2.imshow(mask.squeeze().cpu().numpy(), cmap='viridis')    # squeeze方法移除张量中所有维度为1的张量，举例说明： (1, H, W) -> (H, W) 或者 (B, 1, H, W) -> (B, H, W)
    ax2.set_title('Mask')
    ax2.axis('off')

    plt.tight_layout()
    # plt.show()

    # 获取当前图形的dpi(清晰度) (通常是默认的100)
    currentDPI = plt.gcf().dpi

    # 使用当前dpi保存，不改变边界框
    plt.savefig(
        f"{config_path.SAVE_S_S_TRANSFORMER_DATASETS_PATH}/transformer_{batchIndex}_{index}.png",
        dpi=currentDPI,
        bbox_inches=None   # 不调整边界框，保持原始尺寸
    )


def save_pretrained_transformer_datasets(originalImg, augmentedImg, cropIndex, batchIndex, savePath, batchNumber):
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 转换通道维度
    originalImg = originalImg.permute(1, 2, 0).detach().cpu().numpy()
    augmentedImg = augmentedImg.permute(1, 2, 0).detach().cpu().numpy()

    ax1.imshow(originalImg, cmap='viridis')
    ax1.set_title('original image')
    ax1.axis('off')     # 关闭坐标轴
    ax2.imshow(augmentedImg, cmap='viridis')
    ax2.set_title(f'augmented image_{cropIndex}')
    ax2.axis('off') # 关闭坐标轴
    print('originalImages.shape: ', originalImg.shape)
    print('augmentedImages.shape: ', augmentedImg.shape)
    plt.tight_layout()

