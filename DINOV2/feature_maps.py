
"""
    本文件程序主要是在模型训练和评估等的过程中，提取模型架构的相关特征图的操作，以便于更好的理解模型的架构
"""
from typing import Callable, Any, List
import torch.nn as nn


class MyFeatureMapHook:
    # 特征图提取类的初始化函数
    def __init__(self, model: nn.Module, outputDir: str, imgIndex: int) -> None:
        self.model = model
        self.outputDir = outputDir
        self.imgIndex = imgIndex
        self.featureMaps = {}
        self.hooks = []

    # 提取保留数据的方法
    def _hook_fn(self, layerName: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, inputData: Any, outputData: Any) -> None:
            self.featureMaps[layerName] = outputData

        return hook

    # 为指定层注册钩子
    def register_hooks(self, layerNames:List[str]) -> None:
        for name, module in self.model.named_modules():
            if name in layerNames:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    # 保存所有捕获的特征图





if __name__ == '__main__':
    exampleHook = MyFeatureMapHook(None, None, None)
