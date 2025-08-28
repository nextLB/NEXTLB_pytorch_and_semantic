
import pkg_resources
import importlib

# 定义需要检查的库及其导入名称
packages = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "segmentation-models-pytorch": "segmentation_models_pytorch",
    "numpy": "numpy",
    "pandas": "pandas",
    "pillow": "PIL",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "albumentations": "albumentations",
    "pytorch_lightning": "pytorch_lightning",
    "ultralytics": "ultralytics",
    "utm": "utm",
    "pyproj": "pyproj",
    "scikit-learn": "sklearn",
    "imageio": "imageio",
    "scikit-image": "skimage",
    "cartopy": "cartopy",
    "certifi": "certifi",
    "contourpy": "contourpy",
    "cycler": "cycler",
    "fonttools": "fontTools",
    "kiwisolver": "kiwisolver",
    "packaging": "packaging",
    "pyparsing": "pyparsing",
    "shapely": "shapely",
    "six": "six",
    "pyshp": "shapefile",
    "pyinstaller": "PyInstaller"
}

def get_version(module_name, import_name):
    try:
        # 尝试通过pkg_resources获取版本（更准确）
        version = pkg_resources.get_distribution(module_name).version
        return version
    except:
        try:
            # 如果失败，尝试导入模块并获取__version__属性
            module = importlib.import_module(import_name)
            if hasattr(module, '__version__'):
                return module.__version__
            else:
                return "版本信息不可用（但成功导入）"
        except ImportError:
            return "未安装"
        except Exception as e:
            return f"错误: {str(e)}"

print("正在检查库安装情况...\n")
print("{:<30} {:<20} {:<10}".format("库名称", "版本", "状态"))
print("-" * 70)

all_success = True

for pkg_name, import_name in packages.items():
    version = get_version(pkg_name, import_name)

    if "未安装" in version or "错误" in version:
        status = "失败"
        all_success = False
    else:
        status = "成功"

    print("{:<30} {:<20} {:<10}".format(pkg_name, version, status))

print("\n" + "= " *70)
if all_success:
    print("所有库都已成功安装！")
else:
    print("部分库安装可能存在问题，请检查上述输出。")

# 额外检查CUDA是否对PyTorch可用
try:
    import torch
    print(f"\nPyTorch CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
except:
    print("\n无法检查PyTorch CUDA状态")



