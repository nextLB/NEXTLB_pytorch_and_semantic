
#  环境配置说明
    在Linux系统下配置一下tkinter环境
        sudo apt update
        sudo apt install python3-tk
        验证以下是否安装成功: python3 -m tkinter
        如果成功安装，会弹出一个Tkinter的演示窗口。

    创建基于python3.11的虚拟环境
        conda create -n semantic_segmentation_pytorch python=3.11

    启动虚拟环境后安装下面的依赖与库等
    pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip3 install segmentation-models-pytorch==0.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.26.4
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas==2.3.1

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow==11.3.0

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib==3.10.5
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm==4.67.1

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python==4.11.0.86 opencv-python-headless==4.8.0.74
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple albumentations==1.3.1
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch_lightning==2.5.2
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple utm==0.8.1

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyproj==3.7.1

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn==1.7.1

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio==2.37.0

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image==0.25.2

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple Cartopy==0.24.1
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple certifi==2025.7.9

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple contourpy==1.3.2

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple Cartopy==0.24.1
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple certifi==2025.7.9
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple contourpy==1.3.2
        
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple cycler==0.12.1
        
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple fonttools==4.58.5
        
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple kiwisolver==1.4.8
        
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple packaging==25.0
        
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyparsing==3.2.3
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple shapely==2.1.1
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple six==1.17.0
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyshp==2.3.1

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyinstaller==6.15.0
    
    安装完上述库后，可以运行我的environmental_inspection.py文件来检验所有的依赖与库是否都完美配置




