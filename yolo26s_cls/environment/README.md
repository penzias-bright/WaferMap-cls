# 安装环境

两套本地 notebook 共用 Python 3.11 的 `wafer-yolo26` 环境。从仓库根目录执行：

```sh
conda create -n wafer-yolo26 python=3.11 pip -y
conda activate wafer-yolo26
python -m pip install -r yolo26s_cls/environment/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install nbformat nbclient jupyterlab
python -m pip check
python -m ipykernel install --user --name wafer-yolo26 --display-name "Python (wafer-yolo26)"
python -m jupyter lab
```

`requirements-lock.txt` 为已有环境快照，包含 CUDA 12.8 版 PyTorch；运行前确认机器的 NVIDIA 驱动支持该环境。本地 YOLO notebook 要求 CUDA，WM notebook 也可使用 CPU。

`environment.yml` 和 `conda-explicit.txt` 是 Windows 环境快照，不是跨平台安装清单。其他平台以 Python 3.11 和 pip 依赖为起点验证；Colab 使用内置 PyTorch，由 notebook 安装指定的 Ultralytics。

`scripts/python.ps1` 是原开发机的隔离启动器，保留用于本地复查；其他电脑使用激活环境后的 `python` 命令即可。
