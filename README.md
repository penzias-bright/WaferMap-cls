# WaferMap-cls

**晶圆缺陷图的 38 类模式识别实验：比较 YOLO26n、YOLO26s 与 WM-PeleeNet 重建模型。**

面向课程研究与协作复现，提供数据预处理、固定划分、训练、评估和可视化 notebook。输入为 52×52 晶圆缺陷图，输出为 8 种基本缺陷组合形成的 38 个互斥类别（含 Normal），不是普通晶圆照片或目标检测任务。

## 从这里开始

| 入口 | 运行位置 | 用途 |
| --- | --- | --- |
| [YOLO26n](yolo26s_cls/notebooks/WaferMap_YOLO26n_Local.ipynb) | 本地 GPU | 推荐先跑：建立 E0 / E1 / E2 对照 |
| [YOLO26s](yolo26s_cls/notebooks/WaferMap_YOLO26s_Colab.ipynb) | Colab GPU | 同一划分上的较大模型实验 |
| [WM-PeleeNet](wm_peleenet_cls/notebooks/WaferMap_WM_PeleeNet_Local.ipynb) | 本地 GPU | 完成 YOLO26n 冻结与测试后进行模型对照 |

**运行顺序：准备环境 → 导出数据 → 短试跑 → 训练 → 验证选模 → 冻结 → 测试。** 训练、冻结与测试开关需手动开启。

## 快速开始

1. 克隆仓库，按[环境说明](yolo26s_cls/environment/README.md)创建 `wafer-yolo26` 环境，并注册 Jupyter 内核。
2. 从原始 [WaferMap 数据仓库](https://github.com/Junliangwangdhu/WaferMap)获取 MixedWM38 数据，将 NPZ 放在本仓库根目录并命名为 `Wafer_Map_Datasets.npz`。
3. 在仓库根目录、已激活的环境中执行：

   ```sh
   python yolo26s_cls/scripts/prepare_data.py
   python -m ipykernel install --user --name wafer-yolo26 --display-name "Python (wafer-yolo26)"
   ```

4. 打开上表 notebook，选择对应内核并按章节运行。Colab 另需将生成的 `data/processed/wafer38_v1.zip` 上传至 Drive 的 `MyDrive/WaferMap/`。

已有数据版本不会被覆盖。原始数据、预训练权重和实验产物不随 Git 仓库分发；首次加载 YOLO 预训练权重需要网络。

## 实验约定

| 项目 | 设置 |
| --- | --- |
| 数据 | 原始 38,015 张；修正噪声并去重后 36,991 张 |
| 划分 | seed=42；训练 / 验证 / 测试：25,893 / 5,549 / 5,549 |
| YOLO 对照 | E0 无增强；E1 旋转与翻转；E2 类别均衡采样 |
| 选模 | 验证集 Macro-F1；冻结后评估测试集 |
| 输入 | 保留完整晶圆，最近邻缩放至 224×224 |

噪声像素 `3→2` 是本项目的预处理假设。WM-PeleeNet 为论文重建实现，从随机权重训练，与使用预训练权重的 YOLO 不构成完全同条件比较。完全去重也不代表已排除近似重复或批次关联，当前结果不能直接推断产线泛化能力。

## 项目导航

- [YOLO 实验说明](yolo26s_cls/README.md)：本地与 Colab 使用、断点恢复。
- [WM-PeleeNet 说明](wm_peleenet_cls/README.md)：前置结果、模型差异。
- [实验流程与协作约定](docs/plan.md)：共享划分、产物和验证方法。
- `*/scripts/`、`*/src/`：实现；`*/tests/`：检查；`*/environment/`：环境快照。

数据来源与使用条件以原仓库为准；WM-PeleeNet 参考 [Yu 等的论文](https://doi.org/10.1049/cit2.12126)。本仓库未附带原始数据或论文全文。
