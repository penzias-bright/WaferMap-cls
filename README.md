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

## 测试结果与性能对比

截至 2026-10-27，以下为同一 `wafer38_v1` 划分、**5,549 张测试样本**上的正式结果。两种 YOLO 均按验证集选中 E1（旋转与翻转），WM-PeleeNet 使用 E1 增强；测试集不参与选模。

| 模型 / 方案 | 验证 Macro-F1 | 测试准确率 | 测试 Macro-F1 |
| --- | ---: | ---: | ---: |
| YOLO26n-cls · E1 | 98.683% | 98.775% | 98.694% |
| YOLO26s-cls · E1 | 98.874% | 98.684% | 98.663% |
| WM-PeleeNet 重建版 · E1 | 98.677% | 98.396% | 98.276% |

本次单种子实验中，YOLO26n 的测试准确率与 Macro-F1 略高；YOLO26s 的验证 Macro-F1 最高。两种 YOLO 的测试差距很小，不据此推断统计显著优势。

### 推理延迟

| 模型 | 测速 GPU | 平均延迟（ms） | 中位延迟（ms） | P95 延迟（ms） |
| --- | --- | ---: | ---: | ---: |
| YOLO26n-cls | RTX 4070 Laptop | 10.387 | 9.192 | 16.824 |
| YOLO26s-cls | 未记录 | — | — | — |
| WM-PeleeNet 重建版 | RTX 4070 Laptop | 6.629 | 5.678 | 11.426 |

本地两模型来自同一次对照测速：batch=1、FP32、预热 20 次、计时 200 次；包含图像预处理、CPU→GPU 传输、模型前向、输出校验与 CUDA 同步，不含磁盘读取。WM-PeleeNet 在该次测量中延迟更低。

YOLO26s 的正式实验运行于 Colab Tesla T4，当前 Drive 报告中未找到推理测速记录，因此不填写延迟，也不进行跨 GPU 速度排名。WM-PeleeNet 从随机权重训练，YOLO 使用预训练权重，训练条件并不完全相同。

指标取自本地与 Drive 的冻结记录、`test_metrics.json` 和双模型 `speed.json`；[结果快照](docs/benchmark-results.json)保留完整精度、原报告位置、划分与权重哈希，便于核对。快照不包含权重或样本数据。

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
