# YOLO 晶圆分类实验

使用同一份 `wafer38_v1` 数据划分，比较 E0 无增强、E1 几何增强和 E2 类别均衡采样。预训练模型来自 ImageNet，训练后才输出晶圆的 38 类标签。

| 入口 | 适用环境 | 输出位置 |
| --- | --- | --- |
| [YOLO26n](notebooks/WaferMap_YOLO26n_Local.ipynb) | 本地 `wafer-yolo26` 内核 | `runs/local_n_manual/` |
| [YOLO26s](notebooks/WaferMap_YOLO26s_Colab.ipynb) | Colab GPU | Drive 的 `WaferMap/` |

## 使用步骤

1. 按[环境说明](environment/README.md)安装依赖，并按[根目录说明](../README.md)生成数据。
2. 本地打开 n notebook；Colab 上传 s notebook，并将数据 ZIP 放在 `MyDrive/WaferMap/`。Colab notebook 已内嵌共用代码，无需另上传脚本。
3. 顺序完成环境、数据和输入检查；手动开启短试跑，确认训练与恢复正常。
4. 依次完成 E0 / E1 / E2，用验证集 Macro-F1 选模；开启冻结后，再进行测试、错误分析和速度测量。

所有训练和测试开关默认关闭。输入采用完整晶圆缩放，不使用随机裁剪；共同设置为 224 输入、batch=32、seed=42、AdamW、最多 100 轮，验证 Macro-F1 连续 15 轮无提升时早停。

## 保存与恢复

- `best_macro.pt`：最早获得最佳验证 Macro-F1 的权重。
- `resume.pt`：保留优化器的恢复检查点，优先于框架的 `last.pt`。
- 中断恢复保持原 `RUN_TAG`、参数、软件版本与 notebook；更改代码或开始新实验时换 `RUN_TAG`。
- `scripts/wafer_colab.py` 是共用实现。依次运行 `build_notebook.py` 与 `build_local_notebook.py` 可生成新 notebook；不要覆盖正在恢复的实验副本。

本地 n 与 Colab s 的产物相互独立。跨设备速度不直接比较；检查命令见[实验约定](../docs/plan.md)。
