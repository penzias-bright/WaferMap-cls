# WM-PeleeNet 模型对照

[打开 notebook](notebooks/WaferMap_WM_PeleeNet_Local.ipynb)，选择 `wafer-yolo26` 内核。模型依据 [Yu 等的论文](https://doi.org/10.1049/cit2.12126)重建，并非作者原始代码；38 类版本包含 222,376 个参数，未使用论文中的中值滤波和自编码器增强。

## 运行前

先完成本地 YOLO26n notebook 的 E0 / E1 / E2、冻结和测试。当前实现会校验以下目录中的冻结权重、测试结果和运行记录：

```text
yolo26s_cls/runs/local_n_manual/reports/yolo26n_v1_seed42_manual/
yolo26s_cls/runs/local_n_manual/runs/yolo26n_v1_seed42_manual/
```

这些产物不随仓库发布，需先自行运行生成。若改变 YOLO 的运行标签，还需同步调整 `src/wm_experiment.py` 的 `data_info()` 路径。

## 实验流程

数据核验 → 网络检查 → 训练 → 冻结 → 测试 → YOLO 对照 → 同设备速度测试。训练和评估开关默认关闭；输出位于 `runs/RUN_TAG/`。

复用相同划分、224 输入、seed=42 与 E1 几何增强。WM 模型随机初始化，YOLO 使用预训练权重，优化器调度也有差异，结果应标注为“WM-PeleeNet 重建模型对照”。

中断后使用相同配置与 `RESUME_TRAINING` 恢复上一完整轮；冻结后重训需更换 `RUN_TAG`。环境安装见[共用环境说明](../yolo26s_cls/environment/README.md)。
