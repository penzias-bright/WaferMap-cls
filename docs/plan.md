# 实验流程与协作约定

1. **数据准备**：原始 NPZ 不改动；处理副本将噪声 3→2，按图像与标签去重，固定 seed=42 的约 70/15/15 分层划分。
2. **YOLO 对照**：E0 无增强、E1 几何增强、E2 类别均衡采样。先短试跑，再完成三组正式训练。
3. **验证与测试**：只用验证集选择方案，冻结权重后测试；报告 Macro-F1、准确率、逐类指标和错误案例。
4. **模型比较**：WM-PeleeNet 依赖本地 YOLO26n 的冻结结果、测试报告及权重；先完成本地 YOLO notebook，再运行 WM notebook。

## 协作约定

- 共用 `data/splits/wafer38_v1/` 中的 `manifest.csv` 和 `classes.json`，不各自重新划分。
- 新实验使用独立 `RUN_TAG`；恢复时保持原标签、参数、代码和软件版本。不要并发写入同一实验目录。
- 提交源码与清空输出的 notebook；数据、权重、运行日志和本地报告由 `.gitignore` 排除。需要共享结果时另行传递完整实验目录。
- YOLO notebook 内嵌辅助代码快照。修改 `scripts/wafer_colab.py` 后先运行 `build_notebook.py`，再运行 `build_local_notebook.py`；新快照用于新实验，已有实验恢复继续使用原 notebook。

## 检查

在仓库根目录、激活 `wafer-yolo26` 环境后运行：

```sh
python yolo26s_cls/tests/test_prepare_data.py
python yolo26s_cls/tests/test_wafer_colab.py
python wm_peleenet_cls/tests/test_wm_pelee.py
python tests/check_notebooks.py
```

`validate_notebook.py` 是另行执行的小样本 GPU 流程检查，会生成产物，不属于纯静态检查。

## 数据语义

[作者确认像素 3 为噪声](https://github.com/Junliangwangdhu/WaferMap/issues/4#issuecomment-2438186171)；将其映射为 2 是工程假设。标签顺序为 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_Full、Scratch、Random；类别 ID 按标签向量字典序生成，不等同原论文 C1～C38。

单随机种子结果不代表统计显著性；完全去重不排除近似重复、合成样本或批次关联。
