"""Create a local YOLO26n walkthrough with every training switch disabled."""
from pathlib import Path
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
notebook = nbf.read(ROOT / 'yolo26s_cls/notebooks/WaferMap_YOLO26s_Colab.ipynb', as_version=4)
for cell in notebook.cells:
    cell.source = cell.source.replace('YOLO26s-cls', 'YOLO26n-cls')
    if cell.cell_type == 'code':
        cell.execution_count = None
        cell.outputs = []

notebook.cells[0].source = '# YOLO26n-cls · 本地实验\n\n将晶圆缺陷图识别为 38 类模式，比较 E0 无增强、E1 几何增强与 E2 类别均衡采样。\n\n**准备**：选择 `wafer-yolo26` 内核，从仓库目录启动 Jupyter，并先生成 `data/processed/wafer38_v1/`。\n\n**流程**：环境与数据（1–6）→ 短试跑（7）→ 训练（8–9）→ 验证与冻结（10）→ 测试与分析（11–14）。训练、冻结和测试开关默认关闭，确认当前步骤后再开启。\n\n数据来自 [WaferMap](https://github.com/Junliangwangdhu/WaferMap)：原始 38,015 张，修正噪声并去重后 36,991 张；训练 / 验证 / 测试为 25,893 / 5,549 / 5,549。`3→2` 为预处理假设，完全去重不排除近似重复或批次关联。'

for i, cell in enumerate(notebook.cells):
    if cell.cell_type == 'markdown':
        if cell.source.startswith('## 1.'):
            cell.source = '## 1. 检查本地环境\n先选择 wafer-yolo26 内核。本单元检查 Python、库版本和 GPU，不安装依赖、不启动训练。'
        elif cell.source.startswith('## 2.'):
            cell.source = '## 2. 设置项目路径与实验参数\n默认指向当前项目。模型固定为 YOLO26n-cls；与 s 保持相同输入尺寸、batch、优化器、种子及数据划分。\n\n修改训练参数时使用新的 RUN_TAG。恢复中断实验时保留原 RUN_TAG、模型、代码、batch 和轮数。Windows 默认 workers=0。'
        elif cell.source.startswith('## 3.'):
            cell.source = '## 3. 加载共用代码\n\n直接运行下方折叠的代码单元。它写入并导入辅助模块，包含训练、评估与恢复逻辑。\n\n所有阶段使用完整晶圆的最近邻缩放（224×224）和 ImageNet 归一化。恢复已有实验时继续使用原 notebook 快照。'
        elif cell.source.startswith('## 4.'):
            cell.source = '## 4. 读取数据与可选完整审计\n直接使用已导出的本地 PNG，无需复制 ZIP 或挂载 Drive。默认读取类别和划分统计。\n\n如需重新核验所有 PNG 哈希、类别和集合互斥，将 RUN_DATA_AUDIT 改为 True；此项只检查数据，不训练。'
        elif cell.source.startswith('## 7.'):
            cell.source = '## 7. 短试跑\n将 RUN_SMOKE 改为 True 后，只运行 SMOKE_EXPERIMENT 指定的一组，两轮，每类最多 30 张训练、10 张验证，不使用测试集。E0/E1/E2 可依次手动选择。\n\n首次设 SMOKE_RESUME=False；若要验证恢复，先设置 SMOKE_STOP_AFTER=1，在保存第一轮后停止，再将 SMOKE_RESUME=True、SMOKE_STOP_AFTER=None，手动重跑。\n\n已有输出且不恢复时会拒绝覆盖；需要重做时更换 RUN_TAG。'
        elif cell.source.startswith('## 8.'):
            cell.source = '## 8. 正式训练\n\n| 实验 | 设置 |\n|---|---|\n| E0 | 无增强、普通打乱 |\n| E1 | 90°倍数旋转＋水平翻转 |\n| E2 | 无增强、按训练类别频数倒数加权采样 |\n\n先选择 EXPERIMENT="E0"，准备好后将 RUN_TRAINING=True 并运行本单元。其他两组由你之后手动选择启动，不会自动连跑。\n\n共同设置：YOLO26n-cls 预训练权重、224 输入、batch=32、seed=42、AdamW lr=0.001、weight decay=0.0005、3 轮预热与余弦下降、最多 100 轮；验证 Macro-F1 连续 15 轮无提升时早停。\n\n中断后重新运行前面配置和代码单元，设置 RESUME=True。检查点每轮保存在本地，恢复使用 resume.pt；不启动隐藏或后台进程。用户中断内核运算后，先确认原单元已停止再恢复。'
        elif cell.source.startswith('## 9.'):
            cell.source = '## 9. 查看已完成的正式实验与学习曲线\n读取本 Notebook 工作区内已完成的 n 实验及日志，绘制损失和验证 Macro-F1 曲线，不重新训练。仅做过短试跑时列表为空是正常的。'
        elif cell.source.startswith('## 14.'):
            cell.source = '## 14. 保存与复现\n\n- `yolo26s_cls/runs/local_n_manual/` 下的 `runs/RUN_TAG/` 保存配置、日志和权重，`reports/RUN_TAG/` 保存评估结果。\n- `best_macro.pt` 用于评估；`resume.pt` 包含优化器状态，用于恢复。\n- 保存带输出的实验副本；提交到 Git 的 notebook 保持输出清空。\n- 新实验更换 `RUN_TAG`；已有目录选择恢复，不直接删除。备份失败时先检查 Drive 挂载与空间。\n\n报告注明划分、随机种子、软件版本和设备；不同设备耗时不能直接归因于模型大小。'
        continue

    source = cell.source
    if source.startswith('import importlib.util'):
        cell.source = '''import os, sys, json
from pathlib import Path
assert sys.version_info >= (3, 11), "请选择 wafer-yolo26 内核"
assert Path(sys.prefix).name == "wafer-yolo26", f"当前内核为 {sys.executable}，请切换"
import torch, ultralytics, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
assert ultralytics.__version__ == "8.4.189"
assert torch.cuda.is_available(), "当前内核未检测到 CUDA"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
print("解释器:", sys.executable)
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)'''
    elif source.startswith('if IN_COLAB:'):
        cell.source = '''PROJECT = Path.cwd().resolve()
while not (PROJECT / "yolo26s_cls/scripts/prepare_data.py").is_file() and PROJECT != PROJECT.parent:
    PROJECT = PROJECT.parent
assert (PROJECT / "yolo26s_cls/scripts/prepare_data.py").is_file(), "请从仓库目录或其子目录启动 Jupyter"
assert (PROJECT / "data/processed/wafer38_v1/metadata/manifest.csv").exists()
WORK = PROJECT / "yolo26s_cls/runs/local_n_manual"
WORK.mkdir(parents=True, exist_ok=True)
LOCAL_ROOT = WORK
RUN_TAG = "yolo26n_v1_seed42_manual"
MODEL = "yolo26n-cls.pt"
BATCH = 32
WORKERS = 0
EPOCHS = 100
DEVICE = 0
os.environ["YOLO_CONFIG_DIR"] = str(WORK / "ultralytics_config")
os.chdir(WORK)
print("工作区:", WORK)
print("模型:", MODEL, "| 所有训练开关默认关闭")'''
    elif 'MODULE_PATH.write_text(' in source:
        module = (ROOT / 'yolo26s_cls/scripts/wafer_colab.py').read_text(encoding='utf-8')
        cell.source = '# @title 生成共用代码（不训练）\nimport importlib\nMODULE_PATH = WORK / "wafer_colab.py"\nMODULE_PATH.write_text(' + repr(module) + ', encoding="utf-8")\nif str(WORK) not in sys.path:\n    sys.path.insert(0, str(WORK))\nimport wafer_colab as wc\nimportlib.reload(wc)\nprint("共用代码已就绪")'
    elif source.startswith('import shutil, json'):
        cell.source = '''DATA = PROJECT / "data/processed/wafer38_v1"
RUN_DATA_AUDIT = False
if RUN_DATA_AUDIT:
    summary = wc.audit(DATA)
else:
    summary = json.loads((DATA / "metadata/summary.json").read_text(encoding="utf-8"))
classes, manifest = wc.read_metadata(DATA)
display(pd.DataFrame([{"split": s, "images": n} for s, n in summary["split_counts"].items()]))
print("数据读取完成，完整审计:", RUN_DATA_AUDIT)'''
    elif source.startswith('RUN_SMOKE ='):
        cell.source = '''RUN_SMOKE = False
SMOKE_EXPERIMENT = "E0"
SMOKE_RESUME = False
SMOKE_STOP_AFTER = None  # 改为 1 可检查第一轮停止后的恢复
if RUN_SMOKE:
    smoke = wc.make_smoke(DATA, WORK / "smoke_data")
    run = WORK / "smoke_runs" / RUN_TAG / SMOKE_EXPERIMENT
    wc.run_experiment(smoke, run, SMOKE_EXPERIMENT, model=MODEL, epochs=2, batch=BATCH,
                      workers=WORKERS, device=DEVICE, resume=SMOKE_RESUME, stop_after=SMOKE_STOP_AFTER)
    display(pd.read_json(run / "profile.jsonl", lines=True))
else:
    print("短试跑未启动。由你设置 RUN_SMOKE=True 后手动运行。")'''
    elif source.startswith('RUN_TRAINING ='):
        cell.source = '''RUN_TRAINING = False
EXPERIMENT = "E0"
RESUME = False
if RUN_TRAINING:
    run = WORK / "runs" / RUN_TAG / EXPERIMENT
    wc.run_experiment(DATA, run, EXPERIMENT, model=MODEL, resume=RESUME, epochs=EPOCHS,
                      batch=BATCH, workers=WORKERS, device=DEVICE)
    print("本组已完成:", run)
else:
    print("正式训练未启动。由你设置 RUN_TRAINING=True 后手动运行。")'''
    else:
        cell.source = source.replace('DRIVE_ROOT', 'LOCAL_ROOT')
        if source.startswith('RUN_BENCHMARK ='):
            cell.source = cell.source.replace('RUN_BENCHMARK = False', 'RUN_BENCHMARK = False\nRUN_DEMO = False')
            cell.source = cell.source.replace('if (REPORTS / "final.pt").exists():', 'if (RUN_DEMO or RUN_BENCHMARK) and (REPORTS / "final.pt").exists():')
            cell.source = cell.source.replace('冻结模型后可使用本单元演示。', '冻结后手动开启 RUN_DEMO 或 RUN_BENCHMARK。')
    ast.parse(cell.source)

notebook.metadata = {'kernelspec': {'display_name': 'Python (wafer-yolo26)', 'language': 'python', 'name': 'wafer-yolo26'},
                     'language_info': {'name': 'python'}}
nbf.validate(notebook)
output = ROOT / 'yolo26s_cls/notebooks/WaferMap_YOLO26n_Local.ipynb'
for cell in notebook.cells:
    if cell.cell_type == 'code' and 'MODULE_PATH.write_text' in cell.source:
        cell.metadata['jupyter'] = {'source_hidden': True}
nbf.write(notebook, output)
print(output)
