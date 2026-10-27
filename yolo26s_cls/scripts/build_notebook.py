"""Build the self-contained Colab notebook from the shared Python module."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('# YOLO26s-cls · Colab 实验\n\n将晶圆缺陷图识别为 38 类模式，比较 E0 无增强、E1 几何增强与 E2 类别均衡采样。\n\n**准备**：选择 GPU 运行时，将 `wafer38_v1.zip` 上传至 Drive 的 `MyDrive/WaferMap/`。\n\n**流程**：环境与数据（1–6）→ 短试跑（7）→ 训练（8–9）→ 验证与冻结（10）→ 测试与分析（11–14）。训练、冻结和测试开关默认关闭，确认当前步骤后再开启。\n\n数据来自 [WaferMap](https://github.com/Junliangwangdhu/WaferMap)：原始 38,015 张，修正噪声并去重后 36,991 张；训练 / 验证 / 测试为 25,893 / 5,549 / 5,549。`3→2` 为预处理假设，完全去重不排除近似重复或批次关联。')
md('## 1. 选择 GPU 并安装依赖\nColab 菜单：**运行时 → 更改运行时类型 → GPU**。预期为 Tesla T4；其他 GPU 也可试跑，但三组应在同一型号上比较耗时。\n\n本单元使用 Colab 自带 PyTorch，只固定 Ultralytics；若此前导入过不同版本的 Ultralytics，安装后重启会话并从头运行。')
code('''import importlib.util
import os, sys, subprocess
from pathlib import Path

IN_COLAB = importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None
if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.189", "scikit-learn", "pandas", "matplotlib"])
    subprocess.run(["nvidia-smi"], check=True)

import torch, ultralytics, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
assert ultralytics.__version__ == "8.4.189", "请重启会话后从头运行"
assert torch.cuda.is_available(), "未获得 GPU：请更改运行时类型后重试"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__, "| ultralytics:", ultralytics.__version__)''')
md('## 2. 挂载 Drive 与设置路径\n第一次运行按 Colab 提示授权挂载。`RUN_TAG` 是一次完整三组实验的名称；修改参数或重新开始实验时换一个名称。\n\n若 T4 的 batch=32 显存不足，改为 16 并重新进行全部三组试跑。恢复时保留原参数与 RUN_TAG。')
code('''if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = Path("/content/wafer_work")
    DRIVE_ROOT = Path("/content/drive/MyDrive/WaferMap")
else:
    # 供本地验证；实际 Colab 不使用本机目录。
    PROJECT = Path.cwd()
    while not (PROJECT / "data/processed/wafer38_v1.zip").exists() and PROJECT != PROJECT.parent:
        PROJECT = PROJECT.parent
    WORK = PROJECT / "yolo26s_cls/runs/notebook_local"
    DRIVE_ROOT = WORK / "backup"
WORK.mkdir(parents=True, exist_ok=True)
RUN_TAG = "wafer38_v1_seed42"
BATCH = 32
WORKERS = 2 if IN_COLAB else 0
EPOCHS = 100
DEVICE = 0
ARCHIVE = DRIVE_ROOT / "wafer38_v1.zip" if IN_COLAB else PROJECT / "data/processed/wafer38_v1.zip"
os.environ["YOLO_CONFIG_DIR"] = str(WORK / "ultralytics_config")
os.chdir(WORK)
print("数据 ZIP:", ARCHIVE)
print("持久备份:", DRIVE_ROOT / "runs" / RUN_TAG)''')
md('## 3. 加载共用代码\n\n直接运行下方折叠的代码单元。它写入并导入辅助模块，包含训练、评估与恢复逻辑。\n\n所有阶段使用完整晶圆的最近邻缩放（224×224）和 ImageNet 归一化。恢复已有实验时继续使用原 notebook 快照。')
module = (ROOT / 'yolo26s_cls/scripts/wafer_colab.py').read_text(encoding='utf-8')
code('# @title 生成共用实验代码（直接运行）\nimport importlib\nMODULE_PATH = WORK / "wafer_colab.py"\nMODULE_PATH.write_text(' + repr(module) + ', encoding="utf-8")\nif str(WORK) not in sys.path:\n    sys.path.insert(0, str(WORK))\nimport wafer_colab as wc\nimportlib.reload(wc)\nprint("共用代码已就绪:", MODULE_PATH)')
cells[-1].metadata['collapsed'] = True
cells[-1].metadata['cellView'] = 'form'
md('## 4. 复制、解压并完整校验数据\n复制压缩包到 `/content/` 后训练，减少 Drive 小文件读取开销。核验固定 ZIP 哈希、每张 PNG 哈希、类别顺序和集合之间的重复泄漏；此步骤可能需要数分钟。\n\n若提示 ZIP 哈希不符，使用本项目导出的 `wafer38_v1.zip`，不要改动或自行重压缩。')
code('''import shutil, json
local_zip = WORK / "wafer38_v1.zip"
if not local_zip.exists():
    shutil.copy2(ARCHIVE, local_zip)
DATA = wc.unpack(local_zip, WORK / "datasets")
summary = wc.audit(DATA)
classes, manifest = wc.read_metadata(DATA)
display(pd.DataFrame([{"split": s, "images": n} for s, n in summary["split_counts"].items()]))
print("数据校验通过，唯一晶圆数:", summary["unique_samples"])
print("本实验参数:", {"batch": BATCH, "epochs": EPOCHS, "seed": 42})''')
md('## 5. 查看类别分布与样本\n去重后 36 类各有 1,000 张，Random 为 850 张、Near_Full 为 141 张。确认晶圆图四周完整，图中无标题或坐标轴。\n\n类别 ID 采用标签向量字典序，与原论文 C1～C38 编号不等价；汇报使用这里保存的名称和类别映射。')
code('''counts = pd.DataFrame(summary["classes"])
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(counts["folder"], counts[["train", "val", "test"]].sum(axis=1), color="#336699")
ax.set(ylabel="Unique wafers", title="Wafer38: class counts after cleaning and deduplication")
ax.tick_params(axis="x", labelrotation=90)
fig.tight_layout(); plt.show()
display(counts.loc[counts["folder"].isin(["class_01", "class_03"]), ["folder", "name", "train", "val", "test"]])

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, class_id in zip(axes.flat, [0, 1, 2, 3, 4, 6, 10, 37]):
    row = next(r for r in manifest if r["split"] == "train" and int(r["class_id"]) == class_id)
    ax.imshow(Image.open(DATA / row["path"]), interpolation="nearest")
    ax.set_title(f"{classes[class_id]['folder']}\\n{classes[class_id]['name']}", fontsize=10)
    ax.axis("off")
fig.tight_layout(); plt.show()''')
md('## 6. 检查输入变换\n整图四角应保留，最近邻缩放不产生新灰度；E1 只增加 90°倍数旋转和水平翻转，保持模式标签含义。')
code('''row = next(r for r in manifest if r["split"] == "train")
image = Image.open(DATA / row["path"]).convert("RGB")
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, geometry, title in zip(axes, [False, False, True], ["Original 52x52", "Full image 224x224", "E1 geometry example"]):
    if title.startswith("Original"):
        shown = np.asarray(image)
    else:
        tensor = wc.transform(geometry)(image)
        shown = (tensor * torch.tensor(wc.STD)[:, None, None] + torch.tensor(wc.MEAN)[:, None, None]).permute(1, 2, 0).numpy().clip(0, 1)
    ax.imshow(shown, interpolation="nearest"); ax.set_title(title); ax.axis("off")
fig.tight_layout(); plt.show()''')
md('## 7. 短试跑与恢复验证\n\n将 `RUN_SMOKE=True`：每类最多 30 张训练图、10 张验证图，三组各跑 2 轮；E0 在第 1 轮后暂停并恢复，不使用测试集。\n\n确认 loss、Macro-F1、显存及 Drive 备份正常，再进入正式训练。恢复使用含优化器状态的 `resume.pt`；不保证中断前后逐位相同。AMP 检查可能另下载小型检测模型。')
code('''RUN_SMOKE = False
if RUN_SMOKE:
    smoke = wc.make_smoke(DATA, WORK / "smoke_data")
    for experiment in wc.EXPERIMENTS:
        run = WORK / "smoke_runs" / RUN_TAG / experiment
        backup = DRIVE_ROOT / "smoke_runs" / RUN_TAG / experiment
        if not (run / "run.json").exists() and not (backup / "run.json").exists():
            wc.run_experiment(smoke, run, experiment, backup=backup, epochs=2, batch=BATCH,
                              workers=WORKERS, device=DEVICE, stop_after=1 if experiment == "E0" else None)
        checkpoint = run / "weights/resume.pt"
        if not checkpoint.exists() and (backup / "weights/resume.pt").exists():
            shutil.copytree(backup, run, dirs_exist_ok=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        completed = state["epoch"] + 1
        del state
        if completed < 2:
            wc.run_experiment(smoke, run, experiment, backup=backup, resume=True, epochs=2,
                              batch=BATCH, workers=WORKERS, device=DEVICE)
        display(pd.read_json(run / "profile.jsonl", lines=True))
    print("三组短试跑已完成。确认上方结果和 Drive 文件后，再开启正式训练。")
else:
    print("已跳过短试跑；首次使用应先开启。")''')
md('## 8. 正式训练：一次选择一个实验\n\n| 实验 | 改动 | 共同设置 |\n|---|---|---|\n| E0 | 无增强、普通打乱 | 同一预训练权重与划分 |\n| E1 | 90°倍数旋转＋水平翻转 | 其余同 E0 |\n| E2 | 无增强、按类别频数倒数加权采样 | 每轮采样量等于训练集大小 |\n\nAdamW，lr=0.001，weight decay=0.0005，3 轮预热与余弦下降，最多 100 轮；验证 Macro-F1 连续 15 轮无提升时早停。严格提升才更新 `best_macro.pt`，同分保留较早权重。\n\n第一次：设置 `RUN_TRAINING=True`，`EXPERIMENT="E0"`。完成后分别改为 E1、E2 运行本单元。**不要同时启动多个 Notebook 使用同一 RUN_TAG。** 每轮复制权重与日志到 Drive；备份失败会报错。\n\n恢复：重新连接 GPU，从第 1 步开始，跳过已完成短试跑，保持 RUN_TAG、EXPERIMENT、batch 和轮数不变，将 `RESUME=True`。完整训练结束后的检查点不再恢复。')
code('''RUN_TRAINING = False
EXPERIMENT = "E0"  # E0 / E1 / E2
RESUME = False
if RUN_TRAINING:
    run = WORK / "runs" / RUN_TAG / EXPERIMENT
    backup = DRIVE_ROOT / "runs" / RUN_TAG / EXPERIMENT
    wc.run_experiment(DATA, run, EXPERIMENT, backup=backup, resume=RESUME, epochs=EPOCHS,
                      batch=BATCH, workers=WORKERS, device=DEVICE)
    print("已完成:", backup)
else:
    print("正式训练未启动；先完成短试跑，再设置 RUN_TRAINING=True。")''')
md('## 9. 加载正式实验与训练曲线\n重新连接后，这一步也可直接使用 Drive 上的权重和记录。只加载正式实验目录，不把短试跑作为正式成绩。')
code('''RUNS = {}
for experiment in wc.EXPERIMENTS:
    local = WORK / "runs" / RUN_TAG / experiment
    saved = DRIVE_ROOT / "runs" / RUN_TAG / experiment
    directory = local if (local / "weights/best_macro.pt").exists() else saved
    if (directory / "weights/best_macro.pt").exists() and (directory / "completed.json").exists():
        RUNS[experiment] = directory
print("已找到正式权重:", list(RUNS))
if RUNS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for experiment, directory in RUNS.items():
        log = pd.read_csv(directory / "results.csv").rename(columns=lambda x: x.strip())
        axes[0].plot(log["epoch"], log["train/loss"], label=experiment)
        axes[1].plot(log["epoch"], log["metrics/macro_f1"], label=experiment)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].set(title="Validation Macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    for ax in axes: ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); plt.show()
else:
    print("完成正式训练后重跑本单元。")''')
md('## 10. 验证集比较与冻结最终方案\n按验证 Macro-F1、准确率排序；仍同分时依次优先 E0、E1、E2。只有三组全部完成后才冻结，写入最终权重哈希和配置。比较使用重新加载的最佳权重，不依赖 Notebook 中残留的模型。\n\n所有调参到这里结束。单随机种子差异不代表统计显著改善。')
code('''RUN_VALIDATION = False
FREEZE_FINAL = False
REPORTS = DRIVE_ROOT / "reports" / RUN_TAG
if RUN_VALIDATION:
    if set(RUNS) != set(wc.EXPERIMENTS):
        raise RuntimeError("请先完成三组正式实验并运行第 9 步")
    comparison = []
    for experiment, directory in RUNS.items():
        result = wc.evaluate(directory / "weights/best_macro.pt", DATA, "val", device=DEVICE, batch=BATCH,
                             output=REPORTS / experiment)
        comparison.append({"experiment": experiment, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
    comparison = pd.DataFrame(comparison).sort_values(["macro_f1", "accuracy", "experiment"], ascending=[False, False, True])
    REPORTS.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS / "validation_comparison.csv", index=False)
    display(comparison)
    if FREEZE_FINAL:
        frozen_path = REPORTS / "frozen.json"
        if frozen_path.exists():
            raise FileExistsError("方案已冻结；保留记录，不能用测试结果重新选模型")
        winner = comparison.iloc[0]["experiment"]
        weights = RUNS[winner] / "weights/best_macro.pt"
        wc.atomic_copy(weights, REPORTS / "final.pt")
        wc.dump(frozen_path, {"experiment": winner, "weights_sha256": wc.sha(REPORTS / "final.pt"),
                              "validation": comparison.iloc[0].to_dict(), "data_sha256": wc.DATA_SHA})
        print("已冻结:", winner, REPORTS / "final.pt")
else:
    print("正式训练完成后设置 RUN_VALIDATION=True；确认比较结果后再设置 FREEZE_FINAL=True。")''')
md('## 11. 最终测试（方案冻结后）\n显式开启后评估三组最佳模型，输出准确率、Macro-F1、逐类指标、分组准确率、预测清单及混淆矩阵。测试结果用于报告，不再用于挑选或调整方案。已生成结果时读取缓存，避免重复计算。\n\n本 Notebook 按验证集选择最终模型；即使测试集另一组分数更高，也保留冻结方案。')
code('''RUN_TEST = False
if RUN_TEST:
    frozen = json.loads((REPORTS / "frozen.json").read_text(encoding="utf-8"))
    assert wc.sha(REPORTS / "final.pt") == frozen["weights_sha256"]
    assert set(RUNS) == set(wc.EXPERIMENTS)
    tests = []
    for experiment, directory in RUNS.items():
        metric_path = REPORTS / experiment / "test_metrics.json"
        weights = directory / "weights/best_macro.pt"
        if metric_path.exists():
            result = json.loads(metric_path.read_text(encoding="utf-8"))
            assert result["weights_sha256"] == wc.sha(weights), "已有结果对应不同权重"
        else:
            result = wc.evaluate(weights, DATA, "test", device=DEVICE, batch=BATCH, output=REPORTS / experiment)
        tests.append({"experiment": experiment, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                      "frozen_final": experiment == frozen["experiment"]})
    display(pd.DataFrame(tests))
else:
    print("测试集尚未评估。")''')
md('## 12. 混淆矩阵、逐类指标与错误案例\n矩阵行是真实类别、列是预测类别；归一化图按行除以该类样本数。类别 ID 的完整含义见下方表格。优先检查少数类 Near_Full 的召回率及高置信度误判。')
code('''if (REPORTS / "frozen.json").exists():
    frozen = json.loads((REPORTS / "frozen.json").read_text(encoding="utf-8"))
    experiment = frozen["experiment"]
    result_dir = REPORTS / experiment
    if (result_dir / "test_metrics.json").exists():
        metrics = json.loads((result_dir / "test_metrics.json").read_text(encoding="utf-8"))
        display(pd.DataFrame(metrics["classification_report"]).T)
        display(pd.DataFrame(metrics["groups"]).T)
        cm = np.loadtxt(result_dir / "test_confusion.csv", delimiter=",")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, values, title in zip(axes, [cm, cm / cm.sum(1, keepdims=True)], ["Confusion counts", "Recall by true class"]):
            shown = ax.imshow(values, cmap="Blues", vmin=0)
            ax.set(title=title, xlabel="Predicted class ID", ylabel="True class ID")
            ax.set_xticks(range(0, 38, 2)); ax.set_yticks(range(0, 38, 2))
            fig.colorbar(shown, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(result_dir / "confusion.png", dpi=180); plt.show()
        errors = pd.read_csv(result_dir / "test_predictions.csv")
        errors = errors.loc[~errors["correct"]].sort_values("confidence", ascending=False)
        errors.to_csv(result_dir / "errors.csv", index=False)
        display(errors.head(10))
        if len(errors):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for ax, (_, row) in zip(axes.flat, errors.head(6).iterrows()):
                ax.imshow(Image.open(DATA / row["path"]), interpolation="nearest")
                ax.set_title(f"True: {int(row['true_id']):02d} / Pred: {int(row['pred_id']):02d}\\nscore={row['confidence']:.3f}")
            for ax in axes.flat: ax.axis("off")
            fig.tight_layout(); fig.savefig(result_dir / "error_examples.png", dpi=180); plt.show()
        else:
            print("本次测试没有错误样本，如实报告，不构造错误案例。")
        display(pd.DataFrame(classes)[["class_id", "name"]])
    else:
        print("请先完成最终测试。")
else:
    print("请先完成验证比较并冻结方案。")''')
md('## 13. 推理速度与单图演示\n速度在同一设备上测量：batch=1，预热 20 次、计时 200 次，包含 CPU 图片变换、传输、FP32 前向及 CUDA 同步，不含磁盘读取。分数是模型输出，不是校准后的可靠性概率。\n\n演示选择数据集中的原始索引，或读取同一 52×52、RGB 三通道一致、0/128/255 编码的 PNG；不能直接输入普通晶圆照片。')
code('''RUN_BENCHMARK = False
DEMO_ORIGINAL_INDEX = None  # None 自动选测试集第一张；可填原始索引整数
if (REPORTS / "final.pt").exists():
    if RUN_BENCHMARK:
        speed_rows = []
        for experiment, directory in RUNS.items():
            speed = wc.benchmark(directory / "weights/best_macro.pt", DATA, device=DEVICE)
            wc.dump(REPORTS / experiment / "speed.json", speed)
            speed_rows.append({"experiment": experiment, "mean_ms": speed["mean_ms"], "p95_ms": speed["p95_ms"]})
        display(pd.DataFrame(speed_rows))
    model = wc.load_model(REPORTS / "final.pt", DATA, DEVICE)
    row = next(r for r in manifest if r["split"] == "test" and
               (DEMO_ORIGINAL_INDEX is None or int(r["original_index"]) == DEMO_ORIGINAL_INDEX))
    image = Image.open(DATA / row["path"]).convert("RGB")
    scores = wc.predict_image(model, image)
    top3 = scores.argsort()[::-1][:3]
    plt.figure(figsize=(4, 4)); plt.imshow(image, interpolation="nearest"); plt.axis("off")
    plt.title(f"True: {classes[int(row['class_id'])]['name']}"); plt.show()
    display(pd.DataFrame([{"rank": rank+1, "class": classes[i]["name"], "score": float(scores[i])} for rank, i in enumerate(top3)]))
else:
    print("冻结模型后可使用本单元演示。")''')
md('## 14. 保存与复现\n\n- Drive 的 `WaferMap/` 下的 `runs/RUN_TAG/` 保存配置、日志和权重，`reports/RUN_TAG/` 保存评估结果。\n- `best_macro.pt` 用于评估；`resume.pt` 包含优化器状态，用于恢复。\n- 保存带输出的实验副本；提交到 Git 的 notebook 保持输出清空。\n- 新实验更换 `RUN_TAG`；已有目录选择恢复，不直接删除。备份失败时先检查 Drive 挂载与空间。\n\n报告注明划分、随机种子、软件版本和设备；不同设备耗时不能直接归因于模型大小。')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                     'colab': {'name': 'WaferMap_YOLO26s_Colab.ipynb', 'provenance': []}}
nbf.validate(notebook)
path = ROOT / 'yolo26s_cls/notebooks/WaferMap_YOLO26s_Colab.ipynb'
for cell in notebook.cells:
    if cell.cell_type == 'code' and 'MODULE_PATH.write_text' in cell.source:
        cell.metadata['jupyter'] = {'source_hidden': True}
nbf.write(notebook, path)
print(path, len(cells), 'cells')
