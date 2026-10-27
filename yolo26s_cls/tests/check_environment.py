"""只读检查数据，验证 GPU 和预训练分类模型；不训练。"""
import json
import site
import sys
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

root = Path(__file__).resolve().parents[2]
assert sys.version_info[:2] == (3, 11), sys.version
assert Path(sys.prefix).name == "wafer-yolo26", sys.prefix
assert not site.ENABLE_USER_SITE
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.version.cuda == "12.8", torch.version.cuda
x = torch.ones((16, 16), device="cuda")
assert torch.all(x @ x == 16).item()
with np.load(root / "MixedWM38.npz", allow_pickle=False) as data:
    assert {"arr_0", "arr_1"}.issubset(data.files)
    images, labels = data["arr_0"], data["arr_1"]
    assert images.ndim == 3 and images.shape[1:] == (52, 52), images.shape
    assert labels.shape == (len(images), 8), labels.shape
    unexpected = ~np.isin(images, [0, 1, 2])
    values, value_counts = np.unique(images, return_counts=True)
    assert np.isin(labels, [0, 1]).all()
    patterns, counts = np.unique(labels, axis=0, return_counts=True)
    assert len(patterns) == 38, len(patterns)
    dataset = dict(keys=data.files, images_shape=list(images.shape), labels_shape=list(labels.shape),
                   pixel_values={str(v): int(c) for v, c in zip(values, value_counts)},
                   unexpected_pixel_count=int(unexpected.sum()),
                   affected_sample_indices=np.flatnonzero(unexpected.any(axis=(1, 2))).tolist(),
                   status="needs_review" if unexpected.any() else "passed",
                   images_dtype=str(images.dtype), labels_dtype=str(labels.dtype),
                   patterns=[dict(label=p.tolist(), count=int(c)) for p, c in zip(patterns, counts)])
model = YOLO(str(root / "yolo26s_cls/weights/yolo26s-cls.pt"))
assert model.task == "classify", model.task
result = model.predict(np.zeros((224, 224, 3), dtype=np.uint8), device=0, verbose=False)[0]
probs = result.probs.data
assert probs.is_cuda and torch.isfinite(probs).all().item()
assert abs(probs.sum().item() - 1) < 1e-4
report = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
              torchvision=__import__("torchvision").__version__, ultralytics=ultralytics.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
              prediction_classes=probs.numel(), dataset=dataset, status="passed")
output = root / "yolo26s_cls/reports/environment_check.json"
output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
