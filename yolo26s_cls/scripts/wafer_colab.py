"""Shared Colab/local helpers for the three wafer classification experiments."""
from copy import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import ultralytics
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from ultralytics.utils.metrics import ClassifyMetrics
from ultralytics.utils.torch_utils import EarlyStopping

DATA_SHA = 'bc1ecce02371a39d68533008aae57ef5c7d21954b795d22e917e5cb40851bd62'
EXPERIMENTS = {'E0': 'baseline', 'E1': 'geometry', 'E2': 'balanced'}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    shutil.copy2(source, temp)
    temp.replace(target)


def unpack(archive, destination):
    archive, destination = Path(archive), Path(destination)
    if sha(archive) != DATA_SHA:
        raise ValueError('数据 ZIP SHA-256 不匹配，请使用本项目的 wafer38_v1.zip')
    root = destination / 'wafer38_v1'
    if not root.exists():
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                path = destination / info.filename
                if not path.resolve().is_relative_to(destination.resolve()) or not info.filename.startswith('wafer38_v1/'):
                    raise ValueError('ZIP contains an unexpected path')
            if z.testzip() is not None:
                raise ValueError('ZIP CRC check failed')
            z.extractall(destination)
    return root


def read_metadata(root):
    root = Path(root)
    classes = json.loads((root / 'metadata/classes.json').read_text(encoding='utf-8'))
    with (root / 'metadata/manifest.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if [c['folder'] for c in classes] != [f'class_{i:02d}' for i in range(38)]:
        raise ValueError('类别顺序必须为 class_00 ... class_37')
    return classes, rows


def audit(root):
    root = Path(root)
    classes, rows = read_metadata(root)
    summary = json.loads((root / 'metadata/summary.json').read_text(encoding='utf-8'))
    if len(rows) != summary['unique_samples']:
        raise ValueError('Manifest count mismatch')
    hashes, counts = {}, {}
    for row in rows:
        relative = Path(row['path'])
        target = root / relative
        class_id, split = int(row['class_id']), row['split']
        if split not in ('train', 'val', 'test') or not 0 <= class_id < 38:
            raise ValueError('Invalid split or class')
        if not target.resolve().is_relative_to(root.resolve()) or relative.parts[:2] != (split, classes[class_id]['folder']):
            raise ValueError('Invalid manifest path')
        if sha(target) != row['png_sha256'] or row['matrix_sha256'] in hashes:
            raise ValueError('PNG changed or duplicate matrix found')
        hashes[row['matrix_sha256']] = split
        counts[split] = counts.get(split, 0) + 1
    for split, count in counts.items():
        if count != summary['split_counts'][split]:
            raise ValueError('Split count mismatch')
        if len(list((root / split).glob('*/*.png'))) != count:
            raise ValueError('Unexpected images in split')
        if {int(r['class_id']) for r in rows if r['split'] == split} != set(range(38)):
            raise ValueError('Missing class')
    return summary


def make_smoke(root, output, train_per_class=30, val_per_class=10):
    root, output = Path(root), Path(output)
    if output.exists():
        read_metadata(output)
        return output
    classes, rows = read_metadata(root)
    selected = []
    for split, limit in [('train', train_per_class), ('val', val_per_class)]:
        for c in classes:
            candidates = [r for r in rows if r['split'] == split and int(r['class_id']) == c['class_id']]
            for row in candidates[:limit]:
                target = output / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / row['path'], target)
                selected.append(row)
    (output / 'metadata').mkdir()
    shutil.copy2(root / 'metadata/classes.json', output / 'metadata/classes.json')
    with (output / 'metadata/manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    return output


def transform(geometry=False):
    operations = []
    if geometry:
        operations += [T.RandomChoice([T.RandomRotation((a, a), interpolation=T.InterpolationMode.NEAREST)
                                       for a in (0, 90, 180, 270)]), T.RandomHorizontalFlip(0.5)]
    return T.Compose(operations + [T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
                                   T.ToTensor(), T.Normalize(MEAN, STD)])


class MacroMetrics(ClassifyMetrics):
    def __init__(self):
        super().__init__()
        self.macro_f1 = 0.0

    @property
    def keys(self):
        return super().keys + ['metrics/macro_f1']

    @property
    def fitness(self):
        return self.macro_f1

    @property
    def results_dict(self):
        return dict(zip(self.keys + ['fitness'], [self.top1, self.top5, self.macro_f1, self.fitness]))


class WaferValidator(ClassificationValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = MacroMetrics()

    def build_dataset(self, img_path):
        dataset = super().build_dataset(img_path)
        dataset.torch_transforms = transform()
        return dataset

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        true = torch.cat(self.targets).numpy()
        pred = torch.cat(self.pred)[:, 0].numpy()
        self.metrics.macro_f1 = float(f1_score(true, pred, labels=list(range(38)), average='macro', zero_division=0))
        return self.metrics.results_dict

    def print_results(self):
        print(f'Validation: Top1={self.metrics.top1:.4f} Macro-F1={self.metrics.macro_f1:.4f}')


class StrictEarlyStopping(EarlyStopping):
    def __call__(self, epoch, fitness):
        if fitness == self.best_fitness:
            delta = epoch - self.best_epoch
            self.possible_stop = delta >= self.patience - 1
            return delta >= self.patience
        return super().__call__(epoch, fitness)


class WaferTrainer(ClassificationTrainer):
    def __init__(self, *args, experiment='E0', **kwargs):
        self.experiment = experiment
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.torch_transforms = transform(mode == 'train' and self.experiment == 'E1')
        return dataset

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if self.experiment != 'E2' or mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank not in (-1, 0):
            raise ValueError('This notebook supports one GPU')
        dataset = self.build_dataset(dataset_path, mode)
        labels = np.array([s[1] for s in dataset.samples])
        counts = np.bincount(labels, minlength=38)
        if (counts == 0).any():
            raise ValueError('Training split missing classes')
        sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[labels]), len(labels), replacement=True)
        return InfiniteDataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=self.args.workers, pin_memory=self.device.type == 'cuda',
                                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))

    def get_validator(self):
        validator = WaferValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
        return validator

    def _setup_train(self):
        super()._setup_train()
        stopper = StrictEarlyStopping(patience=self.args.patience)
        stopper.__dict__.update(self.stopper.__dict__)
        self.stopper = stopper

    def final_eval(self):
        # Ultralytics overwrites best.pt on ties; retain the earliest strict Macro-F1 maximum.
        selected = self.wdir / 'best_macro.pt'
        if selected.exists():
            shutil.copy2(selected, self.best)
        return super().final_eval()


def environment():
    return dict(torch=torch.__version__, torchvision=torchvision.__version__, ultralytics=ultralytics.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')


def run_experiment(root, output, experiment, *, backup=None, resume=False, epochs=100, batch=32,
                   workers=2, device=0, stop_after=None, model='yolo26s-cls.pt'):
    """Train one experiment; backup contains a pre-strip resume.pt after every epoch."""
    if experiment not in EXPERIMENTS or ultralytics.__version__ != '8.4.189':
        raise ValueError('Require E0/E1/E2 and ultralytics==8.4.189')
    if model not in ('yolo26n-cls.pt', 'yolo26s-cls.pt'):
        raise ValueError('Model must be yolo26n-cls.pt or yolo26s-cls.pt')
    root, output = Path(root).resolve(), Path(output).resolve()
    classes, _ = read_metadata(root)
    identity = dict(experiment=experiment, model=model, manifest_sha256=sha(root / 'metadata/manifest.csv'),
                    classes_sha256=sha(root / 'metadata/classes.json'), code_sha256=sha(__file__),
                    environment=environment(), epochs=epochs, batch=batch)
    backup = Path(backup).resolve() if backup is not None else None
    if backup == output:
        raise ValueError('Backup directory must differ from working directory')
    if resume:
        if not (output / 'run.json').exists() and backup is not None:
            shutil.copytree(backup, output)
        previous = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        if previous != identity:
            raise ValueError('恢复要求实验、数据、代码、软件环境、轮数及 batch 与原记录一致')
        checkpoint = output / 'weights/resume.pt'
        if not checkpoint.exists():
            raise FileNotFoundError('没有可恢复检查点；请先完成一轮训练')
        checkpoint_state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if checkpoint_state['epoch'] + 1 >= epochs or checkpoint_state.get('optimizer') is None:
            raise ValueError('训练已完成或检查点缺少优化器，不能继续恢复')
        del checkpoint_state
    else:
        if output.exists() or (backup is not None and backup.exists()):
            raise FileExistsError('实验目录已存在；使用 resume=True 或更换 RUN_TAG')
        output.mkdir(parents=True)
        dump(output / 'run.json', identity)
        for name in ('classes.json', 'manifest.csv'):
            shutil.copy2(root / 'metadata' / name, output / name)
        shutil.copy2(__file__, output / 'wafer_colab.py')
        checkpoint = model
    selection_path = output / 'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else dict(macro_f1=-1.0, epoch=0)
    clock = {}

    def epoch_start(trainer):
        clock['start'] = time.perf_counter()
        if trainer.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def saved(trainer):
        metric = float(trainer.metrics['metrics/macro_f1'])
        if not math.isfinite(metric):
            raise ValueError('Validation Macro-F1 is not finite')
        if metric > selection['macro_f1']:
            selection.update(macro_f1=metric, epoch=trainer.epoch + 1)
            atomic_copy(trainer.last, output / 'weights/best_macro.pt')
            dump(selection_path, selection)
        atomic_copy(trainer.last, output / 'weights/resume.pt')
        profile = dict(epoch=trainer.epoch + 1, seconds=time.perf_counter()-clock['start'],
                       batch=trainer.batch_size, macro_f1=metric,
                       peak_gpu_mb=torch.cuda.max_memory_allocated(trainer.device)/2**20 if trainer.device.type=='cuda' else 0)
        if trainer.batch_size != batch:
            raise RuntimeError('框架自动更改了 batch；请统一三组 batch 后重新试跑')
        with (output / 'profile.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(profile) + '\n')
        if backup is not None:
            for p in output.rglob('*'):
                if p.is_file() and p.suffix in ('.json', '.jsonl', '.yaml', '.csv', '.pt', '.py'):
                    atomic_copy(p, backup / p.relative_to(output))
        if stop_after is not None and trainer.epoch + 1 >= stop_after:
            trainer.stop = True

    overrides = dict(model=str(checkpoint), data=str(root), project=str(output.parent), name=output.name,
                     save_dir=str(output), exist_ok=True, epochs=epochs, batch=batch, imgsz=224,
                     optimizer='AdamW', lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3,
                     weight_decay=0.0005, patience=15, amp=True, dropout=0.0, seed=42, deterministic=True,
                     workers=workers, device=device, save=True, plots=False, val=True, cache=False,
                     auto_augment=None, erasing=0.0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0,
                     hsv_v=0.0, close_mosaic=0, resume=str(checkpoint) if resume else False)
    trainer = WaferTrainer(overrides=overrides, experiment=experiment)
    trainer.add_callback('on_train_epoch_start', epoch_start)
    trainer.add_callback('on_model_save', saved)
    trainer.train()
    if stop_after is None or trainer.epoch + 1 >= epochs:
        dump(output / 'completed.json', dict(epochs_completed=trainer.epoch + 1,
                                             best=selection, state='finished'))
    if backup is not None:
        for p in output.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.yaml', '.csv', '.pt', '.py'):
                atomic_copy(p, backup / p.relative_to(output))
    return output


def load_model(weights, root, device=0):
    classes, _ = read_metadata(root)
    model = YOLO(str(weights)).model.to(device).float().eval()
    if [model.names[i] for i in range(38)] != [c['folder'] for c in classes]:
        raise ValueError('Model class mapping differs from dataset')
    return model


@torch.inference_mode()
def probabilities(model, inputs):
    output = model(inputs)
    output = output[0] if isinstance(output, (tuple, list)) else output
    if output.ndim != 2 or output.shape[1] != 38 or not torch.isfinite(output).all():
        raise ValueError('Invalid model output')
    if not torch.allclose(output.sum(1), torch.ones(len(output), device=output.device), atol=1e-3):
        raise ValueError('Expected classification probabilities')
    return output


def evaluate(weights, root, split='val', *, device=0, batch=32, output=None):
    if split not in ('val', 'test'):
        raise ValueError('Evaluation split must be val or test')
    root = Path(root)
    classes, rows = read_metadata(root)
    dataset = torchvision.datasets.ImageFolder(root / split, transform=transform())
    if dataset.classes != [c['folder'] for c in classes]:
        raise ValueError('ImageFolder class mapping differs')
    model = load_model(weights, root, device)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    true, probs = [], []
    for images, labels in loader:
        with torch.inference_mode():
            p = probabilities(model, images.to(device))
        probs.extend(p.cpu().numpy())
        true.extend(labels.tolist())
    p = np.asarray(probs)
    true, pred = np.asarray(true), p.argmax(1)
    names = [c['name'] for c in classes]
    report = classification_report(true, pred, labels=list(range(38)), target_names=names, zero_division=0, output_dict=True)
    pattern_sizes = np.array([sum(c['label']) for c in classes])
    groups = {}
    for name, mask in [('normal', pattern_sizes[true]==0), ('single', pattern_sizes[true]==1), ('mixed', pattern_sizes[true]>=2)]:
        groups[name] = dict(samples=int(mask.sum()), accuracy=float((true[mask]==pred[mask]).mean()))
    metrics = dict(split=split, weights_sha256=sha(weights), manifest_sha256=sha(root/'metadata/manifest.csv'),
                   samples=len(true), accuracy=float(accuracy_score(true, pred)),
                   macro_f1=float(f1_score(true, pred, labels=list(range(38)), average='macro', zero_division=0)),
                   groups=groups, classification_report=report)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        dump(output / f'{split}_metrics.json', metrics)
        np.savez_compressed(output / f'{split}_predictions.npz', true=true, probabilities=p)
        cm = confusion_matrix(true, pred, labels=list(range(38)))
        np.savetxt(output / f'{split}_confusion.csv', cm, fmt='%d', delimiter=',')
        by_path = {r['path']: r for r in rows}
        with (output / f'{split}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['original_index', 'path', 'true_id', 'pred_id', 'confidence', 'correct'])
            for (path, _), t, pred_id, confidence in zip(dataset.samples, true, pred, p.max(1)):
                relative = Path(path).relative_to(root).as_posix()
                writer.writerow([by_path[relative]['original_index'], relative, int(t), int(pred_id), float(confidence), bool(t==pred_id)])
    return metrics


def predict_image(model, image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    pixels = np.asarray(image.convert('RGB'))
    if pixels.shape != (52, 52, 3) or not np.isin(pixels, [0, 128, 255]).all() or not (pixels==pixels[..., :1]).all():
        raise ValueError('演示输入要求本项目 52x52 RGB PNG 编码；不是普通晶圆照片')
    with torch.inference_mode():
        return probabilities(model, transform()(image.convert('RGB')).unsqueeze(0).to(next(model.parameters()).device))[0].cpu().numpy()


def benchmark(weights, root, *, device=0):
    model = load_model(weights, root, device)
    _, rows = read_metadata(root)
    row = next(r for r in rows if r['split']=='val')
    image = Image.open(Path(root)/row['path']).convert('RGB')
    preprocess = transform()
    def iteration():
        with torch.inference_mode():
            probabilities(model, preprocess(image).unsqueeze(0).to(device))
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
    for _ in range(20):
        iteration()
    timings = []
    for _ in range(200):
        start = time.perf_counter()
        iteration()
        timings.append((time.perf_counter()-start)*1000)
    return dict(batch=1, warmup=20, repeats=200, mean_ms=float(np.mean(timings)),
                median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings,95)),
                includes='PIL transform + host-to-device + fp32 model + synchronization; excludes disk read',
                environment=environment())
