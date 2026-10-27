"""Manual WM-PeleeNet experiment using the project's existing wafer split."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from ultralytics.utils import TQDM

from wm_pelee import WMPeleeNet
import wafer_colab as wafer


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def data_info(project):
    project = Path(project)
    root = project / 'data/processed/wafer38_v1'
    classes, rows = wafer.read_metadata(root)
    manifest_hash = sha(root / 'metadata/manifest.csv')
    baseline = project / 'yolo26s_cls/runs/local_n_manual/reports/yolo26n_v1_seed42_manual'
    frozen = json.loads((baseline / 'frozen.json').read_text(encoding='utf-8'))
    yolo_test = json.loads((baseline / frozen['experiment'] / 'test_metrics.json').read_text(encoding='utf-8'))
    yolo_run = project / 'yolo26s_cls/runs/local_n_manual/runs/yolo26n_v1_seed42_manual' / frozen['experiment'] / 'run.json'
    yolo_identity = json.loads(yolo_run.read_text(encoding='utf-8'))
    if yolo_test['manifest_sha256'] != manifest_hash or yolo_test['weights_sha256'] != frozen['weights_sha256']:
        raise ValueError('YOLO 对照记录的数据划分或冻结权重不匹配。')
    if yolo_identity['classes_sha256'] != sha(root / 'metadata/classes.json'):
        raise ValueError('YOLO 对照记录的类别映射不匹配。')
    if sha(baseline / 'final.pt') != frozen['weights_sha256']:
        raise ValueError('YOLO 冻结权重文件与对照记录不匹配。')
    if set(row['split'] for row in rows) != {'train', 'val', 'test'}:
        raise ValueError('Manifest 缺少 train、val 或 test。')
    return root, classes, rows, baseline, yolo_test


def datasets(root, classes, rows):
    expected = {split: {} for split in ('train', 'val', 'test')}
    for row in rows:
        split = row['split']
        if row['path'] in expected[split]:
            raise ValueError('Manifest 含重复图片路径。')
        expected[split][row['path']] = int(row['class_id'])
    result = {}
    for split in expected:
        dataset = ImageFolder(root / split, transform=wafer.transform(split == 'train'))
        actual = {Path(path).relative_to(root).as_posix(): label for path, label in dataset.samples}
        if dataset.classes != [c['folder'] for c in classes] or actual != expected[split]:
            raise ValueError(f'{split} 图片或类别顺序与 manifest 不一致。')
        result[split] = dataset
    return result


def identity(root, config):
    return {
        'model': 'WM-PeleeNet reconstruction',
        'initialization': 'random',
        'manifest_sha256': sha(root / 'metadata/manifest.csv'),
        'classes_sha256': sha(root / 'metadata/classes.json'),
        'model_code_sha256': sha(Path(__file__).with_name('wm_pelee.py')),
        'experiment_code_sha256': sha(__file__),
        'transform_code_sha256': sha(wafer.__file__),
        'torch': str(torch.__version__),
        'torchvision': str(wafer.torchvision.__version__),
        'ultralytics': str(wafer.ultralytics.__version__),
        'config': config,
    }


@torch.inference_mode()
def evaluate(model, loader, device, class_names):
    model.eval()
    true, pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        if logits.ndim != 2 or logits.shape[1] != len(class_names) or not torch.isfinite(logits).all():
            raise ValueError('模型输出无效。')
        true.extend(labels.tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
    labels = list(range(len(class_names)))
    return {
        'samples': len(true),
        'accuracy': float(accuracy_score(true, pred)),
        'macro_f1': float(f1_score(true, pred, labels=labels, average='macro', zero_division=0)),
        'classification_report': classification_report(true, pred, labels=labels, target_names=class_names,
                                                       output_dict=True, zero_division=0),
    }


def train(project, run, config, device, resume=False):
    root, classes, rows, _, _ = data_info(project)
    data = datasets(root, classes, rows)
    run = Path(run)
    current = identity(root, config)
    if resume:
        if (run / 'completed.json').exists() or (run / 'frozen.pt').exists():
            raise FileExistsError('实验已完成或冻结；如需重训请更换 RUN_TAG。')
        checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['identity'] != current:
            raise ValueError('恢复失败：数据、配置或代码与原实验不一致。')
    else:
        run.mkdir(parents=True, exist_ok=False)
        checkpoint = None
        write_json(run / 'run.json', current)

    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = WMPeleeNet(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    warmup = 3
    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, config['epochs'] - warmup)
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, best_epoch, best_score, start = [], 0, -1.0, 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        history = checkpoint['history']
        best_epoch, best_score = checkpoint['best_epoch'], checkpoint['best_score']
        start = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    train_loader = DataLoader(data['train'], batch_size=config['batch'], shuffle=True, num_workers=0,
                              pin_memory=device.type == 'cuda')
    val_loader = DataLoader(data['val'], batch_size=config['batch'], shuffle=False, num_workers=0,
                            pin_memory=device.type == 'cuda')
    loss_fn = nn.CrossEntropyLoss()
    class_names = [c['name'] for c in classes]
    for epoch in range(start, config['epochs'] + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, count = 0.0, 0
        for images, labels in TQDM(train_loader, desc=f'Epoch {epoch}/{config["epochs"]}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = loss_fn(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError('训练损失不是有限数，已停止。')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        metrics = evaluate(model, val_loader, device, class_names)
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': loss_sum / count, 'val_accuracy': metrics['accuracy'],
               'val_macro_f1': metrics['macro_f1'], 'seconds': time.perf_counter() - started}
        history.append(row)
        if metrics['macro_f1'] > best_score:
            best_epoch, best_score = epoch, metrics['macro_f1']
            save_checkpoint(run / 'best.pt', {'model': model.state_dict(), 'identity': current,
                                              'epoch': epoch, 'validation': metrics})
        save_checkpoint(run / 'last.pt', {'model': model.state_dict(), 'identity': current, 'epoch': epoch,
                                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                         'scaler': scaler.state_dict(), 'history': history,
                                         'best_epoch': best_epoch, 'best_score': best_score,
                                         'torch_rng': torch.get_rng_state(),
                                         'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        print(f"第 {epoch} 轮  loss={row['train_loss']:.4f}  val_acc={row['val_accuracy']:.4f}  "
              f"val_macro_f1={row['val_macro_f1']:.4f}  最佳第 {best_epoch} 轮  {row['seconds']:.1f}s")
        if epoch - best_epoch >= config['patience']:
            print(f"验证 Macro-F1 连续 {config['patience']} 轮未提高，提前停止。")
            break
    write_json(run / 'completed.json', {'epochs_completed': history[-1]['epoch'], 'best_epoch': best_epoch,
                                        'best_macro_f1': best_score, 'identity': current})
    return history


def freeze(project, run, config):
    run = Path(run)
    root, _, _, _, _ = data_info(project)
    current = identity(root, config)
    if (run / 'frozen.pt').exists():
        raise FileExistsError('该实验已经冻结，请勿覆盖。')
    completed = json.loads((run / 'completed.json').read_text(encoding='utf-8'))
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if completed['identity'] != current or best['identity'] != current or best['epoch'] != completed['best_epoch']:
        raise ValueError('训练配置、最佳权重或完成记录不一致。')
    save_checkpoint(run / 'frozen.pt', best)
    write_json(run / 'frozen.json', {'epoch': best['epoch'], 'weights_sha256': sha(run / 'frozen.pt'),
                                    'manifest_sha256': current['manifest_sha256'],
                                    'validation': {'accuracy': best['validation']['accuracy'],
                                                   'macro_f1': best['validation']['macro_f1']}})


def load_frozen(project, run, device):
    run = Path(run)
    root, classes, rows, baseline, yolo_test = data_info(project)
    frozen = json.loads((run / 'frozen.json').read_text(encoding='utf-8'))
    if sha(run / 'frozen.pt') != frozen['weights_sha256'] or sha(root / 'metadata/manifest.csv') != frozen['manifest_sha256']:
        raise ValueError('冻结模型或数据划分发生变化。')
    checkpoint = torch.load(run / 'frozen.pt', map_location='cpu', weights_only=True)
    if checkpoint['identity'] != identity(root, checkpoint['identity']['config']):
        raise ValueError('冻结模型的代码、环境或类别映射与当前不一致。')
    model = WMPeleeNet(len(classes)).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval(), root, classes, rows, baseline, yolo_test, frozen


def test(project, run, device, batch):
    run = Path(run)
    if (run / 'test_metrics.json').exists():
        raise FileExistsError('测试结果已存在，请直接查看，避免重复覆盖。')
    model, root, classes, rows, _, _, frozen = load_frozen(project, run, device)
    data = datasets(root, classes, rows)
    metrics = evaluate(model, DataLoader(data['test'], batch_size=batch, shuffle=False, num_workers=0),
                       device, [c['name'] for c in classes])
    metrics.update(split='test', manifest_sha256=frozen['manifest_sha256'], weights_sha256=frozen['weights_sha256'])
    write_json(run / 'test_metrics.json', metrics)
    return metrics


def compare(project, run):
    root, _, _, baseline, yolo_test = data_info(project)
    run = Path(run)
    frozen = json.loads((run / 'frozen.json').read_text(encoding='utf-8'))
    wm_test = json.loads((run / 'test_metrics.json').read_text(encoding='utf-8'))
    if (wm_test['manifest_sha256'] != yolo_test['manifest_sha256'] or
            wm_test['weights_sha256'] != frozen['weights_sha256'] or
            wm_test['samples'] != yolo_test['samples'] or sha(run / 'frozen.pt') != frozen['weights_sha256']):
        raise ValueError('比较结果的测试集或冻结权重不一致。')
    yolo_frozen = json.loads((baseline / 'frozen.json').read_text(encoding='utf-8'))
    return [
        {'model': 'WM-PeleeNet reconstruction', 'val_accuracy': frozen['validation']['accuracy'],
         'val_macro_f1': frozen['validation']['macro_f1'], 'test_accuracy': wm_test['accuracy'],
         'test_macro_f1': wm_test['macro_f1'], 'test_samples': wm_test['samples']},
        {'model': 'YOLO26n E1', 'val_accuracy': yolo_frozen['validation']['accuracy'],
         'val_macro_f1': yolo_frozen['validation']['macro_f1'], 'test_accuracy': yolo_test['accuracy'],
         'test_macro_f1': yolo_test['macro_f1'], 'test_samples': yolo_test['samples']},
    ]


def benchmark(project, run, device):
    run = Path(run)
    if (run / 'speed.json').exists():
        raise FileExistsError('速度结果已存在，请直接查看，避免重复覆盖。')
    wm, root, _, rows, baseline, _, _ = load_frozen(project, run, device)
    yolo = wafer.load_model(baseline / 'final.pt', root, device)
    image = Image.open(root / next(r['path'] for r in rows if r['split'] == 'val')).convert('RGB')
    transform = wafer.transform()
    result = {'batch': 1, 'warmup': 20, 'repeats': 200, 'device': str(device),
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'includes': 'PIL transform + host-to-device + fp32 model + output validation + synchronization; excludes disk read'}
    for name, model in (('wm_pelee', wm), ('yolo26n', yolo)):
        def iteration():
            with torch.inference_mode():
                output = model(transform(image).unsqueeze(0).to(device))
                output = output[0] if isinstance(output, (list, tuple)) else output.softmax(1)
                if not torch.isfinite(output).all() or not torch.allclose(output.sum(1), torch.ones(1, device=device), atol=1e-3):
                    raise ValueError('速度测试模型输出无效。')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        for _ in range(result['warmup']):
            iteration()
        elapsed = []
        for _ in range(result['repeats']):
            start = time.perf_counter()
            iteration()
            elapsed.append((time.perf_counter() - start) * 1000)
        result[name] = {'mean_ms': float(np.mean(elapsed)), 'median_ms': float(np.median(elapsed)),
                        'p95_ms': float(np.percentile(elapsed, 95))}
    write_json(run / 'speed.json', result)
    return result
