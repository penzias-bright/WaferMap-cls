"""Build a versioned, auditable wafer classification dataset without changing the source."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random']
LUT = np.array([0, 128, 255], dtype=np.uint8)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_and_group(x, y):
    if x.ndim != 3 or x.shape[1:] != (52, 52) or y.shape != (len(x), 8):
        raise ValueError('Unexpected array shapes')
    if not np.isin(x, [0, 1, 2, 3]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Unexpected pixel or label values')
    corrected = x.copy()
    positions = np.argwhere(corrected == 3)
    corrected[corrected == 3] = 2
    corrected = corrected.astype(np.uint8)
    groups = {}
    for i, wafer in enumerate(corrected):
        digest = hashlib.sha256(wafer.tobytes()).hexdigest()
        groups.setdefault(digest, []).append(i)
    conflicts = [dict(matrix_sha256=h, indices=ids) for h, ids in groups.items()
                 if not (y[ids] == y[ids[0]]).all()]
    return corrected, positions, groups, conflicts


def split_indices(ids, labels):
    train, rest = train_test_split(ids, test_size=0.30, stratify=labels, random_state=42)
    label_by_id = dict(zip(ids, labels))
    val, test = train_test_split(rest, test_size=0.5,
                                stratify=[label_by_id[i] for i in rest], random_state=42)
    return {'train': sorted(train), 'val': sorted(val), 'test': sorted(test)}


def rgb(wafer):
    return np.repeat(LUT[wafer][..., None], 3, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'Wafer_Map_Datasets.npz')
    parser.add_argument('--version', default='wafer38_v1')
    args = parser.parse_args()
    if not args.version.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Version must contain only letters, digits, underscores or hyphens')
    out = ROOT / 'data/processed' / args.version
    shared = ROOT / 'data/splits' / args.version
    archive = out.with_suffix('.zip')
    if any(p.exists() for p in (out, shared, archive)):
        raise FileExistsError('Output version already exists; choose a new --version, no overwrite allowed')
    source_hash = sha(args.source)
    with np.load(args.source, allow_pickle=False) as data:
        if set(data.files) != {'arr_0', 'arr_1'}:
            raise ValueError('Expected arr_0 and arr_1')
        original, labels = data['arr_0'], data['arr_1']
    if len(original) != 38015:
        raise ValueError('Expected 38015 source samples')
    x, positions, groups, conflicts = clean_and_group(original, labels)
    patterns = np.unique(labels, axis=0)
    if len(patterns) != 38:
        raise ValueError('Expected 38 label patterns')
    if conflicts:
        conflict_path = ROOT / 'yolo26s_cls/reports' / f'{args.version}_conflicts.json'
        if conflict_path.exists():
            raise FileExistsError(conflict_path)
        write_json(conflict_path, conflicts)
        raise ValueError(f'Conflicting labels found: {conflict_path}')
    pattern_ids = {tuple(p): i for i, p in enumerate(patterns)}
    keep = sorted(ids[0] for ids in groups.values())
    class_by_id = {i: pattern_ids[tuple(labels[i])] for i in keep}
    splits = split_indices(keep, [class_by_id[i] for i in keep])
    assert len(set(sum(splits.values(), []))) == len(keep)
    assert sum(map(len, splits.values())) == len(keep)
    for ids in splits.values():
        assert {class_by_id[i] for i in ids} == set(range(38))
    out.mkdir(parents=True)
    shared.mkdir(parents=True)
    metadata = out / 'metadata'
    metadata.mkdir()
    classes = [dict(class_id=i, folder=f'class_{i:02d}', label=p.tolist(),
                    name='+'.join(n for n, v in zip(NAMES, p) if v) or 'Normal')
               for i, p in enumerate(patterns)]
    write_json(metadata / 'classes.json', classes)
    write_csv(metadata / 'noise_corrections.csv',
              (dict(original_index=int(i), row=int(r), column=int(c), old=3, new=2) for i, r, c in positions),
              ['original_index', 'row', 'column', 'old', 'new'])
    write_json(metadata / 'duplicate_groups.json',
               [dict(matrix_sha256=h, representative=ids[0], original_indices=ids)
                for h, ids in groups.items() if len(ids) > 1])
    hash_by_id = {ids[0]: h for h, ids in groups.items()}
    rows = []
    for split, ids in splits.items():
        for c in classes:
            (out / split / c['folder']).mkdir(parents=True)
        for i in ids:
            relative = f'{split}/class_{class_by_id[i]:02d}/wafer_{i:05d}.png'
            target = out / relative
            pixels = rgb(x[i])
            Image.fromarray(pixels).save(target)
            with Image.open(target) as im:
                assert im.mode == 'RGB' and im.size == (52, 52)
                assert np.array_equal(np.asarray(im), pixels)
            rows.append(dict(original_index=int(i), class_id=class_by_id[i], split=split,
                             path=relative, matrix_sha256=hash_by_id[i], png_sha256=sha(target)))
        print(f'{split}: {len(ids)} images exported and verified', flush=True)
    write_csv(metadata / 'manifest.csv', rows,
              ['original_index', 'class_id', 'split', 'path', 'matrix_sha256', 'png_sha256'])
    counts = [dict(**c, raw=int((labels == patterns[c['class_id']]).all(axis=1).sum()),
                   **{s: sum(class_by_id[i] == c['class_id'] for i in ids) for s, ids in splits.items()})
              for c in classes]
    assert sha(args.source) == source_hash, 'Source changed during processing'
    report = dict(version=args.version, source=args.source.name, source_sha256=source_hash,
                  seed=42, source_samples=len(x), unique_samples=len(keep),
                  removed_duplicates=len(x)-len(keep), duplicate_groups=sum(len(ids)>1 for ids in groups.values()),
                  corrected_pixels=len(positions), corrected_samples=len(np.unique(positions[:, 0])),
                  split_counts={k: len(v) for k, v in splits.items()}, classes=counts,
                  matrix_hash_encoding='C-order uint8 52x52 after 3->2',
                  checks=dict(source_unchanged=True, conflicting_labels=0, split_hash_overlap=0,
                              all_png_roundtrips_passed=True, all_splits_have_38_classes=True))
    write_json(metadata / 'summary.json', report)
    (out / 'README.md').write_text(
        '# Wafer38 v1\n\n'
        '原始文件未修改。副本中将噪声 3 映射为 2，然后去重；保留最小原始索引。\n'
        '作者确认噪声：https://github.com/Junliangwangdhu/WaferMap/issues/4#issuecomment-2438186171\n'
        '3→2 是使用者提出的工程假设：https://github.com/Junliangwangdhu/WaferMap/issues/2\n'
        '标签名称依据 Dataset Figure/Single-Type.png：C2–C9 对应 8 维；C7 Near_Full、C9 Random。\n'
        'PNG 为 RGB 52×52，三个通道一致，0/1/2 映射至 0/128/255，不添加文字或裁剪。\n'
        '按类别分层，seed=42，两次划分得到约 70/15/15；未进行增强。\n'
        'metadata 保存类别、原始索引、修正位置、重复组、统计与 SHA-256。全部索引从 0 开始。\n'
        '训练输入 data 指向含 train/val/test 的本目录。测试集仅用于最终评估。\n'
        '本轮未排查近似重复、批次关联或验证噪声修复的真实语义。\n', encoding='utf-8')
    for name in ['manifest.csv', 'classes.json', 'summary.json']:
        shutil.copy2(metadata / name, shared / name)
    files = sorted(p for p in out.rglob('*') if p.is_file())
    (out / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(out).as_posix()}\n' for p in files), encoding='utf-8')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, f'{args.version}/{p.relative_to(out).as_posix()}')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files) + 1
    archive.with_suffix('.zip.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'classes'}, ensure_ascii=False, indent=2))
    print(f'ZIP: {archive} ({archive.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
