"""Execute a local validation copy with tiny smoke data; leave the Colab handoff clean."""
import ast
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'yolo26s_cls/notebooks/WaferMap_YOLO26s_Colab.ipynb'
report = ROOT / 'yolo26s_cls/reports/notebook_validation'
report.mkdir(parents=True, exist_ok=True)
notebook = nbformat.read(source, as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    ast.parse(cell.source)
    cell.source = cell.source.replace('"yolo26s_cls/runs/notebook_local"', '"yolo26s_cls/runs/notebook_validation_work"')
    cell.source = cell.source.replace('DATA = wc.unpack(local_zip, WORK / "datasets")',
                                      'assert wc.sha(local_zip) == wc.DATA_SHA\nDATA = PROJECT / "data/processed/wafer38_v1"')
    cell.source = cell.source.replace('wc.make_smoke(DATA, WORK / "smoke_data")',
                                      'wc.make_smoke(DATA, WORK / "smoke_data", train_per_class=2, val_per_class=1)')
kernel_dir = report / 'jupyter/kernels/wafer-check'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps(dict(
    argv=['F:/ProgramData/anaconda3/envs/wafer-yolo26/python.exe', '-I', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    display_name='Wafer local check', language='python')), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(report / 'jupyter')
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ.pop('PYTHONPATH', None)
notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
    '# LOCAL FLOW VALIDATION ONLY\nTiny smoke subset: 2 train/class, 1 val/class. '
    'Uses the already exported local dataset; Drive and Colab are not exercised. No formal scores.'))
client = NotebookClient(notebook, timeout=900, kernel_name='wafer-check', resources={'metadata': {'path': str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, report / 'local_executed.ipynb')
image_count = 0
for cell in notebook.cells:
    for output in cell.get('outputs', []):
        if 'image/png' in output.get('data', {}):
            import base64
            image_count += 1
            (report / f'figure_{image_count}.png').write_bytes(base64.b64decode(output.data['image/png']))
print('Notebook execution passed:', len(notebook.cells), 'cells;', image_count, 'figures')
