"""Validate public notebooks without training or loading datasets."""
import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
paths = sorted(ROOT.glob('*/notebooks/*.ipynb'))
assert len(paths) == 3
for path in paths:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        assert cell.execution_count is None and not cell.outputs, path
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith(('RUN_', 'FREEZE_')):
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                            assert node.value.value is False, (path, target.id)
        if 'Local' in path.name and 'PROJECT = Path' in cell.source:
            # Exercise root discovery from both clone root and notebook directory.
            prefix = cell.source.split('PROJECT = Path.cwd().resolve()', 1)[1].split('RUN_TAG', 1)[0]
            locator = 'PROJECT = Path.cwd().resolve()' + prefix.split('assert (PROJECT / "yolo26s_cls/scripts/prepare_data.py").is_file()', 1)[0]
            import os
            before = Path.cwd()
            try:
                for cwd in (ROOT, path.parent):
                    os.chdir(cwd)
                    scope = {'Path': Path}
                    exec(locator, scope)
                    assert scope['PROJECT'] == ROOT
            finally:
                os.chdir(before)
print(f'Validated {len(paths)} notebooks: schema, syntax, clean outputs, switches and root discovery.')
