import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'yolo26s_cls/scripts'))
sys.path.insert(0, str(PROJECT / 'wm_peleenet_cls/src'))
import wm_experiment as experiment
from wm_pelee import WMPeleeNet


class WMPeleeTests(unittest.TestCase):
    def test_architecture_and_output(self):
        torch.set_num_threads(2)
        model = WMPeleeNet().eval()
        with torch.inference_mode():
            stem = model.stem(torch.zeros(1, 3, 224, 224))
            first = model.transition1(model.dense1(stem))
            second = model.transition2(model.dense2(first))
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(stem.shape), (1, 32, 56, 56))
        self.assertEqual(tuple(first.shape), (1, 128, 28, 28))
        self.assertEqual(tuple(second.shape), (1, 256, 28, 28))
        self.assertEqual(tuple(output.shape), (1, 38))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 222376)

    def test_existing_split_matches_manifest(self):
        root, classes, rows, _, yolo_test = experiment.data_info(PROJECT)
        data = experiment.datasets(root, classes, rows)
        self.assertEqual({split: len(ds) for split, ds in data.items()},
                         {'train': 25893, 'val': 5549, 'test': 5549})
        self.assertEqual(len(data['test']), yolo_test['samples'])

    def test_evaluation_uses_all_classes_for_macro_f1(self):
        class FixedModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.tensor([[4.0, 1.0], [4.0, 1.0]], device=inputs.device)
        loader = DataLoader(TensorDataset(torch.zeros(2, 1), torch.tensor([0, 1])), batch_size=2)
        result = experiment.evaluate(FixedModel(), loader, torch.device('cpu'), ['a', 'b'])
        self.assertAlmostEqual(result['accuracy'], 0.5)
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)

    def test_freeze_and_resume_guards_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'data'
            (root / 'metadata').mkdir(parents=True)
            (root / 'metadata/manifest.csv').write_text('manifest', encoding='utf-8')
            (root / 'metadata/classes.json').write_text('classes', encoding='utf-8')
            run = base / 'run'
            run.mkdir()
            config = {'seed': 42, 'epochs': 100, 'patience': 15, 'batch': 32,
                      'learning_rate': 0.001, 'weight_decay': 0.0005, 'device': 'cpu', 'gpu': 'CPU'}
            current = experiment.identity(root, config)
            metrics = {'accuracy': 0.8, 'macro_f1': 0.75}
            experiment.save_checkpoint(run / 'best.pt', {'model': WMPeleeNet().state_dict(), 'identity': current,
                                                          'epoch': 3, 'validation': metrics})
            experiment.write_json(run / 'completed.json', {'identity': current, 'best_epoch': 3})
            with patch.object(experiment, 'data_info', return_value=(root, [], [], None, None)):
                experiment.freeze(PROJECT, run, config)
                frozen = json.loads((run / 'frozen.json').read_text(encoding='utf-8'))
                self.assertEqual(frozen['weights_sha256'], experiment.sha(run / 'frozen.pt'))
                with self.assertRaises(FileExistsError):
                    experiment.freeze(PROJECT, run, config)
                with patch.object(experiment, 'datasets', return_value={}):
                    with self.assertRaises(FileExistsError):
                        experiment.train(PROJECT, run, config, torch.device('cpu'), resume=True)

    def test_checkpoint_roundtrip_and_mismatched_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'data'
            (root / 'metadata').mkdir(parents=True)
            (root / 'metadata/manifest.csv').write_text('manifest', encoding='utf-8')
            (root / 'metadata/classes.json').write_text('classes', encoding='utf-8')
            run = base / 'run'
            run.mkdir()
            config = {'seed': 42, 'epochs': 100, 'patience': 15, 'batch': 32,
                      'learning_rate': 0.001, 'weight_decay': 0.0005, 'device': 'cpu', 'gpu': 'CPU'}
            model = WMPeleeNet()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            checkpoint = {'model': model.state_dict(), 'identity': experiment.identity(root, config),
                          'epoch': 1, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                          'scaler': torch.amp.GradScaler('cuda', enabled=False).state_dict(),
                          'history': [], 'best_epoch': 1, 'best_score': 0.5,
                          'torch_rng': torch.get_rng_state(), 'cuda_rng': []}
            experiment.save_checkpoint(run / 'last.pt', checkpoint)
            restored = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(restored['identity'], checkpoint['identity'])
            changed = dict(config, batch=16)
            with patch.object(experiment, 'data_info', return_value=(root, [], [], None, None)), \
                 patch.object(experiment, 'datasets', return_value={}), \
                 patch.object(experiment, 'WMPeleeNet', side_effect=AssertionError('must not train')):
                with self.assertRaisesRegex(ValueError, '恢复失败'):
                    experiment.train(PROJECT, run, changed, torch.device('cpu'), resume=True)

    def test_comparison_requires_same_split_and_frozen_model(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            run, baseline = base / 'run', base / 'baseline'
            run.mkdir()
            baseline.mkdir()
            (run / 'frozen.pt').write_bytes(b'fixed synthetic weights')
            weights_hash = experiment.sha(run / 'frozen.pt')
            scores = {'accuracy': 0.8, 'macro_f1': 0.7}
            experiment.write_json(run / 'frozen.json', {'weights_sha256': weights_hash, 'validation': scores})
            experiment.write_json(baseline / 'frozen.json', {'validation': scores})
            test = {'manifest_sha256': 'same-split', 'weights_sha256': weights_hash,
                    'samples': 10, **scores}
            experiment.write_json(run / 'test_metrics.json', test)
            with patch.object(experiment, 'data_info', return_value=(base, [], [], baseline, test)):
                result = experiment.compare(PROJECT, run)
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0]['test_macro_f1'], 0.7)
                experiment.write_json(run / 'test_metrics.json', dict(test, manifest_sha256='other-split'))
                with self.assertRaisesRegex(ValueError, '不一致'):
                    experiment.compare(PROJECT, run)

    def test_notebook_actions_default_off(self):
        notebook = json.loads((PROJECT / 'wm_peleenet_cls/notebooks/WaferMap_WM_PeleeNet_Local.ipynb').read_text(encoding='utf-8'))
        flags = {'RUN_TRAINING', 'RESUME_TRAINING', 'FREEZE_MODEL', 'RUN_TEST', 'RUN_BENCHMARK'}
        found = {}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in flags:
                            found[target.id] = isinstance(node.value, ast.Constant) and node.value.value is False
        self.assertEqual(set(found), flags)
        self.assertTrue(all(found.values()))


if __name__ == '__main__':
    unittest.main()
