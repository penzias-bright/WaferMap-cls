import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import wafer_colab as wc


class ColabChecks(unittest.TestCase):
    def test_full_image_transform_preserves_corners_and_values(self):
        pixels = np.zeros((52, 52, 3), dtype=np.uint8)
        pixels[0, 0] = pixels[-1, -1] = 255
        pixels[20:30, 20:30] = 128
        transformed = wc.transform()(Image.fromarray(pixels))
        restored = transformed * torch.tensor(wc.STD)[:, None, None] + torch.tensor(wc.MEAN)[:, None, None]
        self.assertEqual(tuple(restored.shape), (3, 224, 224))
        torch.testing.assert_close(restored[:, 0, 0], torch.ones(3))
        torch.testing.assert_close(restored[:, -1, -1], torch.ones(3))
        self.assertEqual(len(torch.unique((restored * 255).round())), 3)

    def test_macro_metric_uses_all_classes(self):
        validator = wc.WaferValidator(args={'device': 'cpu', 'plots': False})
        validator.targets = [torch.tensor([0, 1])]
        validator.pred = [torch.tensor([[0, 1], [0, 1]])]
        stats = validator.get_stats()
        self.assertAlmostEqual(stats['metrics/macro_f1'], (2/3)/38)
        self.assertEqual(stats['fitness'], stats['metrics/macro_f1'])

    def test_early_stopping_does_not_reset_on_ties(self):
        stopper = wc.StrictEarlyStopping(patience=2)
        self.assertFalse(stopper(1, 0.4))
        self.assertFalse(stopper(2, 0.4))
        self.assertTrue(stopper(3, 0.4))
        self.assertEqual(stopper.best_epoch, 1)

    def test_atomic_backup_and_bad_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_text('checkpoint')
            target = root / 'backup/checkpoint'
            wc.atomic_copy(source, target)
            self.assertEqual(target.read_text(), 'checkpoint')
            self.assertFalse(target.with_name(target.name+'.tmp').exists())
            with self.assertRaises(ValueError):
                wc.unpack(source, root / 'extract')

    def test_unknown_model_is_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            with self.assertRaises(ValueError):
                wc.run_experiment(directory, run, 'E0', model='unknown.pt')
            self.assertFalse(run.exists())


if __name__ == '__main__':
    unittest.main()
