import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_data import clean_and_group, rgb, split_indices


class PreparationTests(unittest.TestCase):
    def test_noise_and_duplicates(self):
        x = np.zeros((3, 52, 52), dtype=np.int32)
        x[0, 1, 1], x[1, 1, 1], x[2, 2, 2] = 3, 2, 1
        y = np.zeros((3, 8), dtype=np.int32)
        cleaned, positions, groups, conflicts = clean_and_group(x, y)
        self.assertEqual(x[0, 1, 1], 3)
        self.assertEqual(cleaned[0, 1, 1], 2)
        self.assertEqual(positions.tolist(), [[0, 1, 1]])
        self.assertEqual(sorted(groups.values()), [[0, 1], [2]])
        self.assertFalse(conflicts)
        y[1, 0] = 1
        self.assertEqual(clean_and_group(x, y)[3][0]['indices'], [0, 1])

    def test_mapping_and_validation(self):
        np.testing.assert_array_equal(rgb(np.array([[0, 1, 2]], dtype=np.uint8)),
                                      [[[0, 0, 0], [128, 128, 128], [255, 255, 255]]])
        with self.assertRaises(ValueError):
            clean_and_group(np.full((1, 52, 52), 4), np.zeros((1, 8)))

    def test_split_reproducible_and_disjoint(self):
        ids = list(range(3800))
        labels = np.repeat(np.arange(38), 100).tolist()
        a = split_indices(ids, labels)
        self.assertEqual(a, split_indices(ids, labels))
        self.assertEqual(sorted(sum(a.values(), [])), ids)
        self.assertEqual([len(a[s]) for s in ['train', 'val', 'test']], [2660, 570, 570])
        for subset in a.values():
            self.assertEqual({labels[i] for i in subset}, set(range(38)))


if __name__ == '__main__':
    unittest.main()
