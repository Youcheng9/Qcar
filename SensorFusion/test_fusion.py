"""python3 -m unittest test_fusion   (from this folder; needs numpy + opencv, no QCar)"""
import os
import tempfile
import unittest

import numpy as np

import fusion


class TestFusion(unittest.TestCase):
    def test_to_points_orientation(self):
        # raw angle FRONT -> straight ahead; raw angles grow clockwise, so FRONT - 90deg is left
        pts = fusion.to_points([fusion.FRONT, fusion.FRONT - np.pi / 2], [1.0, 2.0])
        np.testing.assert_allclose(pts, [[1, 0], [0, 2]], atol=1e-9)

    def test_to_points_drops_invalid(self):
        pts = fusion.to_points([0, 0.1, 0.2, 0.3, np.nan, 0.5],
                               [0.0, np.inf, 0.05, 99.0, 1.0, 1.0])
        self.assertEqual(len(pts), 1)

    def test_to_points_column_vectors_and_mismatch(self):
        # Lidar() starts with (384, 1) arrays; a partial read can leave sizes mismatched
        self.assertEqual(fusion.to_points(np.ones((384, 1)), np.ones((380, 1))).shape, (380, 2))

    def test_pad(self):
        p, m = fusion.pad(np.array([[1.0, 0.0], [0.0, -2.0]]), n=4)
        self.assertEqual(p.shape, (4, 2))
        self.assertEqual(m.tolist(), [True, True, False, False])
        np.testing.assert_allclose(p[0], [0, -2 / fusion.MAX_RANGE])  # sorted by bearing

    def test_pad_truncates_to_nearest(self):
        pts = np.array([[d, 0.0] for d in (4.0, 1.0, 3.0, 2.0)])
        p, m = fusion.pad(pts, n=2)
        self.assertTrue(m.all())
        np.testing.assert_allclose(sorted(p[:, 0] * fusion.MAX_RANGE), [1, 2])

    def test_roundtrip_with_failures(self):
        folder = os.path.join(tempfile.mkdtemp(), 's')
        w = fusion.SessionWriter(folder, 'person')
        img = np.full((220, 420, 3), 128, np.uint8)
        ang = np.linspace(0, 2 * np.pi, 384, endpoint=False)
        w.add(1.0, img, 1.01, ang, np.full(384, 1.5), 0.98, 0.25, 0.05, False)
        w.add(1.1, None, 0, ang, np.full(384, 1.5), 1.08, 0.0, 0.0, True)   # camera failure
        w.add(1.2, img, 1.21, None, None, None, -0.1, 0.05, False)          # lidar failure
        w.close()
        with open(os.path.join(folder, 'images', '000000.jpg'), 'wb') as f:
            f.write(b'not a jpeg')                                          # corrupt file
        d = fusion.load(folder)
        self.assertEqual(d['image'].shape, (3, 120, 160, 3))
        self.assertEqual(d['points'].shape, (3, 384, 2))
        self.assertEqual(d['image_ok'].tolist(), [False, False, True])
        self.assertEqual(d['lidar_ok'].tolist(), [True, True, False])
        self.assertEqual(d['stop'].tolist(), [0, 1, 0])
        self.assertEqual(d['mask'][0].sum(), 384)
        self.assertAlmostEqual(d['skew'][0], 0.03, places=3)
        self.assertTrue(np.isnan(d['skew'][2]))
        self.assertEqual(set(d['tag']), {'person'})


if __name__ == '__main__':
    unittest.main()
