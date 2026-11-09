"""roadcurves.py: road bends rounded, ends and junctions kept (the user, 2026-11-08: roads "too rough and angled")."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import roadcurves  # noqa: E402


def turns(xy):
    d = np.diff(xy, axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    t = np.abs((np.diff(a) + math.pi) % (2 * math.pi) - math.pi)
    return np.degrees(t)


class RoundLineTest(unittest.TestCase):
    def test_freeway_joint_rounded(self):
        # a 2 degree joint on a freeway: a visible kink in a long lane line before (roadcurves 1 left joints under 4)
        line = np.array([[0.0, 0.0], [200.0, 0.0], [400.0, 200.0 * math.tan(math.radians(2.0))]])
        out = roadcurves.round_line(line, roadcurves.CLASS_R["motorway"], 7.0)
        self.assertGreater(len(out), 3)
        self.assertLessEqual(turns(out).max(), roadcurves.STEP_DEG + 0.1)

    # an OSM-style 90 degree residential curve: a node every ~15 m, each a 22.5 degree joint
    a = np.linspace(0, math.pi / 2, 5)
    LINE = np.r_[[[-30.0, 0.0]], np.c_[60 * np.sin(a), 60 - 60 * np.cos(a)], [[60.0, 90.0]]]

    def test_joints_become_small(self):
        out = roadcurves.round_line(self.LINE, roadcurves.CLASS_R["residential"], 3.0)
        self.assertGreater(turns(self.LINE).max(), 20.0)
        self.assertLessEqual(turns(out).max(), roadcurves.STEP_DEG + 0.5)

    def test_ends_kept_and_length_close(self):
        out = roadcurves.round_line(self.LINE, 35.0, 3.0)
        np.testing.assert_allclose(out[0], self.LINE[0])
        np.testing.assert_allclose(out[-1], self.LINE[-1])
        L0 = np.hypot(*np.diff(self.LINE, axis=0).T).sum()
        L1 = np.hypot(*np.diff(out, axis=0).T).sum()
        self.assertLess(abs(L1 - L0) / L0, 0.02)       # (the rules' 0..1 fractions keep their places)

    def test_pinned_vertex_stays(self):
        out = roadcurves.round_line(self.LINE, 35.0, 3.0, pins=[3])
        self.assertTrue(np.isclose(np.hypot(*(out - self.LINE[3]).T), 0.0).any())

    def test_straight_and_gentle_unchanged(self):
        line = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.3]])        # a 0.34 degree joint (under BEND_DEG)
        np.testing.assert_array_equal(roadcurves.round_line(line, 35.0, 3.0), line)

    def test_deterministic(self):
        a = roadcurves.round_line(self.LINE, 35.0, 3.0)
        b = roadcurves.round_line(self.LINE.copy(), 35.0, 3.0)
        np.testing.assert_array_equal(a, b)

    def test_short_arms_keep_a_tight_corner(self):
        line = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 6.0]])           # a 90 degree corner, 6 m arms
        out = roadcurves.round_line(line, 400.0, 3.0)
        mid = out[1:-1]
        self.assertTrue(len(mid) > 1)
        self.assertTrue(((mid >= -1e-6) & (mid <= 6.0 + 1e-6)).all())  # inside the corner's arms


if __name__ == "__main__":
    unittest.main()
