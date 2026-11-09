"""buildings.rect_parts / suburb_flags: suburban houses roofed as houses, not as flat city blocks (the user,
2026-11-08 at St. George UT). The footprints are real (Overture, St. George, local metres)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import buildings as B  # noqa: E402

L_HOUSE = np.array([[5.4, 7.5], [-5.6, 7.5], [-5.5, 0.3], [-2.6, 0.3], [-2.6, -11.6], [5.5, -11.6]])
T_HOUSE = np.array([[5.2, 11.4], [-11.1, 11.3], [-11.0, -4.0], [-4.5, -3.9], [-4.4, -12.4], [5.2, -12.3],
                    [5.1, -3.1], [2.5, -3.1], [2.5, 2.3], [5.3, 2.3]])
RANCH = np.array([[-12.4, -3.9], [18.5, -4.0], [18.6, 5.8], [-12.3, 6.0]])      # 31 x 10 m, 300 m2


def ring(p):
    return B.clean_ring(np.asarray(p, np.float64))


class RectPartsTest(unittest.TestCase):
    def test_t_house_two_parts(self):
        parts = B.rect_parts(ring(T_HOUSE))
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)

    def test_thin_bump_no_part_of_its_own(self):
        # the L's 2.9 m wing is narrower than PARTS_MIN_SIDE_M: one roof over the main body, the nook a low lid
        parts = B.rect_parts(ring(L_HOUSE))
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 1)

    def test_parts_cover_the_footprint(self):
        r = ring(T_HOUSE)
        parts = B.rect_parts(r)
        self.assertIsNotNone(parts)
        cover = sum(4 * hl * hw for _c, _u, hl, hw in parts)
        self.assertGreaterEqual(cover, B.PARTS_COVER * abs(B._area2(r)) / 2 - 1e-6)

    def test_not_rectilinear(self):
        tri = np.array([[0.0, 0.0], [12.0, 0.0], [6.0, 10.0]])
        self.assertIsNone(B.rect_parts(ring(tri)))


class SuburbTest(unittest.TestCase):
    def street(self, big):
        rings = [ring(np.array([[0, 0], [12, 0], [12, 10], [0, 10]], float) + [20 * k - 40, 0]) for k in range(7)]
        rings.append(ring(big + [0, 40]))
        n = len(rings)
        return rings, [{"seed_id": k} for k in range(n)], [0] * n, [0.0] * n

    def test_house_among_houses(self):
        rings, attrs, shared, mins = self.street(RANCH)
        self.assertTrue(B.suburb_flags(rings, attrs, shared, mins)[-1])

    def test_school_among_houses_is_not(self):
        school = np.array([[0, 0], [60, 0], [60, 40], [0, 40]], float)
        rings, attrs, shared, mins = self.street(school)
        self.assertFalse(B.suburb_flags(rings, attrs, shared, mins)[-1])

    def test_suburb_house_pitched_and_low(self):
        for k in range(40):
            sp = B.plan(ring(RANCH), {"seed_id": k, "suburb": True}, 0, 37.1, -113.6, None, True)
            self.assertNotEqual(sp.roof, "flat")
            self.assertLessEqual(sp.floors, 2)

    def test_l_house_pitched(self):
        flat = sum(B.plan(ring(L_HOUSE), {"seed_id": k}, 0, 37.1, -113.6).roof == "flat" for k in range(40))
        self.assertEqual(flat, 0)

    def test_without_context_unchanged(self):
        # a 300 m2 unclassified footprint with no houses around keeps the old prediction (a flat-roofed block)
        self.assertTrue(all(B.plan(ring(RANCH), {"seed_id": k}, 0, 51.5, -0.1).roof == "flat" for k in range(20)))


if __name__ == "__main__":
    unittest.main()
