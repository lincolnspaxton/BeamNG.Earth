"""Road centre lines with their bends rounded (the user, 2026-11-08: "the roads are too rough and angled").

The map draws a curve as a polygon: a node every 5-30 m, each a 5-30 degree joint. Every road the world builds from
those lines (the painted surface, its lines, curbs and sidewalks, the graded ground) kinked at each node, and a car
following the lane felt each joint. Dirt strips were already rounded (roadmesh.round_dirt_centerline, PR #455); this
rounds every road's line once, where a cell's roads layer is decoded (mapdata._decoded_layer), so every reader sees
the same rounded geometry.

What stays where it was (ROADMAP "deterministic generation": a function of the feature's own data alone, the same in
every cell that holds it):
- both ends of a segment, and every point where another road joins it (Overture's connectors along the segment:
  their "at" positions, part of the segment's own record);
- the line's length, near enough for the rules along it (fractions 0..1: bridges, widths, lanes) to keep their places.
A bend's arc has the radius of its road's class (a motorway's sweep, a residential street's corner), less where its
arms are short: a corner between two junctions 10 m apart stays tight.
"""
import math
import os

import numpy as np

import mapgeom

VERSION = 2                 # 2: every joint over BEND_DEG 0.5 (the lines on a freeway kinked at 1-4 degree joints)
ON = os.environ.get("BNE_ROAD_CURVES", "1") != "0"     # (0: the mapped polygon as before)

BEND_DEG = 0.5              # bends sharper than this are rounded (a 2 degree joint shows in a long lane line)
STEP_DEG = 2.0              # an arc is cut into pieces of at most this
ARM_SHARE = 0.95            # a corner's two arcs may use this share of the segment between them
CONNECTOR_TOL = 0.75        # m: a vertex this close to a connector's position is the junction (pinned)
CLASS_R = {                 # m, the radius a bend of this class is drawn with at most
    "motorway": 900.0, "trunk": 500.0, "primary": 200.0, "secondary": 100.0, "tertiary": 70.0,
    "unclassified": 45.0, "residential": 35.0, "living_street": 20.0, "service": 15.0, "track": 28.0,
    "unknown": 30.0}
DEFAULT_R = 25.0
HALF_W = {"motorway": 7.0, "trunk": 6.0, "primary": 5.0, "secondary": 4.0, "tertiary": 3.5}


def round_line(xy, radius, half, pins=()):
    """xy (k, 2): each bend sharper than BEND_DEG replaced by an arc of radius up to `radius` (shorter arms: less), the
    ends and the vertices `pins` (indices) kept. (m, 2). Pure."""
    P = np.asarray(xy, np.float64)
    n = len(P)
    if n < 3:
        return P
    d = np.diff(P, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    if (seg < 1e-6).any():
        keep = np.r_[True, seg >= 1e-6]
        pin_mask = np.zeros(n, bool)
        pin_mask[list(pins)] = True
        P, pin_mask = P[keep], pin_mask[keep]
        return round_line(P, radius, half, np.nonzero(pin_mask)[0])
    la, lb = seg[:-1], seg[1:]
    t0, t1 = d[:-1] / la[:, None], d[1:] / lb[:, None]
    delta = np.arctan2(t0[:, 0] * t1[:, 1] - t0[:, 1] * t1[:, 0], (t0 * t1).sum(1))
    alpha = np.abs(delta) / 2.0
    ok = (np.abs(delta) > math.radians(BEND_DEG)) & (np.abs(delta) < math.radians(170.0))
    if len(pins):
        pm = np.zeros(n, bool)
        pm[np.asarray(list(pins), np.int64)] = True
        ok &= ~pm[1:-1]
    if not ok.any():
        return P
    want = np.where(ok, radius * np.tan(np.where(ok, alpha, 0.0)), 0.0)    # m of each arm a full arc needs
    back = np.r_[0.0, want, 0.0]
    fwd = back.copy()
    budget = ARM_SHARE * seg
    need = fwd[:-1] + back[1:]
    scale = np.where(need > budget, budget / np.maximum(need, 1e-12), 1.0)
    fwd[:-1] *= scale
    back[1:] *= scale
    dd = np.minimum(back, fwd)[1:-1]
    r = np.where(ok, dd / np.tan(np.where(ok, alpha, 1.0)), 0.0)
    ok &= (dd >= 0.25) & (r >= max(1.0, half * 0.5))
    if not ok.any():
        return P
    out = [P[0]]
    for k in range(n - 2):
        b = P[k + 1]
        if not ok[k]:
            out.append(b)
            continue
        T0 = b - t0[k] * dd[k]
        side = 1.0 if delta[k] > 0 else -1.0
        c = T0 + side * r[k] * np.array([-t0[k, 1], t0[k, 0]])
        a0 = math.atan2(T0[1] - c[1], T0[0] - c[0])
        steps = max(1, int(math.ceil(abs(delta[k]) / math.radians(STEP_DEG))))
        for j in range(steps + 1):
            ang = a0 + delta[k] * j / steps
            out.append(c + r[k] * np.array([math.cos(ang), math.sin(ang)]))
    out.append(P[-1])
    out = np.asarray(out)
    keep = np.r_[True, np.hypot(*np.diff(out, axis=0).T) >= 1e-3]
    return out[keep]


def _pins(xy, ats):
    """Vertex indices of the line xy at the connector fractions ats (0..1 of its length)."""
    if not len(ats):
        return []
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    L = s[-1]
    if L <= 0:
        return []
    out = []
    for a in ats:
        k = int(np.argmin(np.abs(s - float(a) * L)))
        if abs(s[k] - float(a) * L) <= CONNECTOR_TOL:
            out.append(k)
    return out


def round_features(f, frame):
    """The roads layer f (mapgeom.Features, localized: f.en) with every road line's bends rounded (round_line), its
    en and lonlat both (lonlat back through the cell's frame). f itself when nothing changes. Rails are left as they
    are (railways.py rounds its own track)."""
    if f is None or not len(f) or f.en is None:
        return f
    sub = f.attrs.get("subtype")
    cls = f.attrs.get("class")
    con = f.rules.get("connectors")
    con_at = {}
    if con is not None and len(con.feature):
        fe, st = np.asarray(con.feature), np.asarray(con.start, np.float64)
        for i, a in zip(fe.tolist(), st.tolist()):
            con_at.setdefault(i, []).append(a)
    new_en, new_ll, new_coord = [], [], [0]
    changed = False
    for i in range(len(f)):
        for p in range(f.part_off[i], f.part_off[i + 1]):
            for r in range(f.ring_off[p], f.ring_off[p + 1]):
                a, b = int(f.coord_off[r]), int(f.coord_off[r + 1])
                en, ll = f.en[a:b], f.lonlat[a:b]
                is_road = f.kind[i] == mapgeom.LINE and (sub is None or str(sub[i]) == "road")
                if is_road and b - a >= 3 and (con is None or i in con_at):   # (no connectors known: as mapped)
                    c = str(cls[i]) if cls is not None else "unknown"
                    out = round_line(en, CLASS_R.get(c, DEFAULT_R), HALF_W.get(c, 3.0), _pins(en, con_at.get(i, ())))
                    if len(out) != len(en):
                        changed = True
                        lat, lon, _ = frame.enu_to_geodetic(out[:, 0], out[:, 1], np.zeros(len(out)))
                        ll = np.stack([np.asarray(lon, np.float64), np.asarray(lat, np.float64)], 1)
                        ll[0], ll[-1] = f.lonlat[a], f.lonlat[b - 1]       # (the ends exactly as mapped)
                        en = out
                new_en.append(en)
                new_ll.append(ll)
                new_coord.append(new_coord[-1] + len(en))
    if not changed:
        return f
    f.en = np.concatenate(new_en).astype(np.float64)
    f.lonlat = np.concatenate(new_ll).astype(np.float64)
    f.coord_off = np.asarray(new_coord, dtype=np.asarray(f.coord_off).dtype)
    return f
