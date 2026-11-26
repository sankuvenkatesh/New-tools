bl_info = {
    "name": "Win Animations - 24 Variations + Rig Creator",
    "author": "Venkatesh Sanku",
    "version": (2, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N) > Win Anim",
    "description": "Create win rigs (MASSIVE/BIG/HUGE) + 24 Entry/Idle/Exit animations",
    "category": "Animation",
}

# Changes in 2.0.0:
#  ANIMATION
#  - All 12 animations re-timed with the principles of animation, and 12 new ones (V13 - V24).
#  - Overlapping action done right: every letter now has the SAME duration and the cascade only
#    offsets it (before, all letters had to arrive on one frame, so late letters were rushed).
#    The cascade runs in reading order with an extra beat before the next word (MASSIVE ... WIN!).
#  - Springs, follow-through and drag are real damped motion (baked), squash & stretch keeps the
#    volume, rotations and pops happen around the letter centre, flights travel on arcs.
#  - No dead holds any more: the entry keeps settling right up to frame 30 and the idle eases in
#    over LOOP_RAMP frames (it used to sit completely still for 7-12 frames around frame 30).
#  - Seamless loop: speed is zero on 30 / 31 / 85 / 86, so ENTRY > LOOP > LOOP > EXIT never kinks.
#  - Word parent bones (MASSIVE_ROOT, WIN_ROOT) are now animated for word-level accents and keyed
#    to rest in every action, so switching actions never leaves a stale word pose behind.
#  - Panel: Intensity (exaggeration) and Stagger sliders, a variation picker, section preview
#    (play Entry / Loop / Exit), timeline markers.
#  BUG FIXES
#  - ELASTIC keys had amplitude 0, which makes Blender cancel the wobble: letters snapped into
#    place in 1-2 frames (V1, V5, V7, V8, V9). Amplitude is now set per channel.
#  - New bones are QUATERNION by default, so Euler keys did nothing until a build switched them;
#    the rig creator and Apply now set XYZ.
#  - Letter mesh names clashed between the three rigs (I1_MESH.001 ...): now prefixed per rig.
#  - The V4 camera shake was random noise on every 2nd frame (jittery); now a damped shake.
#  - Word parent bones were missing from the axis detection and from Verify.
#  - Keys are written in bulk, flat runs are removed (much smaller actions, faster builds).

import bpy
import math
import random
import re
import traceback

# ============================================================
# CONSTANTS
# ============================================================

RIG_NAMES = ["MASSIVE WIN_ARMATURE", "BIG WIN_ARMATURE", "HUGE WIN_ARMATURE"]
RIG_WORDS = ["MASSIVE WIN", "BIG WIN", "HUGE WIN"]

# Manual axis mapping. It is only used when "Auto-detect bone axes" is switched
# off in the panel (or when a bone's axes can not be worked out).
# Each value is the index of the bone-LOCAL axis: 0 = X, 1 = Y, 2 = Z.
AX_SIDE, AX_DEPTH, AX_UP = 0, 1, 2
ROT_PITCH, ROT_ROLL, ROT_YAW = 0, 1, 2
SORT_LEFT_TO_RIGHT = True

# Clip layout - identical for all variations
#   ENTRY  F_START      .. F_ENTRY_END   (1 .. 30)
#   LOOP   F_LOOP_START .. F_IDLE_END    (31 .. 85)   first and last frame = same pose
#   EXIT   F_EXIT_START .. F_END         (86 .. 100)  first frame = same pose as loop end
F_START, F_ENTRY_END = 1, 30
F_LOOP_START, F_IDLE_END = 31, 85
F_EXIT_START, F_END = 86, 100
HIDDEN = 0.001

# Rig creator layout
STACK_WORDS = True       # True: every word on its own line (MASSIVE over WIN). False: one line
LETTER_SPACING = 1.0     # distance between two letters
LINE_HEIGHT = 1.25       # distance between two lines of words
WORD_GAP = 0.5           # extra distance between words when STACK_WORDS is False
LETTER_SIZE = 0.9
LETTER_CENTER = 0.5      # height of the middle of a letter above its bone
LETTER_EXTRUDE = 0.05
BONE_LENGTH = 0.5

# TIMING & SPACING (frames, designed for 30 fps). Every animation reads these.
ENTRY_LEAD = 1.0          # frames before the first letter moves
ENTRY_GAP = 1.25          # frames between two letters of a cascade (overlapping action)
ENTRY_WORD_BEAT = 3.0     # extra beat before the next word starts (MASSIVE ... WIN!)
ENTRY_STAGGER_MAX = 9.0   # a whole cascade never spreads over more frames than this
EXIT_GAP = 0.6
EXIT_WORD_BEAT = 1.0
EXIT_STAGGER_MAX = 4.0
ANTIC = 3.0               # anticipation (wind-up) frames before an exit action
LOOP_RAMP = 7.0           # frames the idle needs to ease out of / back into the rest pose
BACK_OVERSHOOT = 1.9      # overshoot of Blender's BACK keyframe easing
PIVOT_H = LETTER_CENTER   # pops, spins and flips turn around the letter centre (not its foot)
HINGE_H = LETTER_CENTER + 0.36   # top edge of a letter (V16 hangs the letters from here)


# ============================================================
# EASING & MOTION MATH  (pure functions, u = 0..1)
# All of them start / end exactly on 0 / 1 and most end with zero speed, so the
# poses on the clip borders are exact and the speed there is zero.
# ============================================================

def clamp01(x):
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def seg(f, a, b):
    """0..1 progress of frame f between frame a and frame b (clamped)."""
    if b <= a:
        return 1.0 if f >= b else 0.0
    return clamp01((f - a) / float(b - a))


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(u):
    u = clamp01(u)
    return u * u * (3.0 - 2.0 * u)


def smoother(u):
    u = clamp01(u)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def linear(u):
    return clamp01(u)


def in_quad(u):
    u = clamp01(u)
    return u * u


def out_quad(u):
    u = clamp01(u)
    return 1.0 - (1.0 - u) * (1.0 - u)


def in_cubic(u):
    u = clamp01(u)
    return u * u * u


def out_cubic(u):
    u = clamp01(u)
    return 1.0 - (1.0 - u) ** 3


def io_cubic(u):
    u = clamp01(u)
    return 4.0 * u ** 3 if u < 0.5 else 1.0 - (-2.0 * u + 2.0) ** 3 / 2.0


def in_quart(u):
    u = clamp01(u)
    return u ** 4


def out_quart(u):
    u = clamp01(u)
    return 1.0 - (1.0 - u) ** 4


def in_expo(u):
    u = clamp01(u)
    return (2.0 ** (10.0 * u) - 1.0) / 1023.0


def out_expo(u):
    u = clamp01(u)
    return (1.0 - 2.0 ** (-10.0 * u)) / (1.0 - 2.0 ** -10.0)


def io_sine(u):
    u = clamp01(u)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def out_back(u, s=1.7):
    u = clamp01(u)
    t = u - 1.0
    return 1.0 + (s + 1.0) * t * t * t + s * t * t


def in_back(u, s=1.7):
    u = clamp01(u)
    return u * u * ((s + 1.0) * u - s)


def back_out(s):
    return lambda u: out_back(u, s)


def back_in(s):
    return lambda u: in_back(u, s)


def spring(u, overshoot=0.2, peak=0.32, soft=False, taper=0.55):
    """0 -> 1 like a damped spring: fast start, ONE clear overshoot of about `overshoot`
    near u = peak, a small undershoot, then it is tapered so it lands exactly on 1 with
    zero speed at u = 1.  soft=True starts with zero speed (mass released from rest)."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    ov = max(1e-4, overshoot)
    w = math.pi / peak
    a = -math.log(ov) / peak
    if soft:
        osc = math.exp(-a * u) * (math.cos(w * u) + (a / w) * math.sin(w * u))
    else:
        osc = math.exp(-a * u) * math.cos(w * u)
    return 1.0 - osc * (1.0 - smoother(seg(u, taper, 1.0)))


def settle(u, cycles=2.0, decay=4.0, taper=0.5):
    """1 -> 0 damped oscillation (a released pendulum / wobble). Exactly 0 with zero speed at u = 1."""
    if u <= 0.0:
        return 1.0
    if u >= 1.0:
        return 0.0
    v = math.exp(-decay * u) * math.cos(2.0 * math.pi * cycles * u)
    return v * (1.0 - smoother(seg(u, taper, 1.0)))


_KICK_NORM = {}


def kick(u, cycles=1.25, decay=5.0):
    """0 -> +1 -> small recoil -> 0: the reaction to a hit (camera kick, word accent).
    Peak value is exactly 1, ends on 0 with zero speed."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    keyc = (cycles, decay)
    if keyc not in _KICK_NORM:
        _KICK_NORM[keyc] = max(abs(math.exp(-decay * x / 400.0) *
                                   math.sin(2.0 * math.pi * cycles * x / 400.0)) for x in range(1, 400))
    v = math.exp(-decay * u) * math.sin(2.0 * math.pi * cycles * u)
    return v / _KICK_NORM[keyc] * (1.0 - smoother(seg(u, 0.55, 1.0)))


def bump(f, c, rise, fall=None):
    """Smooth 0 -> 1 -> 0 pulse peaking on frame c. Zero outside [c - rise, c + fall]."""
    if fall is None:
        fall = rise
    if f <= c - rise or f >= c + fall:
        return 0.0
    if f <= c:
        return smoother(1.0 - (c - f) / rise)
    return smoother(1.0 - (f - c) / fall)


def hop(u):
    """Parabolic jump 0 -> 1 -> 0 (gravity arc)."""
    u = clamp01(u)
    return 4.0 * u * (1.0 - u)


def pw(t, *pts):
    """Piecewise curve:  pw(t, (t0, v0), (t1, v1, ease), (t2, v2, ease), ...)
    Every segment uses its own ease (default smooth). Holds the first / last value outside."""
    if t <= pts[0][0]:
        return pts[0][1]
    for a, b in zip(pts, pts[1:]):
        if t <= b[0]:
            ease = b[2] if len(b) > 2 else smooth
            u = (t - a[0]) / float(b[0] - a[0]) if b[0] > a[0] else 1.0
            return a[1] + (b[1] - a[1]) * ease(u)
    return pts[-1][1]


def bounce_drop(u, bounces=2, rest=0.3):
    """Height 1 -> 0 of a dropped ball (u 0..1): a free fall, then `bounces` hops, each
    `rest` times as high. Returns (height, [impact times as u])."""
    times = [1.0]                                   # fall from height 1 takes "1"
    for k in range(1, bounces + 1):
        times.append(2.0 * math.sqrt(rest ** k))    # air time of bounce k
    total = sum(times)
    impacts, acc = [], 0.0
    for d in times:
        acc += d
        impacts.append(acc / total)
    u = clamp01(u)
    if u >= 1.0:
        return 0.0, impacts
    t = u * total
    if t < times[0]:
        x = t / times[0]
        return 1.0 - x * x, impacts
    t -= times[0]
    for k in range(1, bounces + 1):
        d = times[k]
        if t < d:
            x = t / d
            return (rest ** k) * 4.0 * x * (1.0 - x), impacts
        t -= d
    return 0.0, impacts


def noise1(f, seed=0.0):
    """Smooth deterministic noise (-1..1): a sum of incommensurate sines."""
    return (math.sin(f * 1.31 + seed * 2.1) * 0.5 + math.sin(f * 2.17 + seed * 5.3) * 0.3
            + math.sin(f * 3.73 + seed * 1.7) * 0.2)


def ddf(fn, f, h=0.5):
    """Speed (per frame) of a scalar curve fn(f)."""
    return (fn(f + h) - fn(f - h)) / (2.0 * h)


def stretch_of(speed, k=1.0, lo=0.75, hi=1.45):
    """Squash & stretch factor from a speed: fast = longer along the motion."""
    return max(lo, min(hi, 1.0 + k * abs(speed)))


def env(f):
    """Idle envelope: 0 on frame 31, eases to 1 over LOOP_RAMP frames, back to 0 on frame 85.
    Zero value AND zero speed on both ends -> perfectly seamless loop seam."""
    return smoother(seg(f, F_LOOP_START, F_LOOP_START + LOOP_RAMP)) * \
        smoother(clamp01((F_IDLE_END - f) / LOOP_RAMP))


def loop_u(f):
    return (f - F_LOOP_START) / float(F_IDLE_END - F_LOOP_START)


def lwave(f, cycles, phase=0.0):
    """Idle sine with a whole number of cycles over the loop, eased in / out by env()."""
    return env(f) * math.sin(2.0 * math.pi * cycles * loop_u(f) - phase)


def lwave2(f, cycles, phase=0.0):
    """Like lwave, but a cosine-shaped bob that rests between moves (0..1, never negative)."""
    return env(f) * (0.5 - 0.5 * math.cos(2.0 * math.pi * cycles * loop_u(f) - phase))


# ============================================================
# TRANSFORM HELPERS
# All vectors are written in "logical" order: (side, depth, up) = world (X, Y, Z).
# key() converts them to the pose-bone channels of every bone.
# ============================================================

def LOC(up=0.0, side=0.0, depth=0.0):
    return (side, depth, up)


def ROT(pitch=0.0, roll=0.0, yaw=0.0):
    # pitch = around the side axis (X), roll = around the depth axis (Y), yaw = around the up axis (Z)
    return (math.radians(pitch), math.radians(roll), math.radians(yaw))


def SCL(s=1.0, up=1.0, side=1.0, depth=None):
    if depth is None:
        depth = side
    return (s * side, s * depth, s * up)


def SQ(s=1.0, stretch=1.0):
    """Uniform scale s with a VOLUME-PRESERVING vertical stretch (>1) / squash (<1)."""
    st = max(0.05, stretch)
    k = 1.0 / math.sqrt(st)
    return (s * k, s * k, s * st)


def SQX(s=1.0, stretch=1.0):
    """Uniform scale s with a volume-preserving horizontal (side) stretch / squash."""
    st = max(0.05, stretch)
    k = 1.0 / math.sqrt(st)
    return (s * st, s * k, s * k)


def vadd(*vs):
    out = [0.0, 0.0, 0.0]
    for v in vs:
        if v is not None:
            out[0] += v[0]
            out[1] += v[1]
            out[2] += v[2]
    return tuple(out)


REST_LOC, REST_ROT, REST_SCL = LOC(), ROT(), SCL()
SMOOTH = ("BEZIER", "AUTO", 0.0)
STEP = ("CONSTANT", "AUTO", 0.0)


def S(interp, ease="EASE_OUT", period=0.4):
    return (interp, ease, period)
# ============================================================
# EXACT AXIS CONVERSION  (pure Python 3x3 math, M[row][col])
# ============================================================

def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mt(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _det3(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _euler_to_mat(e):
    """Blender 'XYZ' Euler (X is applied first, then Y, then Z) -> Rz * Ry * Rx."""
    return _mm(_rz(e[2]), _mm(_ry(e[1]), _rx(e[0])))


_TWO_PI = 2.0 * math.pi


def _near(angle, ref):
    """angle + k * 360 deg, choosing k so the result is as close as possible to ref."""
    return angle + _TWO_PI * round((ref - angle) / _TWO_PI)


def _mat_to_euler_near(M, ref):
    """'XYZ' Euler angles of the rotation matrix M that are closest to `ref`.
    Both Euler solutions are tried and every angle is shifted by whole turns, so the
    winding the animator asked for (360 / 540 degree spins ...) is kept."""
    cy = math.hypot(M[0][0], M[1][0])
    cands = []
    if cy > 1e-8:
        cands.append((math.atan2(M[2][1], M[2][2]),
                      math.atan2(-M[2][0], cy),
                      math.atan2(M[1][0], M[0][0])))
        cands.append((math.atan2(-M[2][1], -M[2][2]),
                      math.atan2(-M[2][0], -cy),
                      math.atan2(-M[1][0], -M[0][0])))
    else:
        # gimbal lock (y = +-90 deg): only x -/+ z is defined, keep z of the reference
        y = math.copysign(math.pi / 2.0, -M[2][0])
        z = ref[2]
        xb = math.atan2(-M[1][2], M[1][1])
        cands.append((xb + z if y > 0.0 else xb - z, y, z))
    best, best_d = None, None
    for c in cands:
        t = tuple(_near(c[i], ref[i]) for i in range(3))
        d = sum((t[i] - ref[i]) ** 2 for i in range(3))
        if best is None or d < best_d:
            best, best_d = t, d
    return best


def _snap(vec, one=False):
    """Remove float noise (1e-17 ...) so rest poses are exactly 0 / 1."""
    base = 1.0 if one else 0.0
    return tuple((base if abs(c - base) < 1e-12 else c) + 0.0 for c in vec)


def _clean_axis_value(x):
    if abs(x) < 1e-9:
        return 0.0
    if abs(abs(x) - 1.0) < 1e-9:
        return math.copysign(1.0, x)
    return x


# Everything the builders "key" is collected here first and written to the
# Action in one go afterwards (see write_action).
TRACK = {}    # (bone name, property) -> {frame: ((x, y, z), style)}
AXINFO = {}   # bone name -> axis information (see _bone_frame / _manual_info)


def _manual_info():
    return {"auto": False,
            "loc": ((AX_SIDE, 1.0), (AX_DEPTH, 1.0), (AX_UP, 1.0)),
            "rot": ((ROT_PITCH, 1.0), (ROT_ROLL, 1.0), (ROT_YAW, 1.0))}


def _manual_convert(vec, amap, scale=False):
    out = [1.0, 1.0, 1.0] if scale else [0.0, 0.0, 0.0]
    for k in range(3):
        out[amap[k][0]] = vec[k]
    return tuple(out)


def _bone_frame(armature, pb):
    """Orientation of a pose bone in the world.
    A    : 3x3 matrix whose columns are the bone's local X/Y/Z axes in world space
           (orthonormal: a rotation, or a rotation + mirror)
    perm : which bone axis (index, sign) lies along world X (side), Y (depth), Z (up)
    det  : +1 for a rotation, -1 for a mirrored frame
    Returns None when the frame is sheared or not lined up with the world axes."""
    try:
        mw = armature.matrix_world
        ml = pb.bone.matrix_local
        cols, lens = [], []
        for j in range(3):
            a = [ml[r][j] for r in range(3)]
            w = [mw[r][0] * a[0] + mw[r][1] * a[1] + mw[r][2] * a[2] for r in range(3)]
            ln = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
            if ln < 1e-9:
                return None
            cols.append([c / ln for c in w])
            lens.append(ln)
        A = [[_clean_axis_value(cols[j][r]) for j in range(3)] for r in range(3)]
        G = _mm(_mt(A), A)
        if max(abs(G[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3)) > 1e-3:
            return None                                   # sheared / non-uniform scale
        perm, used = [], set()
        for k in range(3):                                # world X, Y, Z
            best, score = None, 0.0
            for j in range(3):
                if j not in used and abs(A[k][j]) > score:
                    best, score = j, abs(A[k][j])
            if best is None or score < 0.85:
                return None
            used.add(best)
            perm.append((best, 1.0 if A[k][best] > 0.0 else -1.0))
        # exact: the bone axes lie exactly on the world axes (a signed permutation matrix)
        exact = all(abs(A[k][j] - (perm[k][1] if j == perm[k][0] else 0.0)) < 1e-9
                    for k in range(3) for j in range(3))
        return {"auto": True, "A": A, "At": _mt(A), "perm": tuple(perm), "exact": exact,
                "det": 1.0 if _det3(A) > 0.0 else -1.0, "sidelen": lens[perm[0][0]]}
    except Exception as err:
        print("WARNING bone frame:", err)
        return None


def _loc_to_local(info, v):
    """World-aligned displacement (side, depth, up) -> pose-bone location:  t = A^T * v."""
    if not info["auto"]:
        return _snap(_manual_convert(v, info["loc"]))
    return _snap(_mv(info["At"], v))


def _scl_to_local(info, v):
    """Scale along the world axes -> scale along the bone axes that lie on them."""
    if not info["auto"]:
        return _snap(_manual_convert(v, info["loc"], True), True)
    out = [1.0, 1.0, 1.0]
    for k in range(3):
        out[info["perm"][k][0]] = v[k]
    return _snap(out, True)


def _rot_to_local(info, v):
    """World-axis rotation (pitch about X, roll about Y, yaw about Z; Euler XYZ) ->
    pose-bone 'XYZ' Euler:   R_bone = A^T * R_world * A,  then back to Euler angles."""
    if not info["auto"]:
        return _snap(_manual_convert(v, info["rot"]))
    nonzero = sum(1 for c in v if abs(c) > 1e-12)
    if nonzero == 0:
        return (0.0, 0.0, 0.0)
    ref = [0.0, 0.0, 0.0]
    for k in range(3):
        j, sgn = info["perm"][k]
        ref[j] = sgn * info["det"] * v[k]        # exact for a single-axis rotation on aligned bones
    if nonzero == 1 and info["exact"]:
        return _snap(ref)
    Rw = _euler_to_mat(v)
    Rb = _mm(info["At"], _mm(Rw, info["A"]))
    return _snap(_mat_to_euler_near(Rb, ref))


def _record(bone_name, prop, frame, vec, style):
    TRACK.setdefault((bone_name, prop), {})[frame] = (vec, style)


def key(bone, frame, loc=None, rot=None, scl=None, style=SMOOTH):
    frame = int(round(frame))
    info = AXINFO.get(bone.name)
    if info is None:
        info = _manual_info()
    if loc is not None:
        _record(bone.name, "location", frame, _loc_to_local(info, loc), style)
    if rot is not None:
        _record(bone.name, "rotation_euler", frame, _rot_to_local(info, rot), style)
    if scl is not None:
        _record(bone.name, "scale", frame, _scl_to_local(info, scl), style)


def key_all(bone, frame, loc=None, rot=None, scl=None, style=SMOOTH):
    key(bone, frame,
        REST_LOC if loc is None else loc,
        REST_ROT if rot is None else rot,
        REST_SCL if scl is None else scl, style)


_BORDER_FRAMES = (F_ENTRY_END, F_LOOP_START, F_IDLE_END, F_EXIT_START)
_BORDER_SET = frozenset(_BORDER_FRAMES)


def ensure_boundaries(bones):
    """Every channel of every bone gets a key (rest pose) on the four clip borders
    30 / 31 / 85 / 86. This makes entry-end, loop-start, loop-end and exit-start identical
    and explicit, and gives every action the same set of channels."""
    for b in bones:
        for prop, rest in (("location", (0.0, 0.0, 0.0)),
                           ("rotation_euler", (0.0, 0.0, 0.0)),
                           ("scale", (1.0, 1.0, 1.0))):
            frames = TRACK.setdefault((b.name, prop), {})
            for f in _BORDER_FRAMES:
                frames[f] = (rest, SMOOTH)


def bake(bone, f0, f1, fn, style=SMOOTH):
    """Key fn(f) -> (loc, rot, scl) on every frame f0..f1 (None = rest pose).
    The four border frames are always the exact rest pose (see ensure_boundaries)."""
    for f in range(int(f0), int(f1) + 1):
        if f in _BORDER_SET:
            continue
        loc, rot, scl = fn(f)
        key_all(bone, f, loc, rot, scl, style)


def pivot(rot=None, scl=None, h=None):
    """Location that makes a rotation / scale happen around the point h above the bone
    (the letter centre by default) instead of around the bone head (the letter's foot).
    Pose bones apply  location + R * S * p,  so  loc = p - R * S * p  keeps p in place."""
    if h is None:
        h = PIVOT_H
    p = [0.0, 0.0, h]
    if scl is not None:
        p2 = [scl[0] * p[0], scl[1] * p[1], scl[2] * p[2]]
    else:
        p2 = p
    if rot is not None and any(abs(c) > 1e-12 for c in rot):
        p2 = _mv(_euler_to_mat(rot), p2)
    return (p[0] - p2[0], p[1] - p2[1], p[2] - p2[2])


def sample(bone, f0, f1, step, fn, style=SMOOTH):
    for f in list(range(f0, f1, step)) + [f1]:
        loc, rot, scl = fn(f)
        key(bone, f, loc, rot, scl, style)


def tpos(i, n):
    return i / float(n - 1) if n > 1 else 0.0


def alt(i):
    return -1.0 if i % 2 else 1.0


# ============================================================
# WORDS AND LETTER POSITIONS
# ============================================================

CTX = {"wx": {}, "wz": {}, "k": {}, "center": 0.0, "span": 0.0, "cz": 0.0,
       "gc": {}, "gspan": {}, "gpos": {}, "groups": [], "roots": [], "wroots": [],
       "rel": {}, "exag": 1.0, "stagger": 1.0}


def _is_letter_bone(pb):
    n = pb.name
    return n != "GLOBAL_MASTER" and not n.endswith("_ROOT") and n.endswith("_BONE")


def _world_pos(armature, pb):
    mw = armature.matrix_world
    h = pb.bone.head_local
    return tuple(mw[r][0] * h[0] + mw[r][1] * h[1] + mw[r][2] * h[2] + mw[r][3] for r in range(3))


def _split_rows(items, pos):
    """Letters that have no word parent bone: split them into text rows by their height."""
    if len(items) < 2:
        return [list(items)]
    xs = sorted(pos(b)[0] for b in items)
    gaps = sorted(b - a for a, b in zip(xs, xs[1:]) if b - a > 1e-6)
    typical = gaps[len(gaps) // 2] if gaps else 1.0
    tol = max(0.5 * typical, 1e-3)
    rows = []
    for b in sorted(items, key=lambda p: -pos(p)[2]):
        z = pos(b)[2]
        if rows and abs(z - rows[-1][0]) <= tol:
            rows[-1][1].append(b)
        else:
            rows.append([z, [b]])
    return [r[1] for r in rows]


def get_word_groups(armature):
    """[(root bone or None, [letter bones left -> right]), ...] in reading order:
    the top word first, then the words below it.  A word is every letter that shares the
    same parent bone (MASSIVE_ROOT, WIN_ROOT ...). Letters that hang directly on
    GLOBAL_MASTER are split into text rows by their height instead."""
    like = [pb for pb in armature.pose.bones if _is_letter_bone(pb)]
    carriers = {pb.parent.name for pb in like if pb.parent is not None}
    letters = [pb for pb in like if pb.name not in carriers]    # a bone that carries letters is a word parent
    if not letters:
        return []
    pos = {pb.name: _world_pos(armature, pb) for pb in letters}
    by_parent, order = {}, []
    for pb in letters:
        p = pb.parent
        pkey = p.name if (p is not None and p.name != "GLOBAL_MASTER") else None
        if pkey not in by_parent:
            by_parent[pkey] = []
            order.append(pkey)
        by_parent[pkey].append(pb)
    groups = []
    for pkey in order:
        members = by_parent[pkey]
        if pkey is None:
            for row in _split_rows(members, lambda b: pos[b.name]):
                groups.append([None, row])
        else:
            groups.append([armature.pose.bones.get(pkey), members])
    if SORT_LEFT_TO_RIGHT:
        for g in groups:
            g[1].sort(key=lambda b: pos[b.name][0])

        def gkey(g):
            zs = [pos[b.name][2] for b in g[1]]
            xs = [pos[b.name][0] for b in g[1]]
            return (-round(sum(zs) / len(zs), 3), round(sum(xs) / len(xs), 3))
        groups.sort(key=gkey)
    return [(r, ls) for r, ls in groups]


def get_letter_bones(armature):
    """All letter bones in reading order (top word first, each word left to right)."""
    return [b for _root, ls in get_word_groups(armature) for b in ls]


def prepare_context(armature, letters, master=None, auto_axes=True, groups=None,
                    exag=1.0, stagger=1.0):
    CTX["wx"], CTX["wz"], CTX["k"], CTX["center"], CTX["span"], CTX["cz"] = {}, {}, {}, 0.0, 0.0, 0.0
    CTX["gc"], CTX["gspan"], CTX["gpos"], CTX["rel"] = {}, {}, {}, {}
    CTX["exag"], CTX["stagger"] = float(exag), float(stagger)
    if groups is None:
        groups = get_word_groups(armature)
    CTX["groups"] = groups
    CTX["wroots"] = [r for r, _ in groups]
    CTX["roots"] = [r for r in CTX["wroots"] if r is not None]
    AXINFO.clear()
    for pb in ([master] if master is not None else []) + CTX["roots"] + list(letters):
        fr = _bone_frame(armature, pb) if auto_axes else None
        AXINFO[pb.name] = fr if fr is not None else _manual_info()
    try:
        mw = armature.matrix_world
        for pb in letters:
            h = pb.bone.head_local
            CTX["wx"][pb.name] = mw[0][0] * h[0] + mw[0][1] * h[1] + mw[0][2] * h[2] + mw[0][3]
            CTX["wz"][pb.name] = mw[2][0] * h[0] + mw[2][1] * h[1] + mw[2][2] * h[2] + mw[2][3]
            info = AXINFO[pb.name]
            if info["auto"]:
                # world distance -> armature units (the sign is handled by the axis conversion)
                CTX["k"][pb.name] = 1.0 / info["sidelen"] if info["sidelen"] > 1e-9 else 1.0
            else:
                ml = pb.bone.matrix_local
                axis = [ml[r][AX_SIDE] for r in range(3)]      # bone axis that carries "side"
                dir_x = mw[0][0] * axis[0] + mw[0][1] * axis[1] + mw[0][2] * axis[2]
                CTX["k"][pb.name] = 1.0 / dir_x if abs(dir_x) > 1e-6 else 1.0
        xs = list(CTX["wx"].values())
        zs = list(CTX["wz"].values())
        CTX["center"] = (min(xs) + max(xs)) / 2.0
        CTX["span"] = max(xs) - min(xs)
        CTX["cz"] = (min(zs) + max(zs)) / 2.0
        for pb in letters:
            kk = abs(CTX["k"][pb.name])
            CTX["rel"][pb.name] = ((CTX["wx"][pb.name] - CTX["center"]) * kk,
                                   (CTX["wz"][pb.name] - CTX["cz"]) * kk)
        for gi, (_root, ls) in enumerate(groups):
            gx = [CTX["wx"][b.name] for b in ls if b.name in CTX["wx"]]
            if not gx:
                continue
            c, sp = (min(gx) + max(gx)) / 2.0, max(gx) - min(gx)
            for k, b in enumerate(ls):
                CTX["gc"][b.name], CTX["gspan"][b.name] = c, sp
                CTX["gpos"][b.name] = (gi, k, len(ls))
    except Exception as err:
        print("WARNING prepare_context:", err)


def side_by(b, dx_world):
    return dx_world * CTX["k"].get(b.name, 1.0)


def group_info(b, i, n):
    """(index of the letter inside its word, number of letters of that word)"""
    g = CTX["gpos"].get(b.name)
    return (g[1], g[2]) if g is not None else (i, n)


def tpos_word(b, i, n):
    """0..1 position of the letter inside ITS word (tpos() counts over the whole text)."""
    k, m = group_info(b, i, n)
    return k / float(m - 1) if m > 1 else 0.0


def letter_offset(b, i, n):
    """World distance of a letter from the middle of its own word."""
    if b.name in CTX["wx"] and CTX["gspan"].get(b.name, CTX["span"]) > 1e-4:
        return CTX["wx"][b.name] - CTX["gc"].get(b.name, CTX["center"])
    k, m = group_info(b, i, n)
    return (k - (m - 1) / 2.0) * 0.5


class LI:
    """Everything an animation needs to know about one letter."""
    __slots__ = ("b", "name", "i", "n", "gi", "k", "m", "s", "sd", "a", "dx", "rx", "rz", "seed")

    def __init__(self, b, i, n):
        self.b, self.name, self.i, self.n = b, b.name, i, n
        g = CTX["gpos"].get(b.name)
        self.gi = g[0] if g is not None else 0
        self.k, self.m = group_info(b, i, n)
        self.s = 2.0 * self.k / (self.m - 1) - 1.0 if self.m > 1 else 0.0   # -1 left .. +1 right of its word
        self.sd = 0.0 if abs(self.s) < 1e-6 else math.copysign(1.0, self.s)
        self.a = alt(i)
        self.dx = side_by(b, letter_offset(b, i, n))      # distance from the word centre (bone units)
        self.rx, self.rz = CTX["rel"].get(b.name, (self.dx, 0.0))   # from the text centre
        self.seed = (i * 7919 + 13) % 1000 / 1000.0


def letter_infos(letters):
    n = len(letters)
    return [LI(b, i, n) for i, b in enumerate(letters)]


def word_count(L):
    return max(li.gi for li in L) + 1 if L else 0


def cascade(L, gap, beat, cap, order="reading", words="sequence", rng=None):
    """Start offset (frames, float) of every letter -> {name: offset}.
    order : reading | reverse | rtl | center (word centre first) | edges | random | radial
    words : sequence (next word starts after the last one + beat) | together
    The cascade is squeezed so it never spreads over more than `cap` frames, and the
    panel's Stagger slider scales it. Letters keep an EVEN rhythm (fractional frames)."""
    st = CTX.get("stagger", 1.0)
    gap, beat, cap = gap * st, beat * st, cap * st
    ng = word_count(L)
    by_word = [[li for li in L if li.gi == g] for g in range(ng)]
    rank = {}
    start = 0.0
    for g, ws in enumerate(by_word):
        if not ws:
            continue
        m = len(ws)
        c = (m - 1) / 2.0
        keys = {}
        for li in ws:
            if order == "center":
                keys[li.name] = abs(li.k - c)
            elif order == "edges":
                keys[li.name] = c - abs(li.k - c)
            elif order == "rtl":
                keys[li.name] = m - 1 - li.k              # words in order, each right to left
            elif order == "radial":
                keys[li.name] = math.hypot(li.rx, li.rz)
            else:
                keys[li.name] = li.k
        if order == "random":
            idx = list(range(m))
            (rng or random.Random(5 + g)).shuffle(idx)
            keys = {li.name: idx[li.k] for li in ws}
        if words == "together":
            start = 0.0
        for li in ws:
            rank[li.name] = start + keys[li.name] * gap
        last = max(keys.values()) * gap
        start = start + last + gap + beat
    if order == "reverse" and words == "sequence":
        top = max(rank.values()) if rank else 0.0
        rank = {k: top - v for k, v in rank.items()}         # last letter first (un-typing)
    elif order == "reverse":
        for g in range(ng):                                   # words together: right to left
            ws = [li for li in L if li.gi == g]
            if ws:
                top = max(rank[li.name] for li in ws)
                for li in ws:
                    rank[li.name] = top - rank[li.name]
    total = max(rank.values()) if rank else 0.0
    if total > cap > 0.0:
        sc = cap / total
        rank = {k: v * sc for k, v in rank.items()}
    return rank


def entry_dur(want, off, lead=ENTRY_LEAD, end=F_ENTRY_END):
    """Duration of each letter's entry so the LAST letter of the cascade still lands on `end`."""
    room = end - (F_START + lead) - max(off.values() or [0.0])
    return max(6.0, min(float(want), room))


def exit_dur(off, start=F_EXIT_START, end=F_END):
    """Duration of each letter's exit so the last letter is gone exactly on `end`."""
    return max(6.0, (end - start) - max(off.values() or [0.0]))


def word_starts(L, off):
    """First / last start offset per word -> ([first...], [last...])."""
    ng = word_count(L)
    first, last = [1e9] * ng, [-1e9] * ng
    for li in L:
        o = off[li.name]
        first[li.gi] = min(first[li.gi], o)
        last[li.gi] = max(last[li.gi], o)
    return first, last


def bake_words(f0, f1, fn):
    """Animate the word parent bones (MASSIVE_ROOT, WIN_ROOT ...): fn(f, gi) -> (loc, rot, scl).
    Rigs without word parents simply skip this word-level layer."""
    for gi, rb in enumerate(CTX["wroots"]):
        if rb is not None:
            bake(rb, f0, f1, lambda f, gi=gi: fn(f, gi))


def X():
    """Exaggeration multiplier from the panel's Intensity slider (1.0 = as designed)."""
    return CTX.get("exag", 1.0)


# ============================================================
# SHARED BUILDING BLOCKS FOR THE VARIATIONS
# ============================================================

EXIT_MIN_DUR = 9.0      # every letter gets at least this many frames to leave
ENTRY_MIN_DUR = 10.0


def entry_cascade(L, order="reading", gap=ENTRY_GAP, beat=ENTRY_WORD_BEAT, cap=ENTRY_STAGGER_MAX,
                  min_dur=ENTRY_MIN_DUR, lead=ENTRY_LEAD, words="sequence", rng=None):
    st = max(1e-3, CTX.get("stagger", 1.0))
    room = (F_ENTRY_END - F_START - lead - min_dur) / st
    return cascade(L, gap, beat, max(0.0, min(cap, room)), order, words, rng)


def exit_cascade(L, order="reading", gap=EXIT_GAP, beat=EXIT_WORD_BEAT, cap=EXIT_STAGGER_MAX,
                 words="sequence", rng=None):
    st = max(1e-3, CTX.get("stagger", 1.0))
    room = (F_END - F_EXIT_START - EXIT_MIN_DUR) / st
    return cascade(L, gap, beat, max(0.0, min(cap, room)), order, words, rng)


def appear(f, s0, n=3.0):
    """Hidden until s0, then grows to full size in n frames (fast start, soft landing)."""
    return pw(f, (s0, HIDDEN), (s0 + n, 1.0, out_cubic))


def vanish(f, a, b, ease=in_quad):
    return pw(f, (a, 1.0), (b, HIDDEN, ease))


def master_push(master, start=0.95):
    """Moving hold for the whole entry: the text block slowly settles into its size, so the
    picture never freezes while the last letters land."""
    bake(master, F_START, F_ENTRY_END,
         lambda f: (None, None, SCL(lerp(start, 1.0, out_cubic(seg(f, F_START, F_ENTRY_END))))))


def word_kicks_entry(L, off, D, frac=0.35, amount=0.06, dur=10.0, roll=0.0, up=0.0):
    """Word-level accent: each word parent gets a punch when its LAST letter lands."""
    _first, last = word_starts(L, off)
    x = X()

    def fn(f, gi):
        tw = F_START + ENTRY_LEAD + last[gi] + frac * D
        d = max(3.0, min(dur, F_ENTRY_END - tw))
        k = kick(seg(f, tw, tw + d))
        return (LOC(up=up * x * k) if up else None,
                ROT(roll=roll * x * k) if roll else None,
                SCL(1.0 + amount * x * k))
    bake_words(F_START, F_ENTRY_END, fn)


def letter_seed(li, salt=0):
    return random.Random(li.i * 101 + salt * 7 + 3)


# ============================================================
# V1 - ELASTIC POP
# ============================================================

def v1_elastic_pop(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: pop out of the letter centre on a spring, rise, the tilt drags 2 frames behind ----
    off = entry_cascade(L)
    D = entry_dur(16, off)
    master_push(master, 0.95)
    word_kicks_entry(L, off, D, frac=0.3, amount=0.06)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        tilt0 = 22.0 * x * (li.sd or li.a)

        def rise(f, s0=s0):
            return -0.35 * (1.0 - spring(seg(f, s0, s0 + D), 0.16 * x, 0.36))

        def entry(f, s0=s0, rise=rise, tilt0=tilt0):
            pop = spring(seg(f, s0, s0 + D), 0.28 * x, 0.30)
            tilt = 1.0 - spring(seg(f, s0 + 2.0, s0 + D), 0.40 * x, 0.30)
            sc = SQ(HIDDEN + (1.0 - HIDDEN) * pop, stretch_of(ddf(rise, f), 1.6 * x))
            rot = ROT(roll=tilt0 * tilt)
            return vadd(LOC(up=rise(f)), pivot(rot, sc)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: lub-dub heartbeat, rippling out from the middle of each word ----
    beats = [(F_LOOP_START + 7, 1.0), (F_LOOP_START + 13, 0.55),
             (F_LOOP_START + 34, 1.0), (F_LOOP_START + 40, 0.55)]

    def heart(f, delay=0.0):
        return env(f) * sum(a * bump(f, c + delay, 2.0, 7.0) for c, a in beats)
    bake(master, F_LOOP_START, F_IDLE_END, lambda f: (None, None, SCL(1.0 + 0.045 * x * heart(f))))
    for li in L:
        d = 1.1 * abs(li.k - (li.m - 1) / 2.0)

        def idle(f, li=li, d=d):
            p = heart(f, d)
            br = lwave(f, 2, li.k * 0.6)
            sc = SQ(1.0 + 0.05 * x * p, 1.0 + 0.05 * x * p)
            rot = ROT(roll=2.0 * x * (li.sd or li.a) * (0.5 * p + 0.5 * br))
            return vadd(LOC(up=0.03 * x * p + 0.015 * x * br), pivot(rot, sc)), rot, sc
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: crouch (anticipation), pop bigger, then shrink away while drifting up ----
    off = exit_cascade(L, "reverse")
    DX = exit_dur(off)
    bake(master, F_EXIT_START, F_END, lambda f: (None, None, SCL(pw(
        f, (86, 1.0), (89, 0.97, io_sine), (91, 1.04, out_quad), (100, 0.9, in_quad)))))
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        a = ANTIC
        side = li.sd or li.a

        def ex(f, x0=x0, a=a, side=side):
            squash = pw(f, (x0, 1.0), (x0 + a, 1.0 - 0.16 * x, io_sine), (x0 + a + 2, 1.0 + 0.10 * x, out_quad),
                        (x0 + a + 5, 1.0))
            s = pw(f, (x0 + a, 1.0), (x0 + a + 2, 1.0 + 0.22 * x, out_cubic), (x0 + DX, HIDDEN, back_in(1.6)))
            up = pw(f, (x0, 0.0), (x0 + a, -0.04, io_sine), (x0 + DX, 0.35, in_quad))
            rot = ROT(roll=pw(f, (x0 + a, 0.0), (x0 + DX, -24.0 * x * side, in_quad)))
            sc = SQ(s, squash)
            return vadd(LOC(up=up), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V2 - BOUNCE DROP
# ============================================================

def v2_bounce_drop(master, letters):
    x, L = X(), letter_infos(letters)
    DROP, REST = 2.2, 0.28
    # ---- ENTRY: real ball physics - free fall, two bounces, squash on every contact ----
    off = entry_cascade(L)
    D = entry_dur(17, off)
    DB = 0.86 * D                                   # bounces done here, the rest is the last squash
    _h, impacts = bounce_drop(0.5, 2, REST)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        hits = [s0 + t * DB for t in impacts]

        def height(f, s0=s0):
            return DROP * bounce_drop(seg(f, s0, s0 + DB), 2, REST)[0]

        def entry(f, s0=s0, hits=hits, height=height, li=li):
            up = height(f)
            st = stretch_of(ddf(height, f), 0.28 * x, hi=1.35)          # stretched while falling fast
            for j, (fi, amt) in enumerate(zip(hits, (0.32, 0.16, 0.07))):
                st *= 1.0 - amt * x * bump(f, fi, 1.2, 3.5 if j < 2 else 2.5)
            tilt = 14.0 * x * li.a * (1.0 - smooth(seg(f, s0, hits[0])))
            tilt -= 7.0 * x * li.a * kick(seg(f, hits[0], hits[0] + 9))    # recoil after the 1st hit
            return LOC(up=up), ROT(roll=tilt), SQ(appear(f, s0, 3), st)
        bake(li.b, F_START, F_ENTRY_END, entry)
    first, last = word_starts(L, off)

    def floor(f, gi):                               # the word "floor" gives under the first impacts
        t = F_START + ENTRY_LEAD + (first[gi] + last[gi]) / 2.0 + impacts[0] * DB
        return LOC(up=-0.05 * x * kick(seg(f, t, min(F_ENTRY_END, t + 9)))), None, None
    bake_words(F_START, F_ENTRY_END, floor)

    # ---- LOOP: stadium wave - crouch, hop, land; left to right, then right to left ----
    n = len(L)
    g = min(1.2, 14.0 / max(1, n - 1))
    HJ = 8.0

    def hop_event(f, c):
        air = seg(f, c + 1, c + 1 + HJ)
        up = 0.16 * x * hop(air)
        vy = 0.16 * x * 4.0 * (1.0 - 2.0 * air) / HJ if 0.0 < air < 1.0 else 0.0
        st = stretch_of(vy, 1.4 * x, hi=1.15)
        st *= 1.0 - 0.13 * x * bump(f, c, 3.0, 2.0)            # crouch before take-off
        st *= 1.0 - 0.12 * x * bump(f, c + 1 + HJ, 1.5, 3.0)   # land
        return up, st
    for li in L:
        c1 = F_LOOP_START + 4 + g * li.i
        c2 = F_LOOP_START + 28 + g * (n - 1 - li.i)

        def idle(f, c1=c1, c2=c2):
            u1, s1 = hop_event(f, c1)
            u2, s2 = hop_event(f, c2)
            return LOC(up=u1 + u2), None, SQ(1.0, s1 * s2)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: crouch, little hop (anticipation), then gravity takes them down ----
    off = exit_cascade(L)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        a = ANTIC

        def upf(f, x0=x0, a=a):
            return pw(f, (x0, 0.0), (x0 + a, -0.05, io_sine), (x0 + a + 3, 0.18 * x, out_quad), (x0 + DX, -2.4, in_quad))

        def ex(f, x0=x0, a=a, upf=upf, li=li):
            crouch = pw(f, (x0, 1.0), (x0 + a, 1.0 - 0.18 * x, io_sine), (x0 + a + 2, 1.0, out_quad))
            st = crouch * stretch_of(ddf(upf, f), 0.12 * x, hi=1.4)
            s = vanish(f, x0 + 0.6 * DX, x0 + DX)
            return LOC(up=upf(f)), ROT(roll=pw(f, (x0 + a + 2, 0.0), (x0 + DX, 32.0 * x * li.a, in_quad))), SQ(s, st)
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V3 - SPIN FLIP
# ============================================================

def v3_spin_flip(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: 360 degree coin spin that overshoots and swings back (follow-through) ----
    off = entry_cascade(L)
    D = entry_dur(16, off)
    master_push(master, 0.9)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        sgn = li.sd or li.a

        def entry(f, s0=s0, sgn=sgn):
            u = seg(f, s0, s0 + D)
            yaw = 360.0 * sgn * (1.0 - spring(u, 0.06 * x, 0.45))
            sc = SCL(HIDDEN + (1.0 - HIDDEN) * spring(seg(f, s0, s0 + 0.7 * D), 0.18 * x, 0.35))
            rot = ROT(yaw=yaw)
            return vadd(LOC(up=-0.25 * (1.0 - out_cubic(u))), pivot(None, sc)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: a flip wave runs left to right, later another flips them back right to left ----
    n = len(L)
    g = min(1.5, 12.0 / max(1, n - 1))
    FD = 9.0
    for li in L:
        c1 = F_LOOP_START + 4 + g * li.i
        c2 = F_LOOP_START + 28 + g * (n - 1 - li.i)
        sgn = li.sd or li.a

        def idle(f, c1=c1, c2=c2, li=li, sgn=sgn):
            fl = io_cubic(seg(f, c1, c1 + FD)) - io_cubic(seg(f, c2, c2 + FD))
            yaw = 360.0 * sgn * fl + 8.0 * x * lwave(f, 2, li.i * 0.5)
            up = 0.07 * x * (hop(seg(f, c1, c1 + FD)) + hop(seg(f, c2, c2 + FD)))
            return LOC(up=up), ROT(yaw=yaw, roll=1.5 * x * lwave(f, 2, li.i * 0.5 + 1.0)), None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: wind up the other way (anticipation), then spin out and up, accelerating ----
    off = exit_cascade(L)
    DX = exit_dur(off)
    bake(master, F_EXIT_START, F_END, lambda f: (None, None, SCL(pw(f, (86, 1.0), (100, 0.8, in_quad)))))
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        sgn = li.sd or li.a

        def ex(f, x0=x0, sgn=sgn):
            yaw = pw(f, (x0, 0.0), (x0 + ANTIC, -25.0 * x * sgn, io_sine), (x0 + DX, 400.0 * sgn, in_cubic))
            sc = SCL(pw(f, (x0 + ANTIC, 1.0), (x0 + DX, HIDDEN, back_in(1.4))))
            up = pw(f, (x0, 0.0), (x0 + ANTIC, -0.04, io_sine), (x0 + DX, 0.45, in_quad))
            return vadd(LOC(up=up), pivot(None, sc)), ROT(yaw=yaw), sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V4 - SLAM SHAKE
# ============================================================

def v4_slam_shake(master, letters):
    x, L = X(), letter_infos(letters)
    IMPACT = 11.0
    # ---- ENTRY: the whole text slams in from huge, camera shake, shockwave through the letters ----
    def shake(f):
        if f <= IMPACT:
            return 0.0
        return math.exp(-0.22 * (f - IMPACT)) * (1.0 - smoother(seg(f, IMPACT, F_ENTRY_END - 1)))

    def m_entry(f):
        s = pw(f, (F_START, 2.8), (IMPACT, 1.0, in_quart))
        st = 1.0 - 0.24 * x * kick(seg(f, IMPACT, IMPACT + 13), 1.25, 4.0)
        e, t = shake(f), f - IMPACT
        loc = LOC(up=pw(f, (F_START, 0.3), (IMPACT, 0.0, in_quart)) + 0.10 * x * e * math.sin(t * 3.1 + 1.0),
                  side=0.16 * x * e * math.sin(t * 2.4))
        return loc, ROT(roll=3.5 * x * e * math.sin(t * 2.0 + 0.5)), SQ(s, st)
    bake(master, F_START, F_ENTRY_END, m_entry)
    off = entry_cascade(L, gap=0.6, beat=0.5, cap=3.0)
    for li in L:
        s0 = F_START + off[li.name]
        ti = IMPACT + 1.0 + 0.7 * abs(li.dx)          # the shockwave reaches the outer letters later

        def entry(f, s0=s0, ti=ti, li=li):
            jump = hop(seg(f, ti, ti + 7))
            st = (1.0 + 0.10 * x * jump) * (1.0 - 0.14 * x * bump(f, ti + 7, 1.2, 3.0))
            rot = ROT(roll=7.0 * x * (li.sd or li.a) * kick(seg(f, ti, min(F_ENTRY_END, ti + 13))))
            sc = SQ(appear(f, s0, 3), st)
            return vadd(LOC(up=0.10 * x * jump), pivot(rot, None)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: power surges with a tremble that never quite stops ----
    surges = [F_LOOP_START + 6 + 12 * k for k in range(4)]

    def surge(f, d=0.0):
        return sum(bump(f, c + d, 2.0, 8.0) for c in surges)

    def m_idle(f):
        s = surge(f)
        return (LOC(side=0.012 * x * s * noise1(f * 1.7, 1.0), up=0.008 * x * s * noise1(f * 1.9, 2.0)),
                ROT(roll=0.8 * x * s * noise1(f * 2.1, 3.0)), SCL(1.0 + 0.03 * x * s))
    bake(master, F_LOOP_START, F_IDLE_END, m_idle)
    for li in L:
        def idle(f, li=li):
            e = env(f)
            s = surge(f, 0.5 * abs(li.dx))
            rot = ROT(roll=1.2 * x * e * noise1(f * 2.9, li.seed * 9 + 3))
            sc = SCL(1.0 + 0.02 * x * s)
            return vadd(LOC(up=0.008 * x * e * noise1(f * 2.3, li.seed * 9)), pivot(rot, sc)), rot, sc
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: squash down (anticipation) then zoom through the camera, letters fan apart ----
    XA = F_EXIT_START + ANTIC + 1
    bake(master, F_EXIT_START, F_END, lambda f: (
        LOC(up=pw(f, (86, 0.0), (XA, -0.05, io_sine), (100, 0.25, in_quad))), None,
        SCL(pw(f, (86, 1.0), (XA, 0.86, io_sine), (100, 3.6, in_expo)))))
    for li in L:
        rng = letter_seed(li, 4)
        spin = (li.sd or li.a) * rng.uniform(25.0, 60.0)
        lift = rng.uniform(-0.4, 0.4)

        def ex(f, li=li, spin=spin, lift=lift):
            u = in_cubic(seg(f, XA, F_END))
            rot = ROT(roll=spin * x * u)
            sc = SCL(vanish(f, 95, F_END))
            return vadd(LOC(side=0.6 * li.dx * u, up=lift * u), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V5 - PENDULUM SWING  (roly-poly: the letters rock on their feet)
# ============================================================

def v5_pendulum_swing(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: drop in tipped over, land, rock on the foot until the swing dies out ----
    off = entry_cascade(L)
    D = entry_dur(19, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        land = s0 + 0.3 * D
        sgn = li.sd or li.a

        def entry(f, s0=s0, land=land, sgn=sgn):
            up = 1.4 * (1.0 - in_quad(seg(f, s0, land)))
            ang = 42.0 * x * sgn * settle(seg(f, land, s0 + D), 2.0, 3.0)
            ang = pw(f, (s0, 60.0 * sgn), (land, ang, out_quad)) if f < land else ang
            st = (1.0 + 0.2 * x * seg(f, s0, land) * (1.0 if f < land else 0.0)) * \
                (1.0 - 0.2 * x * bump(f, land, 1.0, 3.0))
            side = -math.radians(ang) * 0.22                 # the rounded foot rolls a little
            return LOC(up=up, side=side), ROT(roll=ang), SQ(appear(f, s0, 3), st)
        bake(li.b, F_START, F_ENTRY_END, entry)
    master_push(master, 0.97)

    # ---- LOOP: metronome sway, every letter a little behind its left neighbour ----
    bake(master, F_LOOP_START, F_IDLE_END, lambda f: (None, ROT(roll=1.2 * x * lwave(f, 1)), None))
    for li in L:
        ph = li.i * 0.35

        def idle(f, ph=ph):
            ang = 9.0 * x * lwave(f, 2, ph)
            return (LOC(side=-math.radians(ang) * 0.22, up=0.01 * x * lwave2(f, 4, 2 * ph)),
                    ROT(roll=ang, pitch=2.0 * x * lwave(f, 2, ph + math.pi / 2)), None)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: rock back (anticipation), then tip over and fall ----
    off = exit_cascade(L, "reverse")
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        sgn = li.sd or li.a

        def ex(f, x0=x0, sgn=sgn):
            ang = pw(f, (x0, 0.0), (x0 + ANTIC, -12.0 * x * sgn, io_sine), (x0 + DX, 85.0 * sgn, in_quad))
            up = pw(f, (x0 + ANTIC + 2, 0.0), (x0 + DX, -1.6, in_quad))
            return LOC(up=up), ROT(roll=ang), SCL(vanish(f, x0 + 0.55 * DX, x0 + DX))
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V6 - ROCKET RISE
# ============================================================

def v6_rocket_rise(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: launch from below, stretched by the speed, overshoot, thrust cut, settle ----
    off = entry_cascade(L)
    D = entry_dur(16, off)
    bake(master, F_START, F_ENTRY_END,
         lambda f: (LOC(up=-0.08 * (1.0 - out_cubic(seg(f, F_START, F_ENTRY_END)))), None, None))
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def upf(f, s0=s0):
            return -1.8 * (1.0 - spring(seg(f, s0, s0 + D), 0.10 * x, 0.42))

        def entry(f, s0=s0, upf=upf, li=li):
            u = seg(f, s0, s0 + D)
            w = kick(u, 2.0, 2.5)                       # rocket instability, dies out
            st = stretch_of(ddf(upf, f), 0.30 * x, hi=1.55)
            return (LOC(up=upf(f), side=0.05 * x * li.a * w), ROT(roll=-6.0 * x * li.a * w),
                    SQ(appear(f, s0, 3), st))
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: hover - bob with a figure-eight tilt, a wave across the letters ----
    for li in L:
        ph = li.i * 0.5

        def idle(f, ph=ph, li=li):
            return (LOC(up=0.07 * x * lwave(f, 2, ph)),
                    ROT(pitch=3.0 * x * lwave(f, 2, ph + math.pi / 2), roll=2.0 * x * li.a * lwave(f, 2, ph + math.pi)),
                    SQ(1.0, 1.0 + 0.03 * x * lwave(f, 2, ph - 0.6)))
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: crouch (anticipation), then blast off - stretched, accelerating ----
    off = exit_cascade(L)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        a = ANTIC + 1

        def upf(f, x0=x0, a=a):
            return pw(f, (x0, 0.0), (x0 + a, -0.10, io_sine), (x0 + DX, 3.6, in_expo))

        def ex(f, x0=x0, a=a, upf=upf, li=li):
            crouch = pw(f, (x0, 1.0), (x0 + a, 1.0 - 0.2 * x, io_sine), (x0 + a + 1.5, 1.0, out_quad))
            st = crouch * stretch_of(ddf(upf, f), 0.22 * x, hi=1.8)
            return (LOC(up=upf(f)), ROT(roll=pw(f, (x0 + a, 0.0), (x0 + DX, 4.0 * li.a, in_quad))),
                    SQ(vanish(f, x0 + 0.75 * DX, x0 + DX), st))
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V7 - JELLY WOBBLE
# ============================================================

def v7_jelly_wobble(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: plop down, splat, then jiggle like jelly (volume stays the same) ----
    off = entry_cascade(L)
    D = entry_dur(18, off)
    master_push(master, 0.94)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        land = s0 + 0.28 * D

        def entry(f, s0=s0, land=land, li=li):
            fall = seg(f, s0, land)
            up = 0.7 * (1.0 - in_quad(fall))
            if f < land:
                st = 1.0 + 0.25 * x * fall
            else:
                st = 1.0 - 0.55 * x * settle(seg(f, land, s0 + D), 2.5, 3.0)
            rot = ROT(roll=6.0 * x * li.a * kick(seg(f, land, s0 + D), 1.5, 3.0))
            return LOC(up=up), rot, SQ(appear(f, s0, 2.5), st)
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: squishy wobble travelling through the word ----
    for li in L:
        ph = li.i * 0.6

        def idle(f, ph=ph):
            j = lwave(f, 3, ph)
            return (LOC(up=0.02 * x * lwave(f, 3, ph - 0.8)), ROT(roll=2.0 * x * lwave(f, 3, ph + 1.2)),
                    SQ(1.0, 1.0 + 0.09 * x * j))
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: big squash (anticipation), spring up stretched, shrink away ----
    off = exit_cascade(L)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        a = ANTIC

        def ex(f, x0=x0, a=a, li=li):
            st = pw(f, (x0, 1.0), (x0 + a, 1.0 - 0.42 * x, io_sine), (x0 + a + 2, 1.0 + 0.5 * x, out_quad),
                    (x0 + DX, 1.15, smooth))
            up = pw(f, (x0 + a, 0.0), (x0 + DX, 0.8, out_quad))
            rot = ROT(roll=pw(f, (x0 + a, 0.0), (x0 + DX, -10.0 * x * li.a)))
            return LOC(up=up), rot, SQ(pw(f, (x0 + 0.4 * DX, 1.0), (x0 + DX, HIDDEN, back_in(1.5))), st)
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V8 - SPIN ZOOM
# ============================================================

def v8_spin_zoom(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: the block zooms in on a 540 degree spin; the letters DRAG behind the spin ----
    def m_ang(f):
        return 540.0 * (1.0 - out_cubic(seg(f, F_START, 22))) - 9.0 * x * kick(seg(f, 20, F_ENTRY_END))

    def m_entry(f):
        s = HIDDEN + (1.0 - HIDDEN) * spring(seg(f, F_START, 24), 0.12 * x, 0.42)
        return LOC(up=-0.4 * (1.0 - out_quad(seg(f, F_START, 22)))), ROT(roll=m_ang(f)), SCL(s)
    bake(master, F_START, F_ENTRY_END, m_entry)
    off = entry_cascade(L, gap=0.8, cap=5.0)
    for li in L:
        s0 = F_START + 2.0 + off[li.name]
        lag = 1.5 + 0.4 * abs(li.dx)

        def entry(f, s0=s0, lag=lag):
            drag = max(-25.0, min(25.0, -0.16 * x * ddf(m_ang, f - lag)))
            drag *= 1.0 - smoother(seg(f, 25, F_ENTRY_END))
            sc = SCL(HIDDEN + (1.0 - HIDDEN) * spring(seg(f, s0, s0 + 12), 0.2 * x, 0.35))
            rot = ROT(roll=drag)
            return pivot(rot, sc), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: a glint sweeps across the letters twice, the block sways ----
    bake(master, F_LOOP_START, F_IDLE_END,
         lambda f: (None, ROT(roll=1.5 * x * lwave(f, 1)), SCL(1.0 + 0.015 * x * lwave2(f, 2))))
    n = len(L)
    g = min(1.3, 13.0 / max(1, n - 1))
    for li in L:
        cs = (F_LOOP_START + 6 + g * li.i, F_LOOP_START + 32 + g * li.i)

        def idle(f, cs=cs, li=li):
            p = sum(bump(f, c, 2.0, 5.0) for c in cs)
            sc = SCL(1.0 + 0.14 * x * p)
            rot = ROT(roll=4.0 * x * li.a * p)
            return vadd(LOC(up=0.05 * x * p), pivot(rot, sc)), rot, sc
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: counter-turn (anticipation), then spin out and away ----
    XA = F_EXIT_START + ANTIC

    def mx_ang(f):
        return pw(f, (86, 0.0), (XA, 18.0 * x, io_sine), (100, -540.0, in_cubic))

    bake(master, F_EXIT_START, F_END, lambda f: (
        LOC(up=pw(f, (XA, 0.0), (100, 0.4, in_quad))), ROT(roll=mx_ang(f)),
        SCL(pw(f, (86, 1.0), (XA, 1.05, io_sine), (100, HIDDEN, back_in(1.3))))))
    for li in L:
        lag = 1.0 + 0.3 * abs(li.dx)

        def ex(f, lag=lag):
            drag = max(-25.0, min(25.0, -0.12 * x * ddf(mx_ang, f - lag)))
            sc = SCL(vanish(f, 94, F_END))
            rot = ROT(roll=drag)
            return pivot(rot, sc), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V9 - SCATTER ASSEMBLE
# ============================================================

def _arc(p0, t, bend):
    """Point on a curved path from p0 (side, up) to (0, 0): quadratic Bezier whose control
    point is pushed sideways by `bend` - so flights travel on ARCS, not straight lines.
    t may go above 1 (springy overshoot continues along the arrival direction)."""
    cx = p0[0] * 0.5 - p0[1] * bend
    cz = p0[1] * 0.5 + p0[0] * bend
    a, b = (1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t)
    return a * p0[0] + b * cx, a * p0[1] + b * cz


def v9_scatter_assemble(master, letters):
    x, L = X(), letter_infos(letters)
    rng = random.Random(21)
    # ---- ENTRY: letters fly in from all around on curved paths, spin settles after landing ----
    off = entry_cascade(L, "random", gap=1.1, rng=random.Random(9))
    D = entry_dur(16, off)
    lands = []
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        ang, r = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(1.6, 2.6)
        p0 = (r * math.cos(ang), r * math.sin(ang))
        bend = rng.choice((-1.0, 1.0)) * rng.uniform(0.25, 0.45)
        spun = (rng.uniform(-60, 60), rng.uniform(-200, 200))
        lands.append(s0 + 0.5 * D)

        def entry(f, s0=s0, p0=p0, bend=bend, spun=spun):
            t = spring(seg(f, s0, s0 + D), 0.07 * x, 0.45)
            px, pz = _arc(p0, t, bend)
            r_ = 1.0 - spring(seg(f, s0 + 2.0, s0 + D), 0.22 * x, 0.32)
            rot = ROT(pitch=spun[0] * r_, roll=spun[1] * r_)
            sc = SCL(HIDDEN + (1.0 - HIDDEN) * spring(seg(f, s0, s0 + 0.55 * D), 0.15 * x, 0.35))
            return vadd(LOC(side=px, up=pz), pivot(rot, sc)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)
    t_all = max(lands)
    bake(master, F_START, F_ENTRY_END, lambda f: (
        None, None, SCL(1.0 + 0.05 * x * kick(seg(f, t_all, F_ENTRY_END)))))

    # ---- LOOP: magnet orbit - every letter circles its spot ----
    for li in L:
        ph = rng.uniform(0.0, 2.0 * math.pi)

        def idle(f, ph=ph, li=li):
            e, w = env(f), 2.0 * math.pi * 2.0 * loop_u(f) + ph
            rot = ROT(roll=2.5 * x * li.a * e * math.sin(w + 1.0))
            return (vadd(LOC(side=0.035 * x * e * math.cos(w), up=0.035 * x * e * math.sin(w)), pivot(rot, None)),
                    rot, None)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: pull in (anticipation), then burst outward on arcs ----
    off = exit_cascade(L, "random", rng=random.Random(3))
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        ang = math.atan2(li.rz + 0.3, li.rx + 1e-6) + rng.uniform(-0.6, 0.6)
        r = rng.uniform(2.0, 3.0)
        p1 = (r * math.cos(ang), r * math.sin(ang))
        bend = rng.choice((-1.0, 1.0)) * 0.3
        spin = (rng.uniform(-70, 70), rng.uniform(-270, 270))

        def ex(f, x0=x0, p1=p1, bend=bend, spin=spin, li=li):
            pull = smooth(seg(f, x0, x0 + ANTIC)) * (1.0 - seg(f, x0 + ANTIC, x0 + ANTIC + 2))
            t = in_cubic(seg(f, x0 + ANTIC, x0 + DX))
            px, pz = _arc(p1, 1.0 - t, bend)
            u = in_quad(seg(f, x0 + ANTIC, x0 + DX))
            rot = ROT(pitch=spin[0] * u, roll=spin[1] * u)
            sc = SQ(vanish(f, x0 + 0.6 * DX, x0 + DX), 1.0 - 0.12 * x * pull)
            return (vadd(LOC(side=px - 0.08 * x * li.dx * pull, up=pz), pivot(rot, sc)), rot, sc)
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V10 - GLITCH FLICKER  (stepped keys on purpose: digital, no in-betweens)
# ============================================================

def _glitch_state(seed, f, strength):
    r = random.Random(int(seed * 100003) + int(f) * 7919)
    return (LOC(side=r.uniform(-1, 1) * 0.22 * strength, up=r.uniform(-1, 1) * 0.06 * strength),
            ROT(roll=r.uniform(-1, 1) * 6.0 * strength),
            SCL(1.0, up=1.0 + r.uniform(-0.25, 0.25) * strength, side=1.0 + r.uniform(-0.2, 0.6) * strength))


def v10_glitch_flicker(master, letters):
    x, L = X(), letter_infos(letters)
    OFF = (None, None, SCL(HIDDEN))
    FL = 23                                           # last flicker frame of the entry
    # ---- ENTRY: letters flicker on (on twos), chaos fades, then a clean lock-in punch ----
    def m_glitch(f):
        amp = x * (1.0 - f / float(FL))
        r = random.Random(int(f) // 2 * 31 + 1)
        return (LOC(side=r.uniform(-1, 1) * 0.08 * amp, up=r.uniform(-1, 1) * 0.03 * amp),
                ROT(roll=r.uniform(-1, 1) * 1.5 * amp), SCL(1.0 + r.uniform(-0.04, 0.06) * amp))
    bake(master, F_START, FL, m_glitch, style=STEP)
    bake(master, FL + 1, F_ENTRY_END, lambda f: (None, None, SCL(1.0 + 0.07 * x * kick(seg(f, FL + 1, F_ENTRY_END)))))
    off = entry_cascade(L, gap=1.0, cap=8.0)
    for li in L:
        f0 = F_START + off[li.name]

        def entry(f, f0=f0, li=li):
            if f >= FL - 1:
                return None, None, None                 # every letter is on for the lock-in
            if f < f0:
                return OFF
            ff = f0 + 2 * ((f - f0) // 2)               # hold every state for 2 frames
            prog = (ff - f0) / float(max(1.0, FL - f0))
            r = random.Random(li.i * 977 + int(ff))
            if r.random() < 0.25 + 0.65 * prog:
                return _glitch_state(li.seed + 0.1, ff, x * (1.0 - prog))
            return OFF
        bake(li.b, F_START, FL, entry, style=STEP)
        bake(li.b, FL + 1, F_ENTRY_END, lambda f: (None, None, None))

    # ---- LOOP: two short glitch bursts ----
    bursts = (F_LOOP_START + 17, F_LOOP_START + 38)

    def in_burst(f):
        return any(c <= f < c + 4 for c in bursts)
    bake(master, F_LOOP_START, F_IDLE_END, lambda f: (
        _glitch_state(0.77, f, 0.25 * x) if in_burst(f) else (None, None, None)), style=STEP)
    for li in L:
        r = random.Random(li.i * 13 + 5)
        hits = [c for c in bursts if r.random() < 0.6]

        def idle(f, li=li, hits=hits):
            if any(c <= f < c + 4 for c in hits):
                rr = random.Random(li.i * 31 + int(f))
                return OFF if rr.random() < 0.15 else _glitch_state(li.seed, f, 0.7 * x)
            return None, None, None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle, style=STEP)

    # ---- EXIT: corruption grows, letters drop out, then a CRT power-off squeezes the rest ----
    CRT = 95

    def m_exit(f):
        if f < CRT:
            amp = x * (f - F_EXIT_START) / float(CRT - F_EXIT_START)
            r = random.Random(int(f) // 2 * 17 + 9)
            return (LOC(side=r.uniform(-1, 1) * 0.10 * amp, up=r.uniform(-1, 1) * 0.04 * amp),
                    ROT(roll=r.uniform(-1, 1) * 2.0 * amp), None)
        up = pw(f, (CRT, 1.0), (98, 0.03, in_cubic))
        side = pw(f, (CRT, 1.0), (98, 1.25, out_quad), (F_END, HIDDEN, in_cubic))
        return None, None, SCL(1.0, up=up, side=side)
    bake(master, F_EXIT_START, CRT - 1, m_exit, style=STEP)
    bake(master, CRT, F_END, m_exit)
    for li in L:
        r = random.Random(li.i * 53 + 11)
        drop = F_EXIT_START + 3 + r.uniform(0, 8) if r.random() < 0.5 else 999

        def ex(f, li=li, drop=drop):
            if f >= F_END:
                return OFF
            if f >= drop:
                return OFF
            ff = int(f) // 2 * 2
            prog = (ff - F_EXIT_START) / float(CRT - F_EXIT_START)
            rr = random.Random(li.i * 71 + ff)
            if f < CRT and rr.random() < 0.15 + 0.5 * prog:
                return _glitch_state(li.seed + 0.3, ff, x * (0.3 + 0.7 * prog)) if rr.random() < 0.7 else OFF
            return None, None, None
        bake(li.b, F_EXIT_START, F_END, ex, style=STEP)


# ============================================================
# V11 - DOMINO FLIP
# ============================================================

def v11_domino_flip(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: stand up from flat in a chain, rock past upright and settle with a clack ----
    off = entry_cascade(L, gap=1.6, cap=10.0)
    D = entry_dur(16, off)
    master_push(master, 0.97)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0):
            u = seg(f, s0, s0 + D)
            pitch = -90.0 * (1.0 - spring(u, 0.10 * x, 0.42))
            st = 1.0 - 0.10 * x * bump(f, s0 + 0.42 * D, 1.0, 3.0)
            return LOC(up=-0.15 * (1.0 - out_cubic(u))), ROT(pitch=pitch), SQ(appear(f, s0, 2.5), st)
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: a domino ripple tips forward left to right, then back right to left ----
    n = len(L)
    g = min(1.3, 12.0 / max(1, n - 1))

    def tip(f, c):
        return pw(f, (c, 0.0), (c + 3, 1.0, out_quad), (c + 11, 0.0, lambda u: spring(u, 0.25, 0.35)))
    for li in L:
        c1 = F_LOOP_START + 2 + g * li.i
        c2 = F_LOOP_START + 28 + g * (n - 1 - li.i)

        def idle(f, c1=c1, c2=c2):
            t1, t2 = tip(f, c1), tip(f, c2)
            return (LOC(up=-0.03 * (abs(t1) + abs(t2))),
                    ROT(pitch=22.0 * x * t1 - 18.0 * x * t2, roll=3.0 * x * (t1 - t2)), None)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: rock back (anticipation), topple with gravity, small rebound, gone ----
    off = exit_cascade(L, gap=1.0)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        hit = x0 + ANTIC + 0.55 * (DX - ANTIC)

        def ex(f, x0=x0, hit=hit):
            pitch = pw(f, (x0, 0.0), (x0 + ANTIC, -8.0 * x, io_sine), (hit, 90.0, in_quad))
            if f > hit:
                pitch = 90.0 - 10.0 * x * kick(seg(f, hit, x0 + DX), 1.0, 3.0)
            return LOC(up=pw(f, (hit - 2, 0.0), (x0 + DX, -0.15))), ROT(pitch=pitch), \
                SCL(vanish(f, x0 + DX - 3, x0 + DX))
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V12 - ACCORDION SPREAD  (fans out around the middle of EACH word)
# ============================================================

def v12_accordion_spread(master, letters):
    x, L = X(), letter_infos(letters)
    # ---- ENTRY: packed in the word centre, the middle letters open first, edges follow ----
    off = entry_cascade(L, "center", gap=1.4)
    D = entry_dur(15, off)
    master_push(master, 0.93)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def sidef(f, s0=s0, li=li):
            return -li.dx * (1.0 - spring(seg(f, s0, s0 + D), 0.10 * x, 0.40))

        def entry(f, s0=s0, sidef=sidef, li=li):
            fan = 1.0 - spring(seg(f, s0 + 1.5, s0 + D), 0.30 * x, 0.30)
            st = stretch_of(ddf(sidef, f), 0.45 * x, hi=1.4)
            sc = SQX(HIDDEN + (1.0 - HIDDEN) * spring(seg(f, s0, s0 + 0.6 * D), 0.12 * x, 0.35), st)
            rot = ROT(roll=30.0 * x * li.s * fan)
            return vadd(LOC(side=sidef(f)), pivot(rot, sc)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: the bellows breathe in and out ----
    for li in L:
        def idle(f, li=li):
            w = lwave(f, 2)
            rot = ROT(roll=3.0 * x * li.s * w)
            sc = SCL(1.0 + 0.02 * x * w)
            return (vadd(LOC(side=0.07 * x * li.dx * w, up=0.02 * x * (1.0 - abs(li.s)) * lwave2(f, 2)),
                         pivot(rot, sc)), rot, sc)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: open a little more (anticipation), then fold back in, edges first ----
    off = exit_cascade(L, "edges")
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0, li=li):
            side = pw(f, (x0, 0.0), (x0 + ANTIC, 0.06 * x * li.dx, io_sine), (x0 + DX, -li.dx, in_cubic))
            rot = ROT(roll=pw(f, (x0 + ANTIC, 0.0), (x0 + DX, -30.0 * x * li.s, in_quad)))
            sc = SCL(vanish(f, x0 + 0.5 * DX, x0 + DX))
            return vadd(LOC(side=side), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V13 - REEL STOP  (slot reels: everything spins, the reels stop left to right)
# ============================================================

def v13_reel_stop(master, letters):
    x, L = X(), letter_infos(letters)
    P, V = 1.2, 0.35                       # reel window height, spin speed (units / frame)
    BR = 3.0 * (P / 2.0) / V               # brake frames (matches the spin speed - no speed jump)

    def window(y):
        """Reel window: fully visible near the letter's own spot, hidden at the window edges."""
        return max(HIDDEN, 1.0 - smooth((abs(y) - 0.28) / 0.3))

    off = entry_cascade(L, gap=1.6, beat=3.0, cap=11.0)
    D = entry_dur(14, off)
    stops = {}
    for li in L:
        ts = F_START + ENTRY_LEAD + off[li.name] + 0.5 * D
        stops[li.name] = ts
        tb = ts - BR

        def ypos(f, ts=ts, tb=tb):
            if f < tb:                                         # spinning: the strip repeats
                y = P / 2.0 + V * (tb - f)
                return ((y - P / 2.0) % P) - P / 2.0
            if f < ts:                                         # brake
                return (P / 2.0) * (1.0 - out_cubic(seg(f, tb, ts)))
            return -0.14 * x * kick(seg(f, ts, min(F_ENTRY_END, ts + 9)), 1.0, 3.5)   # reel bounce

        def entry(f, ts=ts, ypos=ypos):
            y = ypos(f)
            spd = V if f < ts - BR else ddf(ypos, f)
            st = stretch_of(spd, 1.7 * x, hi=1.6) * (1.0 - 0.12 * x * bump(f, ts + 1, 1.0, 3.0))
            return LOC(up=y), None, SQ(window(y), st)
        bake(li.b, F_START, F_ENTRY_END, entry)
    ng = word_count(L)
    wstop = [max([stops[li.name] for li in L if li.gi == g] or [F_START]) for g in range(ng)]
    bake_words(F_START, F_ENTRY_END, lambda f, gi: (
        None, None, SCL(1.0 + 0.06 * x * kick(seg(f, wstop[gi] + 1, min(F_ENTRY_END, wstop[gi] + 11))))))

    # ---- LOOP: reel nudges ripple across, twice ----
    n = len(L)
    g = min(1.4, 12.0 / max(1, n - 1))
    for li in L:
        cs = (F_LOOP_START + 4 + g * li.i, F_LOOP_START + 29 + g * li.i)

        def idle(f, cs=cs):
            k = sum(kick(seg(f, c, c + 10), 1.0, 3.5) for c in cs)
            glint = sum(bump(f, c + 2, 2.0, 4.0) for c in cs)
            return LOC(up=-0.07 * x * k), None, SQ(1.0 + 0.05 * x * glint, 1.0 + 0.06 * x * glint)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: pull up (anticipation), then the reels spin away downwards ----
    off = exit_cascade(L, gap=0.8)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ypos(f, x0=x0):
            return pw(f, (x0, 0.0), (x0 + ANTIC, 0.08 * x, io_sine), (x0 + DX, -P * 0.75, in_quad))

        def ex(f, ypos=ypos, x0=x0):
            y = ypos(f)
            s = window(y) if f < x0 + DX else HIDDEN
            return LOC(up=y), None, SQ(s, stretch_of(ddf(ypos, f), 1.7 * x, hi=1.6))
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V14 - STAMP SLAM  (typewriter / rubber stamp, one letter per beat)
# ============================================================

def v14_stamp_slam(master, letters):
    x, L = X(), letter_infos(letters)
    SL = 3.0                                    # frames from lift to impact
    off = entry_cascade(L, gap=1.8, beat=3.0, cap=13.0, min_dur=9.0)
    hits = []
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        hit = s0 + SL
        hits.append(hit)
        tilt = letter_seed(li, 1).uniform(-6.0, 6.0) * x

        def entry(f, s0=s0, hit=hit, tilt=tilt):
            if f < s0:
                return None, None, SCL(HIDDEN)
            if f < hit:                                   # coming down, accelerating
                u = in_quart(seg(f, s0, hit))
                sc = SCL(lerp(2.3, 1.0, u))
                rot = ROT(roll=tilt * (1.0 - u) + tilt * 0.5 * u)
                return vadd(LOC(depth=-0.4 * (1.0 - u), up=0.15 * (1.0 - u)), pivot(rot, sc)), rot, sc
            st = 1.0 - 0.22 * x * kick(seg(f, hit, min(F_ENTRY_END, hit + 8)), 1.0, 3.0)
            rot = ROT(roll=tilt * 0.5 * (1.0 - spring(seg(f, hit, min(F_ENTRY_END, hit + 10)), 0.3, 0.3)))
            return pivot(rot, None), rot, SQ(1.0, st)
        bake(li.b, F_START, F_ENTRY_END, entry)

    def m_entry(f):
        k = sum(kick(seg(f, h, min(F_ENTRY_END, h + 6)), 1.0, 4.0) for h in hits)
        return LOC(up=-0.012 * x * k), None, SCL(1.0 - 0.006 * x * k)
    bake(master, F_START, F_ENTRY_END, m_entry)

    # ---- LOOP: the typewriter re-inks the line: a press runs left to right, then back ----
    n = len(L)
    g = min(1.6, 14.0 / max(1, n - 1))
    for li in L:
        cs = (F_LOOP_START + 5 + g * li.i, F_LOOP_START + 30 + g * (n - 1 - li.i))

        def idle(f, cs=cs, li=li):
            p = sum(bump(f, c, 1.5, 4.0) for c in cs)
            rot = ROT(roll=1.5 * x * li.a * p)
            return pivot(rot, None), rot, SQ(1.0 - 0.04 * x * p, 1.0 - 0.06 * x * p)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: backspace - press (anticipation), lift off towards the camera, gone; right to left ----
    off = exit_cascade(L, "reverse", gap=0.9)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0):
            press = pw(f, (x0, 1.0), (x0 + 2, 1.0 - 0.12 * x, io_sine), (x0 + 3, 1.0, out_quad))
            s = pw(f, (x0 + 2, 1.0), (x0 + 5, 1.25, out_quad), (x0 + DX, HIDDEN, in_quad))
            sc = SQ(s, press)
            lift = seg(f, x0 + 2, x0 + DX)
            return vadd(LOC(depth=-0.4 * out_quad(lift), up=0.2 * in_quad(lift)), pivot(None, sc)), None, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V15 - FIREWORK BURST
# ============================================================

def v15_firework_burst(master, letters):
    x, L = X(), letter_infos(letters)
    ng = word_count(L)
    BEAT = 4.0 * CTX.get("stagger", 1.0)
    launch = [F_START + g * BEAT for g in range(ng)]
    burst = [t + 7.0 for t in launch]
    dmax = max([0.3 * abs(li.dx) for li in L] or [0.0])
    D = max(8.0, min(15.0, F_ENTRY_END - max(burst) - dmax))
    # ---- ENTRY: one rocket per word rises, bursts, the letters fly out on arcs to their spots ----
    for li in L:
        b0 = burst[li.gi]
        l0 = launch[li.gi]
        d = 0.3 * abs(li.dx)
        sgn = li.sd or li.a

        def entry(f, b0=b0, l0=l0, d=d, li=li, sgn=sgn):
            if f < l0:
                return None, None, SCL(HIDDEN)
            if f < b0 + d:                                    # the rocket (a small spark)
                u = seg(f, l0, b0)
                up = lerp(-1.9, 0.1, out_quad(u))
                return LOC(side=-li.dx, up=up), None, SQ(0.2, 1.6 - 0.6 * u)
            u = seg(f, b0 + d, b0 + d + D)
            t = spring(u, 0.08 * x, 0.40)
            lag = 1.0 - spring(seg(f, b0 + d + 1.5, b0 + d + D), 0.2 * x, 0.32)
            rot = ROT(roll=300.0 * sgn * lag)
            sc = SCL(0.2 + 0.8 * spring(seg(f, b0 + d, b0 + d + 0.7 * D), 0.3 * x, 0.3))
            return (vadd(LOC(side=-li.dx * (1.0 - t), up=0.1 * (1.0 - t) + 0.5 * x * hop(min(1.0, t))),
                         pivot(rot, sc)), rot, sc)
        bake(li.b, F_START, F_ENTRY_END, entry)
    bake(master, F_START, F_ENTRY_END, lambda f: (None, None, SCL(1.0 + 0.05 * x * sum(
        kick(seg(f, b, min(F_ENTRY_END, b + 10))) for b in burst))))

    # ---- LOOP: random twinkles (each letter sparkles twice) ----
    for li in L:
        r = letter_seed(li, 15)
        cs = (r.uniform(F_LOOP_START + 4, F_LOOP_START + 24), r.uniform(F_LOOP_START + 28, F_IDLE_END - 9))

        def idle(f, cs=cs, li=li):
            p = sum(bump(f, c, 1.5, 5.0) for c in cs)
            rot = ROT(roll=8.0 * x * li.a * p)
            sc = SCL(1.0 + 0.15 * x * p)
            return pivot(rot, sc), rot, sc
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)
    bake(master, F_LOOP_START, F_IDLE_END, lambda f: (None, None, SCL(1.0 + 0.012 * x * lwave2(f, 3))))

    # ---- EXIT: implode a little (anticipation), then explode - ballistic arcs with gravity ----
    off = exit_cascade(L, "random", gap=0.4, cap=2.0, words="together", rng=random.Random(15))
    DX = exit_dur(off)
    for li in L:
        r = letter_seed(li, 16)
        x0 = F_EXIT_START + off[li.name]
        vx = (li.dx if abs(li.dx) > 1e-3 else r.uniform(-0.5, 0.5)) * 1.3 + r.uniform(-0.4, 0.4)
        vz = r.uniform(1.2, 2.2)
        spin = r.uniform(-300, 300)

        def ex(f, x0=x0, vx=vx, vz=vz, spin=spin, li=li):
            pull = smooth(seg(f, x0, x0 + ANTIC)) * (1.0 - seg(f, x0 + ANTIC, x0 + ANTIC + 1.5))
            t = seg(f, x0 + ANTIC, x0 + DX)
            fly = out_quad(t)
            side = vx * x * fly - 0.15 * li.dx * pull
            up = vz * x * fly - 2.6 * t * t
            rot = ROT(roll=spin * t)
            sc = SCL(pw(f, (x0, 1.0), (x0 + ANTIC, 0.9), (x0 + ANTIC + 2, 1.1, out_quad),
                        (x0 + DX, HIDDEN, in_quad)))
            return vadd(LOC(side=side, up=up), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V16 - HINGE SWING  (hanging signs: the letters hang from their top edge)
# ============================================================

def v16_hinge_swing(master, letters):
    x, L = X(), letter_infos(letters)
    off = entry_cascade(L, gap=1.5)
    D = entry_dur(20, off)
    # ---- ENTRY: drop, the chain catches with a jolt, swing on the hinge until it calms ----
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        catch = s0 + 0.22 * D
        A = 38.0 * x * (li.sd or li.a)

        def entry(f, s0=s0, catch=catch, A=A):
            up = 1.3 * (1.0 - in_quad(seg(f, s0, catch)))
            j = kick(seg(f, catch, catch + 6), 1.0, 3.0)
            ang = A * settle(seg(f, catch, s0 + D), 2.2, 2.6)
            rot = ROT(roll=ang)
            sc = SQ(appear(f, s0, 2.5), 1.0 + 0.15 * x * j)
            return vadd(LOC(up=up - 0.06 * x * j), pivot(rot, sc, HINGE_H)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: wind - a slow sway with gusts, each sign a little after its neighbour ----
    for li in L:
        ph = li.i * 0.45

        def idle(f, ph=ph):
            rot = ROT(roll=6.0 * x * lwave(f, 2, ph) + 1.5 * x * lwave(f, 5, 2 * ph),
                      pitch=4.0 * x * lwave(f, 2, ph + 1.3))
            return pivot(rot, None, HINGE_H), rot, None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: swing back (anticipation), the hinge snaps, the letter tumbles down ----
    off = exit_cascade(L, "random", rng=random.Random(16))
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        sgn = li.sd or li.a

        def ex(f, x0=x0, sgn=sgn):
            sw = pw(f, (x0, 0.0), (x0 + 2, -10.0 * x * sgn, io_sine), (x0 + ANTIC + 1, 0.0, io_sine))
            if f < x0 + ANTIC + 1:
                rot = ROT(roll=sw)
                return pivot(rot, None, HINGE_H), rot, None
            u = seg(f, x0 + ANTIC + 1, x0 + DX)
            rot = ROT(roll=55.0 * sgn * in_quad(u))
            sc = SCL(vanish(f, x0 + 0.6 * DX, x0 + DX))
            return vadd(LOC(up=-2.3 * in_quad(u)), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V17 - WHEEL ROLL  (rolls without slipping: turn = distance / radius)
# ============================================================

def v17_wheel_roll(master, letters):
    x, L = X(), letter_infos(letters)
    R, DIST = PIVOT_H, 3.4
    off = entry_cascade(L, "rtl", gap=1.4)
    D = entry_dur(17, off)
    # ---- ENTRY: roll in from the left, the leader first, rock back on the stop ----
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0):
            side = -DIST * (1.0 - spring(seg(f, s0, s0 + D), 0.04 * x, 0.5))
            rot = ROT(roll=math.degrees(side / R))
            sc = SCL(appear(f, s0, 3))
            return vadd(LOC(side=side), pivot(rot, sc)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)
    master_push(master, 0.97)

    # ---- LOOP: rocking back and forth, rolling (no sliding) ----
    for li in L:
        ph = li.i * 0.5

        def idle(f, ph=ph):
            th = math.radians(7.0 * x * lwave(f, 2, ph))
            rot = ROT(roll=math.degrees(th))
            return vadd(LOC(side=th * R), pivot(rot, None)), rot, None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: roll back a touch (anticipation), then accelerate away to the right ----
    off = exit_cascade(L, "rtl")
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0):
            side = pw(f, (x0, 0.0), (x0 + ANTIC, -0.12 * x, io_sine), (x0 + DX, 3.6, in_cubic))
            rot = ROT(roll=math.degrees(side / R))
            sc = SCL(vanish(f, x0 + 0.65 * DX, x0 + DX))
            return vadd(LOC(side=side), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V18 - ZIPPER SLIDE  (letters slide in from alternating sides and interlock)
# ============================================================

def v18_zipper_slide(master, letters):
    x, L = X(), letter_infos(letters)
    DIST = 3.0
    off = entry_cascade(L, gap=1.2)
    D = entry_dur(14, off)
    # ---- ENTRY: slide in, stretched by the speed; braking makes the top lean forward (drag) ----
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        dr = -1.0 if li.i % 2 == 0 else 1.0

        def sidef(f, s0=s0, dr=dr):
            return dr * DIST * (1.0 - spring(seg(f, s0, s0 + D), 0.05 * x, 0.45))

        def entry(f, s0=s0, sidef=sidef):
            acc = sidef(f + 1) - 2.0 * sidef(f) + sidef(f - 1)
            lean = max(-25.0, min(25.0, -260.0 * x * acc))
            st = stretch_of(ddf(sidef, f), 0.35 * x, hi=1.7)
            return LOC(side=sidef(f)), ROT(roll=lean), SQX(appear(f, s0, 2), st)
        bake(li.b, F_START, F_ENTRY_END, entry)
    master_push(master, 0.97)

    # ---- LOOP: shimmy - the two halves of the zipper rub against each other ----
    for li in L:
        dr = -1.0 if li.i % 2 == 0 else 1.0

        def idle(f, dr=dr):
            return LOC(side=0.04 * x * dr * lwave(f, 3)), ROT(roll=3.0 * x * dr * lwave(f, 3, 0.8)), None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: wind back (anticipation), then zip straight through to the other side ----
    off = exit_cascade(L)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        dr = -1.0 if li.i % 2 == 0 else 1.0           # came from dr, leaves to -dr

        def sidef(f, x0=x0, dr=dr):
            return pw(f, (x0, 0.0), (x0 + ANTIC, 0.08 * x * dr, io_sine), (x0 + DX, -3.2 * dr, in_cubic))

        def ex(f, x0=x0, sidef=sidef, dr=dr):
            lean = pw(f, (x0, 0.0), (x0 + ANTIC, -8.0 * x * dr, io_sine), (x0 + DX, 4.0 * dr))
            st = stretch_of(ddf(sidef, f), 0.35 * x, hi=1.8)
            return LOC(side=sidef(f)), ROT(roll=lean), SQX(vanish(f, x0 + 0.7 * DX, x0 + DX), st)
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V19 - RUBBER SNAP  (each word is a rubber band; uses the word parent bones)
# ============================================================

def v19_rubber_snap(master, letters):
    x, L = X(), letter_infos(letters)
    ng = word_count(L)
    BEAT = 5.0 * CTX.get("stagger", 1.0)
    have_roots = any(r is not None for r in CTX["wroots"])
    t0s = [F_START + g * BEAT for g in range(ng)] if have_roots else [F_START]
    SNAP = 4.0
    SD = 18.0

    def band_entry(f, t0):
        if f < t0:
            return None, None, SCL(HIDDEN)
        line = seg(f, t0, t0 + 3)
        u = seg(f, t0 + SNAP, t0 + SNAP + SD)
        side = lerp(2.6, 1.0, spring(u, 0.20 * x, 0.28))
        up = lerp(0.06, 1.0, spring(u, 0.28 * x, 0.34))
        if f < t0 + SNAP:
            side, up = lerp(1.6, 2.6, out_cubic(line)), lerp(HIDDEN, 0.06, out_cubic(line))
        return None, None, SCL(1.0, up=up, side=side)

    # ---- ENTRY: a thin stretched line snaps back into the word, wobbling like rubber ----
    if have_roots:
        bake_words(F_START, F_ENTRY_END, lambda f, gi: band_entry(f, t0s[gi]))
    else:
        bake(master, F_START, F_ENTRY_END, lambda f: band_entry(f, t0s[0]))
    for li in L:
        t0 = t0s[li.gi] if have_roots else t0s[0]
        d = t0 + SNAP + 0.6 * abs(li.dx)              # the ends of the band react last

        def entry(f, d=d, li=li):
            k = kick(seg(f, d, min(F_ENTRY_END, d + 12)), 1.25, 3.0)
            rot = ROT(roll=10.0 * x * li.s * k)
            return vadd(LOC(up=0.04 * x * k), pivot(rot, None)), rot, None
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: two tugs on the band ----
    tugs = (F_LOOP_START + 3, F_LOOP_START + 27)

    def band_idle(f, gi):
        k = sum(kick(seg(f, c + 2 * gi, c + 2 * gi + 14), 1.25, 3.0) for c in tugs)
        return None, None, SQX(1.0, 1.0 + 0.07 * x * k)
    if have_roots:
        bake_words(F_LOOP_START, F_IDLE_END, band_idle)
    else:
        bake(master, F_LOOP_START, F_IDLE_END, lambda f: band_idle(f, 0))
    for li in L:
        def idle(f, li=li):
            k = sum(kick(seg(f, c + 2 * li.gi + 1 + 0.5 * abs(li.dx),
                             c + 2 * li.gi + 13 + 0.5 * abs(li.dx)), 1.25, 3.0) for c in tugs)
            rot = ROT(roll=4.0 * x * li.s * k)
            return pivot(rot, None), rot, None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: squeeze (anticipation), stretch out into a line, snap away ----
    def band_exit(f, gi):
        x0 = F_EXIT_START + 2.0 * gi * CTX.get("stagger", 1.0)
        side = pw(f, (x0, 1.0), (x0 + ANTIC, 0.85, io_sine), (F_END - 2, 3.0, in_cubic), (F_END, HIDDEN, in_quad))
        up = pw(f, (x0, 1.0), (x0 + ANTIC, 1.0 / 0.85 ** 0.5, io_sine), (F_END - 3, 0.04, in_cubic))
        return None, None, SCL(1.0, up=up, side=side)
    if have_roots:
        bake_words(F_EXIT_START, F_END, band_exit)
    else:
        bake(master, F_EXIT_START, F_END, lambda f: band_exit(f, 0))
    for li in L:
        bake(li.b, F_EXIT_START, F_END, lambda f: (None, None, SCL(vanish(f, F_END - 2, F_END))))


# ============================================================
# V20 - SPLIT FLAP  (airport departure board)
# ============================================================

def v20_split_flap(master, letters):
    x, L = X(), letter_infos(letters)
    FLIPS = 4
    off = entry_cascade(L, gap=1.2)
    D = entry_dur(19, off)
    SETTLE = 5.0
    per = (D - SETTLE) / FLIPS
    # ---- ENTRY: the flaps fall half a turn at a time (each one accelerates, then clacks) ----
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        t_end = s0 + FLIPS * per

        def entry(f, s0=s0, t_end=t_end):
            done = sum(in_quad(seg(f, s0 + k * per, s0 + k * per + 0.72 * per)) for k in range(FLIPS))
            pitch = -180.0 * (FLIPS - done) - 8.0 * x * kick(seg(f, t_end, t_end + SETTLE), 1.0, 3.0)
            st = 1.0 - 0.05 * x * sum(bump(f, s0 + k * per + 0.72 * per, 0.8, 1.5) for k in range(FLIPS))
            rot = ROT(pitch=pitch)
            sc = SQ(appear(f, s0, 2), st)
            return pivot(rot, sc), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: now and then a flap flips over, and later flips back ----
    for li in L:
        r = letter_seed(li, 20)
        flips = r.random() < 0.6
        c1 = r.uniform(F_LOOP_START + 3, F_LOOP_START + 20)
        c2 = r.uniform(F_LOOP_START + 29, F_IDLE_END - 11)

        def idle(f, c1=c1, c2=c2, flips=flips, li=li):
            pitch = 0.0
            if flips:
                pitch = 360.0 * (io_cubic(seg(f, c1, c1 + 8)) - io_cubic(seg(f, c2, c2 + 8)))
            pitch += 2.0 * x * lwave(f, 2, li.i * 0.7)
            rot = ROT(pitch=pitch)
            return pivot(rot, None), rot, None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: lift back (anticipation), then each flap falls forward and disappears ----
    off = exit_cascade(L, gap=0.8)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0):
            pitch = pw(f, (x0, 0.0), (x0 + ANTIC, -12.0 * x, io_sine), (x0 + DX - 1, 90.0, in_quad))
            sc = SCL(vanish(f, x0 + DX - 3, x0 + DX))
            rot = ROT(pitch=pitch)
            return pivot(rot, sc), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V21 - VORTEX SPIRAL
# ============================================================

def v21_vortex_spiral(master, letters):
    x, L = X(), letter_infos(letters)
    RADD, TURN = 2.0, 1.3 * math.pi
    off = entry_cascade(L, gap=1.0, cap=7.0)
    D = entry_dur(18, off, end=F_ENTRY_END - 2)
    # ---- ENTRY: spiral in from a wide orbit, the letter turns with the orbit and lags behind ----
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        r0, a0 = math.hypot(li.rx, li.rz), math.atan2(li.rz, li.rx)

        def ang(f, s0=s0):
            return TURN * (1.0 - out_quad(seg(f, s0, s0 + D)))

        def entry(f, s0=s0, r0=r0, a0=a0, ang=ang):
            r = r0 + RADD * (1.0 - spring(seg(f, s0, s0 + D), 0.06 * x, 0.45, soft=True))
            a = a0 + ang(f)
            side, up = r * math.cos(a) - li.rx, r * math.sin(a) - li.rz
            rot = ROT(roll=-math.degrees(ang(f - 1.5)))
            sc = SCL(0.25 + 0.75 * spring(seg(f, s0, s0 + 0.8 * D), 0.15 * x, 0.35))
            if f < s0:
                sc = SCL(HIDDEN)
            return vadd(LOC(side=side, up=up), pivot(rot, sc)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: a slow rotating wave around the text centre ----
    bake(master, F_LOOP_START, F_IDLE_END, lambda f: (None, ROT(roll=1.5 * x * lwave(f, 1)), None))
    for li in L:
        a0 = math.atan2(li.rz, li.rx)

        def idle(f, a0=a0):
            e, w = env(f), 2.0 * math.pi * 2.0 * loop_u(f) + a0
            rot = ROT(roll=2.0 * x * lwave(f, 2, a0))
            return vadd(LOC(side=0.03 * x * e * math.cos(w), up=0.03 * x * e * math.sin(w)), pivot(rot, None)), rot, None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: swell outward (anticipation), then get sucked into the centre ----
    rad = cascade(L, 1.0, 0.0, 4.0, "radial", "together")
    top = max(rad.values() or [0.0])
    off = {k: top - v for k, v in rad.items()}                  # outer letters go first
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        r0, a0 = math.hypot(li.rx, li.rz), math.atan2(li.rz, li.rx)

        def ex(f, x0=x0, r0=r0, a0=a0):
            r = pw(f, (x0, r0), (x0 + ANTIC, r0 + 0.12 * x, io_sine), (x0 + DX, 0.0, in_cubic))
            da = TURN * 0.9 * in_cubic(seg(f, x0 + ANTIC, x0 + DX))
            a = a0 + da
            rot = ROT(roll=-math.degrees(da))
            sc = SCL(vanish(f, x0 + 0.4 * DX, x0 + DX))
            return vadd(LOC(side=r * math.cos(a) - li.rx, up=r * math.sin(a) - li.rz), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V22 - CARD DEAL  (dealt one by one from a deck at the bottom left)
# ============================================================

def v22_card_deal(master, letters):
    x, L = X(), letter_infos(letters)
    DECK = (-(CTX["span"] * 0.5 + 2.2), -1.8)             # from the text centre, bone units
    off = entry_cascade(L, gap=1.6, beat=2.5, cap=11.0)
    D = entry_dur(14, off)
    # ---- ENTRY: thrown along an arc, spinning, slides past its spot and settles back ----
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        p0 = (DECK[0] - li.rx, DECK[1] - li.rz)

        def entry(f, s0=s0, p0=p0):
            u = seg(f, s0, s0 + D)
            t = spring(u, 0.06 * x, 0.5)
            px, pz = _arc(p0, t, -0.25)
            rot = ROT(roll=-150.0 * (1.0 - spring(seg(f, s0 + 1, s0 + D), 0.12 * x, 0.35)))
            sc = SCL(appear(f, s0, 2) * lerp(0.7, 1.0, out_cubic(seg(f, s0, s0 + 0.4 * D))))
            return vadd(LOC(side=px, up=pz, depth=-0.25 * hop(min(1.0, t))), pivot(rot, sc)), rot, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: the cards float and tilt like they rest on air ----
    for li in L:
        ph = li.i * 0.55

        def idle(f, ph=ph):
            rot = ROT(pitch=4.0 * x * lwave(f, 2, ph), roll=2.0 * x * lwave(f, 2, ph + 1.6))
            return vadd(LOC(up=0.025 * x * lwave(f, 2, ph + 0.8)), pivot(rot, None)), rot, None
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: lift (anticipation), then gathered back into the deck, last card first ----
    off = exit_cascade(L, "reverse", gap=0.8)
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        p0 = (DECK[0] - li.rx, DECK[1] - li.rz)

        def ex(f, x0=x0, p0=p0):
            lift = smooth(seg(f, x0, x0 + ANTIC))
            t = 1.0 - in_cubic(seg(f, x0 + ANTIC, x0 + DX))
            px, pz = _arc(p0, t, -0.25)
            rot = ROT(roll=-150.0 * in_quad(seg(f, x0 + ANTIC, x0 + DX)))
            sc = SCL(pw(f, (x0 + ANTIC, 1.0), (x0 + DX - 2, 0.6), (x0 + DX, HIDDEN, in_quad)))
            return vadd(LOC(side=px, up=pz + 0.05 * x * lift, depth=-0.1 * x * lift), pivot(rot, sc)), rot, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V23 - SHOCKWAVE RIPPLE  (everything radiates from the text centre)
# ============================================================

def v23_shockwave_ripple(master, letters):
    x, L = X(), letter_infos(letters)
    dist = {li.name: math.hypot(li.rx, li.rz) for li in L}
    dmax = max(dist.values() or [1.0]) or 1.0
    st = CTX.get("stagger", 1.0)
    C = min(2.2, 8.0 / dmax) * st                         # frames per unit: speed of the ring
    # ---- ENTRY: a ring expands; each letter pops out of the screen as it passes ----
    bake(master, F_START, F_ENTRY_END, lambda f: (None, None, SCL(1.0 + 0.04 * x * kick(seg(f, 2, 14)))))
    for li in L:
        d = dist[li.name] * C
        s0 = F_START + 1.0 + d
        t2 = F_START + 12.0 + d                             # second, smaller ring

        def entry(f, s0=s0, t2=t2):
            k1 = kick(seg(f, s0, min(F_ENTRY_END, s0 + 12)), 1.0, 3.0)
            k2 = kick(seg(f, t2, min(F_ENTRY_END, t2 + 9)), 1.0, 3.0)
            s = HIDDEN + (1.0 - HIDDEN) * spring(seg(f, s0, s0 + 10), 0.22 * x, 0.32)
            sc = SCL(s * (1.0 + 0.06 * x * k2))
            loc = LOC(up=0.25 * x * k1 + 0.08 * x * k2, depth=-0.5 * (1.0 - out_cubic(seg(f, s0, s0 + 8))))
            return vadd(loc, pivot(None, sc)), None, sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: two ripples run out from the centre ----
    for li in L:
        d = dist[li.name] * C * 0.8

        def idle(f, d=d):
            k = sum(kick(seg(f, c + d, c + d + 10), 1.0, 3.0) for c in (F_LOOP_START + 2, F_LOOP_START + 26))
            sc = SCL(1.0 + 0.05 * x * k)
            return vadd(LOC(up=0.06 * x * k), pivot(None, sc)), None, sc
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: push outward (anticipation), then implode into the centre, outer letters first ----
    rad = cascade(L, 1.0, 0.0, 4.0, "radial", "together")
    top = max(rad.values() or [0.0])
    off = {k: top - v for k, v in rad.items()}
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        dd = dist[li.name] or 1.0
        ux, uz = li.rx / dd, li.rz / dd

        def ex(f, x0=x0, ux=ux, uz=uz, li=li):
            push = 0.08 * x * smooth(seg(f, x0, x0 + ANTIC)) * (1.0 - seg(f, x0 + ANTIC, x0 + ANTIC + 2))
            t = in_cubic(seg(f, x0 + ANTIC, x0 + DX))
            rot = ROT(roll=90.0 * (li.sd or li.a) * t)
            sc = SCL(vanish(f, x0 + ANTIC, x0 + DX))
            return (vadd(LOC(side=ux * push - li.rx * t, up=uz * push - li.rz * t), pivot(rot, sc)), rot, sc)
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V24 - BUBBLE FLOAT
# ============================================================

def v24_bubble_float(master, letters):
    x, L = X(), letter_infos(letters)
    off = entry_cascade(L, "random", gap=1.1, rng=random.Random(24))
    D = entry_dur(18, off)
    # ---- ENTRY: rise in a wobbling bubble that drifts side to side, bob into place ----
    for li in L:
        r = letter_seed(li, 24)
        s0 = F_START + ENTRY_LEAD + off[li.name]
        ph = r.uniform(0.0, 2.0 * math.pi)

        def entry(f, s0=s0, ph=ph):
            u = seg(f, s0, s0 + D)
            fade = 1.0 - smoother(u)
            up = -1.7 * (1.0 - spring(u, 0.05 * x, 0.5, soft=True))
            side = 0.12 * x * math.sin(2.0 * math.pi * 1.5 * u + ph) * fade
            st = 1.0 + 0.12 * x * math.sin(2.0 * math.pi * 3.0 * u + ph) * fade
            s = appear(f, s0, 2) * lerp(0.25, 1.0, out_cubic(seg(f, s0, s0 + 0.6 * D)))
            s *= 1.0 + 0.08 * x * kick(seg(f, s0 + 0.7 * D, s0 + D))
            sc = SQ(s, st)
            return vadd(LOC(up=up, side=side), pivot(None, sc)), ROT(roll=6.0 * x * math.sin(2.0 * math.pi * 1.5 * u + ph + 1.0) * fade), sc
        bake(li.b, F_START, F_ENTRY_END, entry)

    # ---- LOOP: every bubble floats with its own rhythm ----
    for li in L:
        r = letter_seed(li, 25)
        cyc = r.choice((2, 3))
        ph, ph2 = r.uniform(0, 2 * math.pi), r.uniform(0, 2 * math.pi)

        def idle(f, cyc=cyc, ph=ph, ph2=ph2):
            sc = SQ(1.0, 1.0 + 0.03 * x * lwave(f, cyc, ph - 1.0))
            rot = ROT(roll=3.0 * x * lwave(f, cyc, ph + 1.0))
            return (vadd(LOC(up=0.05 * x * lwave(f, cyc, ph), side=0.025 * x * lwave(f, 1, ph2)), pivot(rot, sc)),
                    rot, sc)
        bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    # ---- EXIT: squeeze down (anticipation), float up, swell ... pop! ----
    off = exit_cascade(L, "random", rng=random.Random(25))
    DX = exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        ph = letter_seed(li, 26).uniform(0.0, 2.0 * math.pi)

        def ex(f, x0=x0, ph=ph):
            st = pw(f, (x0, 1.0), (x0 + ANTIC, 1.0 - 0.15 * x, io_sine), (x0 + ANTIC + 2, 1.05, out_quad))
            up = pw(f, (x0, 0.0), (x0 + ANTIC, -0.06, io_sine), (x0 + DX, 1.0, in_quad))
            side = 0.08 * x * math.sin(ph + 2.0 * math.pi * seg(f, x0, x0 + DX))
            side -= 0.08 * x * math.sin(ph)
            s = pw(f, (x0, 1.0), (x0 + DX - 3, 1.08), (x0 + DX - 1.5, 1.0 + 0.3 * x, out_quad),
                   (x0 + DX, HIDDEN, in_quad))
            sc = SQ(s, st)
            return vadd(LOC(up=up, side=side * seg(f, x0, x0 + 2)), pivot(None, sc)), None, sc
        bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# VARIATIONS
# ============================================================

VARIATIONS = [
    ("V1_Elastic_Pop", v1_elastic_pop),
    ("V2_Bounce_Drop", v2_bounce_drop),
    ("V3_Spin_Flip", v3_spin_flip),
    ("V4_Slam_Shake", v4_slam_shake),
    ("V5_Pendulum_Swing", v5_pendulum_swing),
    ("V6_Rocket_Rise", v6_rocket_rise),
    ("V7_Jelly_Wobble", v7_jelly_wobble),
    ("V8_Spin_Zoom", v8_spin_zoom),
    ("V9_Scatter_Assemble", v9_scatter_assemble),
    ("V10_Glitch_Flicker", v10_glitch_flicker),
    ("V11_Domino_Flip", v11_domino_flip),
    ("V12_Accordion_Spread", v12_accordion_spread),
    ("V13_Reel_Stop", v13_reel_stop),
    ("V14_Stamp_Slam", v14_stamp_slam),
    ("V15_Firework_Burst", v15_firework_burst),
    ("V16_Hinge_Swing", v16_hinge_swing),
    ("V17_Wheel_Roll", v17_wheel_roll),
    ("V18_Zipper_Slide", v18_zipper_slide),
    ("V19_Rubber_Snap", v19_rubber_snap),
    ("V20_Split_Flap", v20_split_flap),
    ("V21_Vortex_Spiral", v21_vortex_spiral),
    ("V22_Card_Deal", v22_card_deal),
    ("V23_Shockwave_Ripple", v23_shockwave_ripple),
    ("V24_Bubble_Float", v24_bubble_float),
]

VARIATION_DESCRIPTIONS = {
    1: "Springy pop-in with dragging tilt, heartbeat idle, crouch + pop-out",
    2: "Real bounce physics with squash, stadium-wave hops, hop + fall away",
    3: "360° coin spin-in with overshoot, flip waves, wind-up spin-out",
    4: "Huge slam + damped shake + shockwave, power tremble, zoom-through",
    5: "Roly-poly drop and rock, metronome sway, rock back and tip over",
    6: "Rocket launch with stretch, hover float, crouch + blast off",
    7: "Jelly plop and jiggle, squishy wobble, squash-jump out",
    8: "540° spin zoom with dragging letters, glint sweep, spin zoom-out",
    9: "Arcing scatter fly-in, magnet orbit, burst scatter out",
    10: "Digital flicker + lock-in, glitch bursts, CRT power-off",
    11: "Stand up from flat in a chain, domino ripple, topple over",
    12: "Fan out from each word's middle, bellows breathe, fold back",
    13: "Slot reels spin and stop left to right, nudges, spin away",
    14: "Typewriter stamps one by one, re-ink press, backspace out",
    15: "Rocket per word bursts into letters, twinkles, explode out",
    16: "Hanging signs catch and swing, wind sway, hinge snaps",
    17: "Rolls in like wheels (no sliding), rocking, rolls out",
    18: "Alternating slides interlock, zipper shimmy, zip through",
    19: "Rubber-band words snap into shape, tugs, stretch away",
    20: "Airport split-flap flips, occasional flips, flaps fall",
    21: "Spiral in from a vortex, rotating wave, sucked into centre",
    22: "Dealt like cards from a deck, floating tilt, gathered back",
    23: "Shockwave rings pop letters out, ripples, implode",
    24: "Bubbles rise and wobble, floating bob, float up and pop",
}
# ============================================================
# EXTRA VARIATIONS  V25 - V44  -  SAFE REVISION  (20 new Entry / Loop / Exit actions)
#
# PASTE this WHOLE block ONCE, directly AFTER the VARIATION_DESCRIPTIONS = { ... } dictionary
# and BEFORE the "# RIG CREATION" banner. Nothing that already exists is edited.
#
# Same builders / numbers as your hardened revision. What is different:
#   * NO existing name is re-defined (bake, entry_dur, exit_dur, entry_cascade, exit_cascade,
#     ddf, master_push and the F_* constants stay exactly as they are), so V1 - V24 build
#     exactly as before. Every helper of this block has its own new name (_x_..., _v...).
#   * Nothing touches Blender data while a variation is being built (the old action, scene
#     frame, F-Curves ...). The check at the end of every variation is a pure audit of the
#     animation functions: each curve must be exactly at rest on frames 29/30, 31, 85 and 86.
#     (The action itself is verified by the add-on's own Verify / Build All.)
#   * _vsettle(u <= 0) is 1.0 again (like settle) - it was 0.0 and made the V29 hello wave pop.
#
# Direction reminder:  LOC(depth = NEGATIVE) = TOWARD the camera, POSITIVE = away.
#   WAVING        V25 - V30
#   CAMERA MOVES  V31 - V38
#   SCALING       V39 - V44
# ============================================================

VERTIGO_K = 0.22          # V44: scale compensation per unit of depth (tweak to taste)
XTRA_AUDIT = True         # print a line when a baked curve is not exactly at rest on a clip border
XTRA_STRICT = False       # True -> raise instead (that variation then fails to build)

_ENTRY_SETTLE_FRAME = float(F_ENTRY_END - 1)    # entries are finished one frame before the border
_X_REC = None                                   # curves of the variation that is being built


def _xs(li):
    return li.sd or li.a


def _xclamp(v, lo, hi):
    return max(lo, min(hi, v))


def _safe_mid(a, b, target):
    """Clamp target strictly inside (a, b) so pw() knots never collide or invert."""
    a, b = float(a), float(b)
    if b <= a:
        return a
    eps = min(0.5, 0.25 * (b - a))
    return _xclamp(float(target), a + eps, b - eps)


def _exit_antic(dx):
    """Anticipation length that always fits inside the exit duration dx."""
    dx = float(dx)
    return min(float(ANTIC), max(0.5, 0.45 * dx))


def _capped_lags(raw_lags, min_m_dur=10.0):
    """Scale the master-follow lags so (master end + biggest lag) <= the settle frame."""
    total_avail = float(_ENTRY_SETTLE_FRAME - F_START)
    min_m = min(float(min_m_dur), max(2.0, total_avail * 0.5))
    max_allowed = max(0.0, total_avail - min_m)
    max_raw = max(raw_lags.values(), default=0.0)
    if max_raw > max_allowed and max_raw > 0.0:
        scale = max_allowed / max_raw
        return {k: v * scale for k, v in raw_lags.items()}
    return dict(raw_lags)


# ============================================================
# CURVE PRIMITIVES (they only call the existing easing functions)
# ============================================================

def _vspring(u, *args, **kwargs):
    """spring() that lands on exactly 1.0 (zero speed) at u >= 1."""
    u = float(u)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    val = float(spring(u, *args, **kwargs))
    return val + (1.0 - val) * smoother(seg(u, 0.75, 1.0))


def _vsettle(u, *args, **kwargs):
    """settle() (1 -> 0 damped wobble) that ends on exactly 0.0 at u >= 1."""
    u = float(u)
    if u <= 0.0:
        return 1.0
    if u >= 1.0:
        return 0.0
    val = float(settle(u, *args, **kwargs))
    return val * (1.0 - smoother(seg(u, 0.75, 1.0)))


def _vkick(u, *args, **kwargs):
    """kick() that ends on exactly 0.0 at u >= 1."""
    u = float(u)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    val = float(kick(u, *args, **kwargs))
    return val * (1.0 - smoother(seg(u, 0.75, 1.0)))


def _vappear(f, s0, dur):
    """appear() that is at exactly 1.0 on or before the settle frame."""
    s0 = float(s0)
    d = min(float(dur), max(0.25, _ENTRY_SETTLE_FRAME - s0))
    if float(f) >= s0 + d:
        return 1.0
    return float(appear(f, s0, d))


def _vvanish(f, t0, t1):
    """vanish() that stays at exactly 1.0 up to frame 86 and ends on HIDDEN at t1."""
    t1 = float(t1)
    min_start = float(F_EXIT_START) + 0.25
    t0 = min(max(float(t0), min_start), max(min_start, t1 - 0.25))
    if t1 <= t0:
        t1 = t0 + 0.25
    f = float(f)
    if f <= t0:
        return 1.0
    if f >= t1:
        return float(HIDDEN)
    return float(vanish(f, t0, t1))


def _vpw(f, *knots):
    """pw() with the first / last value pinned outside the knots."""
    if not knots:
        return 0.0
    f = float(f)
    if f <= float(knots[0][0]):
        return float(knots[0][1])
    if f >= float(knots[-1][0]):
        return float(knots[-1][1])
    return float(pw(f, *knots))


# ============================================================
# TIMING HELPERS  (own copies - the originals are left alone)
# ============================================================

def _x_entry_cascade(L, *args, **kwargs):
    off = dict(entry_cascade(L, *args, **kwargs))
    if not off:
        return off
    max_off = max(off.values())
    max_allowed_off = max(0.0, (_ENTRY_SETTLE_FRAME - float(F_START + ENTRY_LEAD)) - 8.0)
    if max_off > max_allowed_off and max_off > 0.0:
        scale = max_allowed_off / max_off
        off = {k: v * scale for k, v in off.items()}
    return off


def _x_entry_dur(base_d, off):
    """Entry length so that the LAST letter lands on the settle frame (29)."""
    max_off = max(off.values()) if off else 0.0
    avail = _ENTRY_SETTLE_FRAME - (float(F_START + ENTRY_LEAD) + max_off)
    if avail <= 0.5:
        raise ValueError("Insufficient entry frames (%.2f)" % avail)
    return min(float(base_d), avail)


def _x_exit_cascade(L, *args, **kwargs):
    off = dict(exit_cascade(L, *args, **kwargs))
    if not off:
        return off
    max_off = max(off.values())
    total_exit = float(F_END - F_EXIT_START)
    min_dx = max(float(ANTIC) + 3.0, 6.0)
    max_allowed_off = max(0.0, total_exit - min_dx)
    if max_off > max_allowed_off and max_off > 0.0:
        scale = max_allowed_off / max_off
        off = {k: v * scale for k, v in off.items()}
    return off


def _x_exit_dur(off):
    max_off = max(off.values()) if off else 0.0
    avail = float(F_END - F_EXIT_START) - max_off
    if avail <= 0.5:
        raise ValueError("Insufficient exit frames (%.2f)" % avail)
    return avail


def _x_ddf(fn, f):
    """Speed (per frame) of a scalar curve fn(f) - a tight central difference."""
    f = float(f)
    h = 1e-4
    val = (fn(f + h) - fn(f - h)) / (2.0 * h)
    return 0.0 if abs(val) < 1e-9 else val


def _env(f, fade=5.0):
    """Idle envelope: exactly 0 (with zero speed) on frame 31 and 85."""
    f = float(f)
    if f <= float(F_LOOP_START) or f >= float(F_IDLE_END):
        return 0.0
    span = max(1.0, float(F_IDLE_END - F_LOOP_START))
    fade = min(float(fade), span * 0.25)
    return (smooth(seg(f, F_LOOP_START, F_LOOP_START + fade))
            * (1.0 - smooth(seg(f, F_IDLE_END - fade, F_IDLE_END))))


def _x_master_push(master, s0=0.96):
    """Moving hold for the master (entry), a breath in the loop and a slight shrink in the exit."""
    _x_bake(master, F_START, F_ENTRY_END,
            lambda f: (None, None, SCL(s0 + (1.0 - s0) * smoother(seg(f, F_START, _ENTRY_SETTLE_FRAME)))))
    _x_bake(master, F_LOOP_START, F_IDLE_END,
            lambda f: (None, None, SCL(1.0 + 0.012 * X() * lwave(f, 1) * _env(f))))
    _x_bake(master, F_EXIT_START, F_END,
            lambda f: (None, None, SCL(1.0 - (1.0 - s0) * in_quad(seg(f, F_EXIT_START, F_END)))))


# ============================================================
# BAKE + AUDIT  (pure Python: no Blender data is read or written)
# ============================================================

def _x_bake(bone, f0, f1, fn):
    """Exactly the existing bake(), plus: remember the curve so it can be audited afterwards."""
    if _X_REC is not None:
        _X_REC.append((bone, int(f0), int(f1), fn))
    bake(bone, f0, f1, fn)


def _x_dev(pose):
    """Largest distance of one (loc, rot, scl) result from the rest pose (None = rest)."""
    d = 0.0
    for v, rest in zip(pose, (0.0, 0.0, 1.0)):
        if v is not None:
            for c in v:
                d = max(d, abs(c - rest))
    return d


def _x_audit(vname, recs):
    """Every curve has to be exactly at rest on the clip borders: frames 29 / 30 (entry end),
    31 and 85 (loop ends), 86 (exit start). Frames 30 / 31 / 85 / 86 themselves are keyed as the
    rest pose by the add-on, which also gives flat Bezier handles there (zero speed)."""
    probes = {(F_START, F_ENTRY_END): (F_ENTRY_END - 1, F_ENTRY_END),
              (F_LOOP_START, F_IDLE_END): (F_LOOP_START, F_IDLE_END),
              (F_EXIT_START, F_END): (F_EXIT_START,)}
    bad = []
    for bone, f0, f1, fn in recs:
        for f in probes.get((f0, f1), ()):
            try:
                d = _x_dev(fn(f))
            except Exception as err:
                bad.append("%s @ %d: %s" % (bone.name, f, err))
                continue
            if d > 1e-5:
                bad.append("%s @ %d: %.5f from rest" % (bone.name, f, d))
    if bad:
        msg = "%s: %d curve(s) not at rest on a clip border: %s" % (vname, len(bad), "; ".join(bad[:4]))
        if XTRA_STRICT:
            raise AssertionError(msg)
        print("[Win Anim V25-44] " + msg)


def _x_guarded(fn):
    """Wrap a builder: record its curves while it runs, audit them when it is done."""
    def wrapper(master, letters):
        global _X_REC
        prev = _X_REC
        _X_REC = [] if XTRA_AUDIT else None
        try:
            res = fn(master, letters)
            if _X_REC is not None:
                _x_audit(fn.__name__, _X_REC)
            return res
        finally:
            _X_REC = prev
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ============================================================
# V25 - FLAG WAVE
# ============================================================

@_x_guarded
def v25_flag_wave(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.0)
    D = _x_entry_dur(17, off)
    _x_master_push(master, 0.96)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0):
            u = seg(f, s0, s0 + D)
            fade = 1.0 - smoother(u)
            w = 2.0 * math.pi * 1.75 * u
            depth = 2.4 * (1.0 - out_cubic(u)) + 0.35 * x * math.sin(w + 0.8) * fade
            up = 0.42 * x * math.sin(w + 1.6) * fade
            rot = ROT(roll=-20.0 * x * math.cos(w + 1.6) * fade)
            return vadd(LOC(up=up, depth=depth), pivot(rot, None)), rot, SCL(_vappear(f, s0, 3))

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        ph = li.rx * 0.9

        def idle(f, ph=ph):
            e = _env(f)
            rot = ROT(roll=-8.0 * x * lwave(f, 2, ph - math.pi / 2) * e)
            loc = LOC(
                up=0.11 * x * lwave(f, 2, ph) * e,
                depth=0.24 * x * lwave(f, 2, ph - 1.1) * e,
            )
            return vadd(loc, pivot(rot, None)), rot, None

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.8)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        amp_end = min(x0 + DX, x0 + antic + 3.0)

        def ex(f, x0=x0, amp_end=amp_end):
            u = seg(f, x0, x0 + DX)
            amp = smooth(seg(f, x0, amp_end))
            w = 2.0 * math.pi * 1.5 * u
            side = _vpw(f, (x0, 0.0), (x0 + antic, -0.12 * x, io_sine), (x0 + DX, 3.6, in_cubic))
            up = 0.35 * x * math.sin(w) * amp
            depth = -0.5 * x * math.sin(w + 1.0) * amp
            rot = ROT(
                roll=_vpw(f, (x0 + antic, 0.0), (x0 + DX, -50.0 * x, in_quad))
                + 14.0 * x * math.cos(w) * amp
            )
            sc = SCL(_vvanish(f, x0 + 0.55 * DX, x0 + DX))
            return vadd(LOC(side=side, up=up, depth=depth), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V26 - OCEAN SWELL
# ============================================================

@_x_guarded
def v26_ocean_swell(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.1)
    D = _x_entry_dur(19, off)
    _x_master_push(master, 0.97)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0):
            u = seg(f, s0, s0 + D)
            fade = 1.0 - smoother(u)
            w = 2.0 * math.pi * 1.5 * u
            side = -2.8 * (1.0 - _vspring(u, 0.06 * x, 0.45)) + 0.10 * x * math.sin(w) * fade
            up = 0.40 * x * math.sin(w + 0.9) * fade
            rot = ROT(
                pitch=5.0 * x * math.sin(w + 2.0) * fade,
                roll=-17.0 * x * math.cos(w + 0.9) * fade,
            )
            return vadd(LOC(side=side, up=up), pivot(rot, None)), rot, SCL(_vappear(f, s0, 3))

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        ph = li.rx * 0.5

        def idle(f, ph=ph):
            e = _env(f)
            rot = ROT(
                roll=-6.5 * x * lwave(f, 1, ph + 0.6) * e,
                pitch=3.5 * x * lwave(f, 2, ph) * e,
            )
            loc = LOC(
                up=(0.17 * x * lwave(f, 1, ph) + 0.03 * x * lwave(f, 3, 2.0 * ph)) * e,
                side=0.14 * x * lwave(f, 1, ph - 1.2) * e,
            )
            return vadd(loc, pivot(rot, None)), rot, None

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, "rtl", gap=0.8)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        amp_end = min(x0 + DX, x0 + antic + 2.0)

        def ex(f, x0=x0, amp_end=amp_end):
            amp = smooth(seg(f, x0, amp_end))
            u = seg(f, x0 + antic, x0 + DX)
            sw = math.sin(2.0 * math.pi * 1.25 * seg(f, x0, x0 + DX)) * amp
            side = _vpw(f, (x0, 0.0), (x0 + antic, -0.15 * x, io_sine), (x0 + DX, 3.8, in_cubic))
            up = 0.3 * x * sw - 0.9 * in_quad(u)
            rot = ROT(
                roll=_vpw(f, (x0 + antic, 0.0), (x0 + DX, -35.0 * x, in_quad)),
                pitch=6.0 * x * sw,
            )
            sc = SCL(_vvanish(f, x0 + 0.55 * DX, x0 + DX))
            return vadd(LOC(side=side, up=up), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V27 - TIDAL NOD
# ============================================================

@_x_guarded
def v27_tidal_nod(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.2)
    D = _x_entry_dur(16, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0):
            t = _vspring(seg(f, s0, s0 + D), 0.12 * x, 0.38)
            lean = 1.0 - t
            sc = SCL(_vappear(f, s0, 3) * (1.0 + 0.35 * lean))
            return LOC(depth=-1.4 * lean), ROT(pitch=68.0 * x * lean), sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        ph = li.rx * 0.8

        def idle(f, ph=ph):
            e = _env(f)
            return (
                LOC(up=0.025 * x * lwave2(f, 2, ph) * e),
                ROT(
                    pitch=10.0 * x * lwave(f, 2, ph) * e,
                    roll=1.5 * x * lwave(f, 2, ph + 1.0) * e,
                ),
                None,
            )

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.8)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        t_mid = _safe_mid(x0 + antic, x0 + DX, x0 + 0.55 * DX)

        def ex(f, x0=x0, t_mid=t_mid):
            pitch = _vpw(f, (x0, 0.0), (x0 + antic, -12.0 * x, io_sine), (x0 + DX, 75.0 * x, in_cubic))
            depth = _vpw(f, (x0 + antic, 0.0), (x0 + DX, -2.4, in_cubic))
            s = _vpw(f, (x0 + antic, 1.0), (t_mid, 1.3, io_sine), (x0 + DX, HIDDEN, in_quad))
            return LOC(depth=depth), ROT(pitch=pitch), SCL(s)

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V28 - SNAKE TRAIL
# ============================================================

@_x_guarded
def v28_snake_trail(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.4, beat=2.0, cap=10.0)
    D = _x_entry_dur(17, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def sidep(f, s0=s0):
            return -3.6 * (1.0 - out_cubic(seg(f, s0, s0 + D)))

        def upp(f, s0=s0):
            u = seg(f, s0, s0 + D)
            return -0.9 * x * math.sin(2.0 * math.pi * 1.5 * u) * (1.0 - u)

        def entry(f, s0=s0, sidep=sidep, upp=upp):
            u = seg(f, s0, s0 + D)
            head = math.degrees(math.atan2(_x_ddf(upp, f), _x_ddf(sidep, f) + 1e-6))
            lean = _xclamp(-0.45 * head, -28.0, 28.0) * (1.0 - smoother(seg(u, 0.65, 1.0)))
            rot = ROT(roll=lean)
            return vadd(LOC(side=sidep(f), up=upp(f)), pivot(rot, None)), rot, SCL(_vappear(f, s0, 3))

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        ph = li.i * 0.7

        def idle(f, ph=ph):
            e = _env(f)
            d = 0.3 * x * lwave(f, 2, ph) * e
            rot = ROT(roll=5.0 * x * lwave(f, 2, ph + 1.2) * e)
            sc = SCL(1.0 - 0.3 * d)
            return (
                vadd(LOC(depth=d, side=0.10 * x * lwave(f, 2, ph + math.pi / 2) * e), pivot(rot, sc)),
                rot,
                sc,
            )

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.9)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        lean_end = min(x0 + DX, x0 + antic + 3.0)

        def sidep(f, x0=x0):
            return _vpw(f, (x0, 0.0), (x0 + antic, -0.1 * x, io_sine), (x0 + DX, 4.0, in_cubic))

        def upp(f, x0=x0):
            u = seg(f, x0 + antic, x0 + DX)
            return 0.9 * x * math.sin(2.0 * math.pi * 1.25 * u) * u

        def ex(f, x0=x0, sidep=sidep, upp=upp, lean_end=lean_end):
            head = math.degrees(math.atan2(_x_ddf(upp, f), _x_ddf(sidep, f) + 1e-6))
            lean = _xclamp(-0.45 * head, -28.0, 28.0) * smooth(seg(f, x0 + antic, lean_end))
            rot = ROT(roll=lean)
            sc = SCL(_vvanish(f, x0 + 0.6 * DX, x0 + DX))
            return vadd(LOC(side=sidep(f), up=upp(f)), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V29 - HELLO WAVE
# ============================================================

@_x_guarded
def v29_hello_wave(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.2)
    D = _x_entry_dur(17, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        s_wv = min(s0 + 1.0, s0 + 0.25 * D)

        def entry(f, s0=s0, s_wv=s_wv, li=li):
            rise = _vspring(seg(f, s0, s0 + D), 0.12 * x, 0.38)
            wv = _vsettle(seg(f, s_wv, s0 + D), 3.0, 2.6)
            rot = ROT(
                roll=38.0 * x * li.a * wv,
                yaw=14.0 * x * _xs(li) * _vsettle(seg(f, s_wv, s0 + D), 2.0, 3.0),
            )
            return LOC(up=-1.1 * (1.0 - rise)), rot, SCL(_vappear(f, s0, 3))

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    n = len(L)
    g = min(1.3, 12.0 / max(1, n - 1))
    FD = 11.0
    for li in L:
        c1 = F_LOOP_START + 3 + g * li.i
        c2 = F_LOOP_START + 28 + g * (n - 1 - li.i)

        def idle(f, c1=c1, c2=c2):
            e = _env(f)
            u1, u2 = seg(f, c1, c1 + FD), seg(f, c2, c2 + FD)
            e1, e2 = math.sin(math.pi * u1) ** 2, math.sin(math.pi * u2) ** 2
            w1 = math.sin(2.0 * math.pi * 2.0 * u1) * e1
            w2 = math.sin(2.0 * math.pi * 2.0 * u2) * e2
            rot = ROT(roll=28.0 * x * (w1 + w2) * e, yaw=10.0 * x * (w1 - w2) * e)
            return LOC(up=0.12 * x * (e1 + e2) * e), rot, SCL(1.0 + 0.1 * x * (e1 + e2) * e)

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.7)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        w_end = min(x0 + DX, x0 + antic + 4.0)
        v_start = _safe_mid(x0, x0 + DX, min(x0 + 0.65 * DX, x0 + antic + 3.0))

        def ex(f, x0=x0, li=li, w_end=w_end, v_start=v_start):
            wu = seg(f, x0, w_end)
            env_w = math.sin(math.pi * wu) ** 2
            wav = math.sin(2.0 * math.pi * 1.75 * wu) * env_w
            t = seg(f, v_start, x0 + DX)
            rot = ROT(roll=30.0 * x * li.a * wav)
            return (
                LOC(depth=3.2 * in_cubic(t), up=0.08 * x * env_w),
                rot,
                SCL(_vvanish(f, v_start, x0 + DX)),
            )

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V30 - FOLDING SCREEN
# ============================================================

@_x_guarded
def v30_folding_screen(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.0)
    D = _x_entry_dur(16, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0, li=li):
            fold = 1.0 - _vspring(seg(f, s0, s0 + D), 0.12 * x, 0.4)
            return (
                LOC(side=-li.rx * 0.45 * fold, depth=-0.5 * li.a * fold),
                ROT(yaw=80.0 * li.a * fold),
                SCL(_vappear(f, s0, 2.5)),
            )

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        ph = li.i * 0.45

        def idle(f, li=li, ph=ph):
            e = _env(f)
            return (
                LOC(depth=0.09 * x * li.a * lwave(f, 2, ph + 1.2) * e),
                ROT(yaw=15.0 * x * li.a * lwave(f, 2, ph) * e),
                None,
            )

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.7)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0, li=li):
            yaw = _vpw(f, (x0, 0.0), (x0 + antic, -8.0 * x * li.a, io_sine), (x0 + DX, 88.0 * li.a, in_cubic))
            side = _vpw(f, (x0 + antic, 0.0), (x0 + DX, -li.rx * 0.5, in_quad))
            depth = _vpw(f, (x0 + antic, 0.0), (x0 + DX, 0.5 * li.a, in_quad))
            return LOC(side=side, depth=depth), ROT(yaw=yaw), SCL(_vvanish(f, x0 + 0.55 * DX, x0 + DX))

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V31 - FLY THROUGH
# ============================================================

@_x_guarded
def v31_fly_through(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.1)
    D = _x_entry_dur(17, off)
    _x_master_push(master, 0.98)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0, li=li):
            t = _vspring(seg(f, s0, s0 + D), 0.10 * x, 0.42)
            k = 1.0 - t
            rot = ROT(roll=25.0 * x * li.a * k)
            sc = SCL(_vappear(f, s0, 2) * (0.25 + 0.75 * t))
            return vadd(LOC(depth=6.0 * k, up=0.2 * k), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        ph = li.rx * 0.5

        def idle(f, ph=ph):
            e = _env(f)
            d = 0.45 * x * lwave(f, 1, ph) * e
            sc = SCL(1.0 - 0.25 * d)
            return vadd(LOC(depth=d, up=0.03 * x * lwave(f, 2, ph + 1.0) * e), pivot(None, sc)), None, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.8)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        t_pre = _safe_mid(x0 + antic, x0 + DX, x0 + DX - 2.0)

        def ex(f, x0=x0, t_pre=t_pre):
            depth = _vpw(f, (x0, 0.0), (x0 + antic, -0.45 * x, io_sine), (x0 + DX, 7.0, in_quad))
            up = _vpw(f, (x0 + antic, 0.0), (x0 + DX, 0.3, in_quad))
            s = _vpw(f, (x0, 1.0), (x0 + antic, 1.1, io_sine), (t_pre, 0.2, in_quad), (x0 + DX, HIDDEN))
            return LOC(depth=depth, up=up), None, SCL(s)

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V32 - CANNON FIRE
# ============================================================

@_x_guarded
def v32_cannon_fire(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, "center", gap=1.2, cap=8.0)
    D = _x_entry_dur(15, off)
    fires = []
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        fires.append(s0 + 0.3 * D)

        def entry(f, s0=s0, li=li):
            k = 1.0 - _vspring(seg(f, s0, s0 + D), 0.08 * x, 0.45)
            rot = ROT(roll=22.0 * x * li.a * k)
            sc = SCL(_vappear(f, s0, 2) * (1.0 + 1.2 * k))
            loc = LOC(
                side=-li.rx * 0.9 * k,
                up=-li.rz * 0.9 * k - 0.3 * k,
                depth=-2.0 * k,
            )
            return vadd(loc, pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    _x_bake(
        master, F_START, F_ENTRY_END,
        lambda f: (
            LOC(
                depth=0.12 * x * sum(
                    _vkick(seg(f, t, min(_ENTRY_SETTLE_FRAME, t + 6.0)), 1.0, 4.0)
                    for t in fires
                )
            ),
            None,
            None,
        ),
    )

    cs = (F_LOOP_START + 3, F_LOOP_START + 19, F_LOOP_START + 35)
    for li in L:
        d = 0.4 * abs(li.rx)

        def idle(f, d=d):
            e = _env(f)
            k = sum(_vkick(seg(f, c + d, c + d + 12), 1.0, 3.5) for c in cs) * e
            sc = SCL(1.0 - 0.05 * x * k)
            return vadd(LOC(depth=0.3 * x * k), pivot(None, sc)), None, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, "edges", gap=0.6)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        pull_rel = min(x0 + DX, x0 + antic + 2.0)

        def ex(f, x0=x0, li=li, pull_rel=pull_rel):
            t = in_cubic(seg(f, x0 + antic, x0 + DX))
            pull = smooth(seg(f, x0, x0 + antic)) * (1.0 - smooth(seg(f, x0 + antic, pull_rel)))
            rot = ROT(roll=-60.0 * li.a * t)
            sc = SCL(_vpw(f, (x0, 1.0), (x0 + antic, 1.1, io_sine), (x0 + DX, HIDDEN, in_cubic)))
            loc = LOC(side=-li.rx * 0.9 * t, up=-li.rz * 0.9 * t, depth=4.0 * t - 0.25 * x * pull)
            return vadd(loc, pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V33 - DOLLY RUSH
# ============================================================

@_x_guarded
def v33_dolly_rush(master, letters):
    x, L = X(), letter_infos(letters)
    lags = _capped_lags({li.name: 1.0 + 0.7 * abs(li.rx) for li in L}, min_m_dur=10.0)
    max_lag = max(lags.values(), default=0.0)
    m_end = _ENTRY_SETTLE_FRAME - max_lag

    def md(f):
        return 9.0 * (1.0 - _vspring(seg(f, F_START, m_end), 0.06 * x, 0.45))

    def m_entry(f):
        t = _vspring(seg(f, F_START, m_end), 0.06 * x, 0.45)
        return LOC(depth=md(f)), None, SCL((0.12 + 0.88 * t) * _vappear(f, F_START, 2))

    _x_bake(master, F_START, F_ENTRY_END, m_entry)
    for li in L:
        lag = lags[li.name]

        def entry(f, lag=lag):
            trail = 0.3 * (md(f - lag) - md(f))
            return LOC(depth=trail), ROT(pitch=-16.0 * x * _xclamp(trail / 2.5, -1.0, 1.0)), None

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    def m_idle(f):
        e = _env(f)
        w = lwave(f, 1) * e
        return LOC(depth=0.5 * x * w), None, SCL(1.0 - 0.04 * x * w)

    _x_bake(master, F_LOOP_START, F_IDLE_END, m_idle)
    for li in L:
        ph = li.rx * 0.35

        def idle(f, ph=ph):
            e = _env(f)
            return LOC(depth=0.18 * x * lwave(f, 1, ph) * e), None, None

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    antic = _exit_antic(F_END - F_EXIT_START)

    def mx(f):
        return _vpw(f, (F_EXIT_START, 0.0), (F_EXIT_START + antic, -0.5 * x, io_sine), (F_END, 9.0, in_expo))

    _x_bake(
        master, F_EXIT_START, F_END,
        lambda f: (
            LOC(depth=mx(f)),
            None,
            SCL(_vpw(f, (F_EXIT_START, 1.0), (F_EXIT_START + antic, 1.05, io_sine), (F_END, 0.12, in_quad))),
        ),
    )
    for li in L:
        lag = 1.0 + 0.5 * abs(li.rx)

        def ex(f, lag=lag):
            return LOC(depth=0.3 * (mx(f - lag) - mx(f))), None, SCL(_vvanish(f, F_END - 5, F_END))

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V34 - TUNNEL RUSH
# ============================================================

@_x_guarded
def v34_tunnel_rush(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, "center", gap=1.0)
    D = _x_entry_dur(17, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0, li=li):
            t = _vspring(seg(f, s0, s0 + D), 0.10 * x, 0.42)
            k = 1.0 - t
            rot = ROT(roll=140.0 * li.a * k)
            sc = SCL(_vappear(f, s0, 2) * (0.2 + 0.8 * t))
            return vadd(LOC(side=-li.rx * 0.9 * k, up=-li.rz * 0.9 * k, depth=5.0 * k), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        d = math.hypot(li.rx, li.rz)

        def idle(f, li=li, d=d):
            e = _env(f)
            w = lwave(f, 2, d * 0.5) * e
            sc = SCL(1.0 + 0.07 * x * w)
            loc = LOC(side=li.rx * 0.05 * x * w, up=li.rz * 0.05 * x * w, depth=-0.3 * x * w)
            return vadd(loc, pivot(None, sc)), None, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, "edges", gap=0.7)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        dx = li.rx if abs(li.rx) > 0.05 else 0.6 * li.a
        dz = li.rz
        t_peak = _safe_mid(x0 + antic, x0 + DX, x0 + DX - 3.0)
        pull_rel = min(x0 + DX, x0 + antic + 2.0)

        def ex(f, x0=x0, dx=dx, dz=dz, li=li, t_peak=t_peak, pull_rel=pull_rel):
            t = in_cubic(seg(f, x0 + antic, x0 + DX))
            pull = smooth(seg(f, x0, x0 + antic)) * (1.0 - smooth(seg(f, x0 + antic, pull_rel)))
            rot = ROT(roll=-80.0 * li.a * t)
            sc = SCL(_vpw(f, (x0 + antic, 1.0), (t_peak, 3.0, in_cubic), (x0 + DX, HIDDEN, in_quad)))
            loc = LOC(side=dx * 2.0 * t, up=dz * 2.0 * t + 0.2 * t, depth=0.3 * x * pull - 4.8 * t)
            return vadd(loc, pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V35 - DEPTH SWAP
# ============================================================

@_x_guarded
def v35_depth_swap(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=0.9)
    D = _x_entry_dur(16, off)
    for li in L:
        near = li.i % 2 == 1
        d0, s_start = (-3.0, 2.0) if near else (3.2, 0.45)
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0, d0=d0, s_start=s_start, near=near):
            k = 1.0 - _vspring(seg(f, s0, s0 + D), 0.08 * x, 0.45)
            rot = ROT(roll=12.0 * x * (1.0 if near else -1.0) * k)
            sc = SCL(_vappear(f, s0, 2) * (1.0 + (s_start - 1.0) * k))
            return vadd(LOC(depth=d0 * k), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        sg = 1.0 if li.i % 2 == 0 else -1.0
        ph = li.i * 0.15

        def idle(f, sg=sg, ph=ph):
            e = _env(f)
            d = 0.45 * x * sg * lwave(f, 2, ph) * e
            sc = SCL(1.0 - 0.28 * d)
            return vadd(LOC(depth=d), pivot(None, sc)), None, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.7)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        near = li.i % 2 == 1
        dirn = -1.0 if near else 1.0
        x0 = F_EXIT_START + off[li.name]
        t_peak = _safe_mid(x0 + antic, x0 + DX, x0 + DX - 2.0)

        def ex(f, x0=x0, dirn=dirn, near=near, t_peak=t_peak):
            depth = _vpw(f, (x0, 0.0), (x0 + antic, -0.3 * dirn * x, io_sine), (x0 + DX, 4.0 * dirn, in_cubic))
            s = _vpw(f, (x0 + antic, 1.0), (t_peak, 2.2 if near else 0.3, in_cubic), (x0 + DX, HIDDEN, in_quad))
            return vadd(LOC(depth=depth), pivot(None, SCL(s))), None, SCL(s)

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V36 - CAROUSEL ORBIT
# ============================================================

@_x_guarded
def v36_carousel_orbit(master, letters):
    x, L = X(), letter_infos(letters)
    lags = _capped_lags({li.name: 1.0 + 0.3 * abs(li.dx) for li in L}, min_m_dur=10.0)
    max_lag = max(lags.values(), default=0.0)
    m_end = _ENTRY_SETTLE_FRAME - max_lag

    def my(f):
        return 540.0 * (1.0 - _vspring(seg(f, F_START, m_end), 0.08 * x, 0.45))

    def m_entry(f):
        t = _vspring(seg(f, F_START, m_end), 0.08 * x, 0.45)
        return None, ROT(yaw=my(f)), SCL(0.55 + 0.45 * t)

    _x_bake(master, F_START, F_ENTRY_END, m_entry)
    off = _x_entry_cascade(L, gap=0.5, beat=1.0, cap=4.0)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        lag = lags[li.name]

        def entry(f, s0=s0, lag=lag):
            drag = _xclamp(0.22 * (my(f - lag) - my(f)), -60.0, 60.0)
            return None, ROT(yaw=drag), SCL(_vappear(f, s0, 3))

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    def m_idle(f):
        return None, ROT(yaw=22.0 * x * lwave(f, 1) * _env(f)), None

    _x_bake(master, F_LOOP_START, F_IDLE_END, m_idle)
    for li in L:
        def idle(f, li=li):
            e = _env(f)
            return (
                LOC(depth=0.05 * x * lwave(f, 2, li.rx * 0.4) * e),
                ROT(yaw=7.0 * x * lwave(f, 2, li.i * 0.5) * e),
                None,
            )

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    antic = _exit_antic(F_END - F_EXIT_START)

    def mx(f):
        return _vpw(f, (F_EXIT_START, 0.0), (F_EXIT_START + antic, -22.0 * x, io_sine), (F_END, 540.0, in_cubic))

    _x_bake(
        master, F_EXIT_START, F_END,
        lambda f: (None, ROT(yaw=mx(f)), SCL(_vpw(f, (F_EXIT_START, 1.0), (F_END, 0.45, in_quad)))),
    )
    for li in L:
        lag = 1.0 + 0.3 * abs(li.dx)

        def ex(f, lag=lag):
            drag = _xclamp(0.2 * (mx(f - lag) - mx(f)), -60.0, 60.0)
            return None, ROT(yaw=drag), SCL(_vvanish(f, F_END - 6, F_END))

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V37 - CURVED SCREEN
# ============================================================

@_x_guarded
def v37_curved_screen(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, "center", gap=1.2)
    D = _x_entry_dur(16, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0, li=li):
            t = _vspring(seg(f, s0, s0 + D), 0.08 * x, 0.45)
            k = 1.0 - t
            rot = ROT(yaw=-70.0 * li.s * k)
            sc = SCL(_vappear(f, s0, 2.5) * (0.35 + 0.65 * t))
            loc = LOC(depth=(3.6 + 0.6 * abs(li.s)) * k, side=li.s * 0.6 * k)
            return vadd(loc, pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        def idle(f, li=li):
            e = _env(f)
            w = lwave2(f, 2) * e
            depth = -0.55 * x * (1.0 - abs(li.s)) * w
            rot = ROT(yaw=-16.0 * x * li.s * w)
            sc = SCL(1.0 - 0.2 * depth)
            return vadd(LOC(depth=depth), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, "edges", gap=0.8)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0, li=li):
            t = in_cubic(seg(f, x0 + antic, x0 + DX))
            rot = ROT(yaw=-80.0 * li.s * t)
            depth = _vpw(f, (x0, 0.0), (x0 + antic, -0.2 * x, io_sine), (x0 + DX, 4.2, in_cubic))
            s = _vpw(f, (x0 + antic, 1.0), (x0 + DX, 0.3, in_quad)) * _vvanish(f, x0 + 0.6 * DX, x0 + DX)
            sc = SCL(s)
            return vadd(LOC(depth=depth), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V38 - TRUCK PAN
# ============================================================

@_x_guarded
def v38_truck_pan(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=0.5, beat=1.5, cap=4.0)
    D = _x_entry_dur(19, off)
    for li in L:
        near = li.i % 2 == 1
        sd0, dp0, sc0 = (6.5, -1.6, 1.35) if near else (3.2, 1.6, 0.72)
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def sidef(f, s0=s0, sd0=sd0):
            return sd0 * (1.0 - _vspring(seg(f, s0, s0 + D), 0.05 * x, 0.45))

        def entry(f, s0=s0, sidef=sidef, dp0=dp0, sc0=sc0):
            k = 1.0 - _vspring(seg(f, s0, s0 + D), 0.05 * x, 0.45)
            st = stretch_of(_x_ddf(sidef, f), 0.30 * x, hi=1.5)
            return (
                LOC(side=sidef(f), depth=dp0 * k),
                None,
                SQX(_vappear(f, s0, 2) * (1.0 + (sc0 - 1.0) * k), st),
            )

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        near = li.i % 2 == 1

        def idle(f, near=near):
            e = _env(f)
            return (
                LOC(
                    side=-(0.22 if near else 0.08) * x * lwave(f, 1) * e,
                    depth=(-0.15 if near else 0.15) * x * lwave(f, 1, 1.0) * e,
                ),
                None,
                None,
            )

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.6)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        near = li.i % 2 == 1
        x0 = F_EXIT_START + off[li.name]

        def sidef(f, x0=x0, near=near):
            return _vpw(f, (x0, 0.0), (x0 + antic, 0.15 * x, smoother), (x0 + DX, -(9.0 if near else 4.5), in_cubic))

        def ex(f, x0=x0, sidef=sidef, near=near):
            st = stretch_of(_x_ddf(sidef, f), 0.30 * x, hi=1.6)
            k = in_quad(seg(f, x0 + antic, x0 + DX))
            s = (1.0 + (0.35 if near else -0.28) * k) * _vvanish(f, x0 + 0.7 * DX, x0 + DX)
            return LOC(side=sidef(f), depth=(-1.6 if near else 1.6) * k), None, SQX(s, st)

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V39 - UNFOLD SCALE
# ============================================================

@_x_guarded
def v39_unfold_scale(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.0)
    D = _x_entry_dur(17, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0):
            if f < s0:
                return LOC(depth=-0.5), None, SCL(HIDDEN)
            sx = 0.03 + 0.97 * out_cubic(seg(f, s0, s0 + 0.4 * D))
            sy = 0.03 + 0.97 * _vspring(seg(f, s0 + 0.3 * D, s0 + D), 0.22 * x, 0.32)
            return LOC(depth=-0.5 * (1.0 - out_cubic(seg(f, s0, s0 + D)))), None, SCL(1.0, up=sy, side=sx, depth=sx)

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        r = letter_seed(li, 39)
        cyc = r.choice((2, 3, 4))
        ph = r.uniform(0.0, 2.0 * math.pi)

        def idle(f, cyc=cyc, ph=ph):
            w = lwave2(f, cyc, ph) * _env(f)
            return None, None, SCL(1.0, up=1.0 + 0.17 * x * w, side=1.0 - 0.04 * x * w)

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.7)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        t_sy_end = _safe_mid(x0 + antic, x0 + DX, x0 + 0.6 * DX)

        def ex(f, x0=x0, t_sy_end=t_sy_end):
            sy = _vpw(f, (x0, 1.0), (x0 + antic, 1.12, io_sine), (t_sy_end, 0.04, in_cubic))
            sx = _vpw(f, (x0, 1.0), (x0 + 0.55 * DX, 1.0), (x0 + DX, HIDDEN, in_quad))
            return None, None, SCL(1.0, up=max(HIDDEN, sy) if f < x0 + DX else HIDDEN, side=sx, depth=sx)

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V40 - ZOOM STEPS
# ============================================================

@_x_guarded
def v40_zoom_steps(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.0)
    D = _x_entry_dur(18, off)
    # all knots stay strictly ordered and the last click is finished by s0 + 0.90 * D
    step_w = min(3.0, 0.28 * D)
    step1_w = min(2.0, 0.24 * D)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def zs(f, s0=s0):
            return _vpw(
                f,
                (s0, HIDDEN),
                (s0 + step1_w, 0.35, out_cubic),
                (s0 + 0.30 * D, 0.35),
                (s0 + 0.30 * D + step_w, 0.7, out_cubic),
                (s0 + 0.62 * D, 0.7),
                (s0 + 0.62 * D + step_w, 1.0, back_out(2.2)),
            )

        def entry(f, zs=zs):
            s = zs(f)
            depth = 2.4 * (1.0 - _xclamp((s - 0.35) / 0.65, -0.3, 1.2))
            return LOC(depth=depth), None, SCL(s)

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    n = len(L)
    g = min(1.3, 12.0 / max(1, n - 1))
    for li in L:
        c1 = F_LOOP_START + 4 + g * li.i
        c2 = F_LOOP_START + 27 + g * (n - 1 - li.i)

        def idle(f, c1=c1, c2=c2):
            e = _env(f)
            s = 1.0
            for c in (c1, c2):
                s += 0.12 * x * (smooth(seg(f, c, c + 3)) - smooth(seg(f, c + 9, c + 12)))
            s = 1.0 + (s - 1.0) * e
            return LOC(depth=-2.0 * (s - 1.0)), None, SCL(s)

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.7)
    DX = _x_exit_dur(off)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0):
            s = _vpw(
                f,
                (x0, 1.0),
                (x0 + 0.2 * DX, 1.1, io_sine),
                (x0 + 0.4 * DX, 0.72, io_sine),
                (x0 + 0.55 * DX, 0.72),
                (x0 + 0.75 * DX, 0.42, io_sine),
                (x0 + DX, HIDDEN, in_quad),
            )
            return LOC(depth=2.6 * (1.0 - s)), None, SCL(s)

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V41 - LENS SWEEP
# ============================================================

@_x_guarded
def v41_lens_sweep(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.0, beat=2.0)
    D = _x_entry_dur(16, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        s_bulge = min(s0 + 2.0, s0 + 0.35 * D)

        def entry(f, s0=s0, s_bulge=s_bulge):
            bulge = 1.0 - smoother(seg(f, s_bulge, s0 + D))
            sc = SCL(_vappear(f, s0, 3) * (1.0 + 0.45 * x * bulge))
            return vadd(LOC(depth=-0.8 * x * bulge), pivot(None, sc)), None, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    n = len(L)
    g = min(1.3, 12.0 / max(1, n - 1))
    for li in L:
        c1 = F_LOOP_START + 9 + g * li.i
        c2 = F_LOOP_START + 34 + g * (n - 1 - li.i)

        def idle(f, c1=c1, c2=c2):
            e = _env(f)
            b = (bump(f, c1, 4.0, 6.0) + bump(f, c2, 4.0, 6.0)) * e
            sc = SCL(1.0 + 0.34 * x * b)
            return vadd(LOC(depth=-0.8 * x * b, up=0.05 * x * b), pivot(None, sc)), None, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.8)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0):
            s = _vpw(f, (x0, 1.0), (x0 + antic, 1.5, io_sine), (x0 + DX, HIDDEN, in_quad))
            depth = _vpw(f, (x0, 0.0), (x0 + antic, -0.9, io_sine), (x0 + DX, -2.0, in_quad))
            sc = SCL(s)
            return vadd(LOC(depth=depth), pivot(None, sc)), None, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V42 - TSUNAMI
# ============================================================

@_x_guarded
def v42_tsunami(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=1.1)
    D = _x_entry_dur(17, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0, li=li):
            u = seg(f, s0, s0 + D)
            k = _vkick(u, 1.0, 3.0)
            rot = ROT(roll=10.0 * x * li.a * k)
            sc = SCL(_vappear(f, s0, 3) * (1.0 + 0.9 * x * k))
            loc = LOC(up=-0.8 * (1.0 - out_cubic(u)) + 1.3 * x * k, depth=-1.6 * x * k)
            return vadd(loc, pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    n = len(L)
    g = min(1.3, 12.0 / max(1, n - 1))
    for li in L:
        c1 = F_LOOP_START + 8 + g * li.i
        c2 = F_LOOP_START + 32 + g * (n - 1 - li.i)

        def idle(f, c1=c1, c2=c2, li=li):
            e = _env(f)
            b = (bump(f, c1, 6.0, 9.0) + bump(f, c2, 6.0, 9.0)) * e
            rot = ROT(roll=-8.0 * x * li.a * b)
            sc = SCL(1.0 + 0.38 * x * b)
            return vadd(LOC(up=0.38 * x * b, depth=-0.9 * x * b), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, gap=0.8)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]

        def ex(f, x0=x0, li=li):
            t = in_cubic(seg(f, x0 + antic, x0 + DX))
            sw = math.sin(math.pi * seg(f, x0, x0 + DX)) ** 2
            rot = ROT(roll=12.0 * x * li.a * t)
            s = _vpw(f, (x0, 1.0), (x0 + 0.5 * DX, 1.8, io_sine), (x0 + DX, HIDDEN, in_quad))
            sc = SCL(s)
            return vadd(LOC(up=0.9 * x * sw, depth=-2.4 * t), pivot(rot, sc)), rot, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V43 - CRANE DOWN
# ============================================================

@_x_guarded
def v43_crane_down(master, letters):
    x, L = X(), letter_infos(letters)
    lags = _capped_lags({li.name: 1.0 + 0.25 * li.i for li in L}, min_m_dur=10.0)
    max_lag = max(lags.values(), default=0.0)
    m_end = _ENTRY_SETTLE_FRAME - max_lag

    def mu(f):
        return 3.6 * (1.0 - _vspring(seg(f, F_START, m_end), 0.07 * x, 0.45))

    def m_entry(f):
        t = _vspring(seg(f, F_START, m_end), 0.07 * x, 0.45)
        return (
            LOC(up=mu(f), depth=2.0 * (1.0 - t)),
            ROT(pitch=-24.0 * x * (1.0 - t)),
            SCL(1.0 + 0.3 * (1.0 - t)),
        )

    _x_bake(master, F_START, F_ENTRY_END, m_entry)
    off = _x_entry_cascade(L, gap=0.35, beat=0.5, cap=3.0)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]
        lag = lags[li.name]

        def entry(f, s0=s0, lag=lag):
            trail = 0.4 * (mu(f - lag) - mu(f))
            return LOC(up=trail), ROT(pitch=_xclamp(10.0 * x * trail / 1.4, -18.0, 18.0)), SCL(_vappear(f, s0, 3))

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    def m_idle(f):
        e = _env(f)
        return (
            LOC(up=0.10 * x * lwave(f, 1) * e, depth=0.15 * x * lwave(f, 1, 1.0) * e),
            ROT(pitch=2.0 * x * lwave(f, 1, 0.8) * e),
            None,
        )

    _x_bake(master, F_LOOP_START, F_IDLE_END, m_idle)
    for li in L:
        ph = li.i * 0.3

        def idle(f, ph=ph):
            e = _env(f)
            return LOC(up=0.035 * x * lwave(f, 2, ph) * e), None, None

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    antic = _exit_antic(F_END - F_EXIT_START)

    def mux(f):
        return _vpw(f, (F_EXIT_START, 0.0), (F_EXIT_START + antic, -0.15 * x, io_sine), (F_END, 4.5, in_cubic))

    def m_exit(f):
        t = in_cubic(seg(f, F_EXIT_START + antic, F_END))
        return LOC(up=mux(f), depth=3.0 * t), ROT(pitch=26.0 * x * t), SCL(1.0 - 0.4 * t)

    _x_bake(master, F_EXIT_START, F_END, m_exit)
    for li in L:
        lag = 1.0 + 0.25 * li.i

        def ex(f, lag=lag):
            return LOC(up=0.4 * (mux(f - lag) - mux(f))), None, SCL(_vvanish(f, F_END - 5, F_END))

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# V44 - VERTIGO ZOOM
# ============================================================

@_x_guarded
def v44_vertigo_zoom(master, letters):
    x, L = X(), letter_infos(letters)
    off = _x_entry_cascade(L, gap=0.3, beat=0.3, cap=2.5)
    D = _x_entry_dur(18, off)
    for li in L:
        s0 = F_START + ENTRY_LEAD + off[li.name]

        def entry(f, s0=s0, li=li):
            k = 1.0 - _vspring(seg(f, s0, s0 + D), 0.06 * x, 0.5)
            depth = 4.5 * k
            sc = SCL(_vappear(f, s0, 2) * (1.0 + VERTIGO_K * depth))
            loc = LOC(side=-li.rx * 0.7 * k, up=-li.rz * 0.7 * k, depth=depth)
            return vadd(loc, pivot(None, sc)), None, sc

        _x_bake(li.b, F_START, F_ENTRY_END, entry)

    for li in L:
        def idle(f, li=li):
            e = _env(f)
            w = lwave(f, 2) * e
            depth = 0.9 * x * w
            sc = SCL(1.0 + VERTIGO_K * depth)
            loc = LOC(side=li.rx * 0.08 * x * w, up=li.rz * 0.08 * x * w, depth=depth)
            return vadd(loc, pivot(None, sc)), None, sc

        _x_bake(li.b, F_LOOP_START, F_IDLE_END, idle)

    off = _x_exit_cascade(L, "center", gap=0.5)
    DX = _x_exit_dur(off)
    antic = _exit_antic(DX)
    for li in L:
        x0 = F_EXIT_START + off[li.name]
        pull_rel = min(x0 + DX, x0 + antic + 2.0)

        def ex(f, x0=x0, li=li, pull_rel=pull_rel):
            k = in_cubic(seg(f, x0 + antic, x0 + DX))
            pull = smooth(seg(f, x0, x0 + antic)) * (1.0 - smooth(seg(f, x0 + antic, pull_rel)))
            depth = 5.0 * k - 0.3 * x * pull
            sc = SCL((1.0 + VERTIGO_K * depth) * _vvanish(f, x0 + 0.6 * DX, x0 + DX))
            loc = LOC(side=-li.rx * 0.8 * k, up=-li.rz * 0.8 * k, depth=depth)
            return vadd(loc, pivot(None, sc)), None, sc

        _x_bake(li.b, F_EXIT_START, F_END, ex)


# ============================================================
# REGISTER (appended after V1 - V24)
# ============================================================

VARIATIONS.extend([
    ("V25_Flag_Wave", v25_flag_wave),
    ("V26_Ocean_Swell", v26_ocean_swell),
    ("V27_Tidal_Nod", v27_tidal_nod),
    ("V28_Snake_Trail", v28_snake_trail),
    ("V29_Hello_Wave", v29_hello_wave),
    ("V30_Folding_Screen", v30_folding_screen),
    ("V31_Fly_Through", v31_fly_through),
    ("V32_Cannon_Fire", v32_cannon_fire),
    ("V33_Dolly_Rush", v33_dolly_rush),
    ("V34_Tunnel_Rush", v34_tunnel_rush),
    ("V35_Depth_Swap", v35_depth_swap),
    ("V36_Carousel_Orbit", v36_carousel_orbit),
    ("V37_Curved_Screen", v37_curved_screen),
    ("V38_Truck_Pan", v38_truck_pan),
    ("V39_Unfold_Scale", v39_unfold_scale),
    ("V40_Zoom_Steps", v40_zoom_steps),
    ("V41_Lens_Sweep", v41_lens_sweep),
    ("V42_Tsunami", v42_tsunami),
    ("V43_Crane_Down", v43_crane_down),
    ("V44_Vertigo_Zoom", v44_vertigo_zoom),
])

VARIATION_DESCRIPTIONS.update({
    25: "Flag unfurls with a travelling sine wave, flag-in-wind idle, blown away",
    26: "Drift in like buoys on the swell, slow ocean roll, surf off to the right",
    27: "Swing back upright from a bow, nodding wave, bow into the camera",
    28: "Follow-the-leader snake path in, depth slither, snake away",
    29: "Hands pop up and wave, hello wave runs through, goodbye wave + away",
    30: "Folding screen opens, panels sway in zig-zag depth, folds up",
    31: "Fly out of the distance toward the camera, depth float, race away",
    32: "Fired from the camera in the middle, cannon recoil, sucked back",
    33: "Dolly rush: block rushes in, letters trail, slow dolly, dive away",
    34: "Stream out of the vanishing point, tunnel breathing, rush past camera",
    35: "Near / far letters meet in depth, swap places, shoot apart in depth",
    36: "Turntable: whole text spins 540 deg around the vertical axis, sways, spins out",
    37: "Curved cinema screen: middle first, bends toward you, edges fold away",
    38: "Camera truck pan: parallax slide-in, camera sway, whip-pan away",
    39: "Line widens then grows from the baseline, equalizer idle, collapses",
    40: "Lens zooms in 3 clicks, zoom-click ripple, zooms out in clicks",
    41: "Magnifier reveal, lens sweeps over the text, zoom-in vanish",
    42: "Giant crest runs through the text, big swells, crest swallows letters",
    43: "Crane down from above with tilt, crane drift, crane up and away",
    44: "Dolly-zoom vertigo: spacing stretches at constant size, pump, squeeze away",
})
# ============================================================
# RIG CREATION
# ============================================================

def _get_or_create_gold_material():
    mat = bpy.data.materials.get("Win_Gold")
    if mat:
        return mat
    mat = bpy.data.materials.new("Win_Gold")
    mat.diffuse_color = (1.0, 0.75, 0.1, 1.0)
    try:
        mat.use_nodes = True      # needed before Blender 5.0, deprecated (always on) afterwards
    except Exception:
        pass
    nt = getattr(mat, "node_tree", None)
    if nt is not None:
        bsdf = nt.nodes.get("Principled BSDF")
        if bsdf is None:
            for node in nt.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf = node
                    break
        if bsdf is None:
            try:
                bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
                out = None
                for node in nt.nodes:
                    if node.type == 'OUTPUT_MATERIAL':
                        out = node
                        break
                if out is None:
                    out = nt.nodes.new("ShaderNodeOutputMaterial")
                nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
            except Exception:
                bsdf = None
        if bsdf is not None:
            for sock, val in (("Base Color", (1.0, 0.75, 0.1, 1.0)),
                              ("Metallic", 1.0), ("Roughness", 0.2)):
                try:
                    bsdf.inputs[sock].default_value = val
                except Exception:
                    pass
    return mat


def _find_view3d_override():
    """A real 3D Viewport, so bpy.ops also works when we run from a timer."""
    try:
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            return {"window": window, "screen": screen,
                                    "area": area, "region": region}
    except Exception:
        pass
    return None


def _run_op(op, **kwargs):
    override = _find_view3d_override()
    if override:
        with bpy.context.temp_override(**override):
            return op(**kwargs)
    return op(**kwargs)


def _ensure_object_mode():
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != 'OBJECT':
            _run_op(bpy.ops.object.mode_set, mode='OBJECT')
    except Exception as err:
        print("[Win Anim] could not switch to Object Mode:", err)


def _restore_selection(view_layer, selected, active):
    try:
        for o in view_layer.objects:
            o.select_set(False)
        for o in selected:
            try:
                o.select_set(True)
            except Exception:
                pass
        if active is not None:
            view_layer.objects.active = active
    except Exception:
        pass


def _rig_ok(rig_name):
    obj = bpy.data.objects.get(rig_name)
    return bool(obj is not None and obj.type == 'ARMATURE' and len(obj.data.bones) > 0)


def _layout(word):
    """Layout of a text: one row per word, the first word on top, every word centered on X = 0.
    [{"word": "MASSIVE", "z": 0.625, "cx": 0.0, "letters": [("M", -3.0), ...]}, ...]
    With STACK_WORDS = False all words sit side by side on one line instead."""
    words = [w for w in word.split() if w]
    if not words:
        return []
    rows = []
    if STACK_WORDS and len(words) > 1:
        count = len(words)
        for k, w in enumerate(words):
            z = ((count - 1) / 2.0 - k) * LINE_HEIGHT
            xs = [(i - (len(w) - 1) / 2.0) * LETTER_SPACING for i in range(len(w))]
            rows.append({"word": w.upper(), "z": z, "letters": [[c.upper(), x] for c, x in zip(w, xs)]})
    else:
        x = 0.0
        for w in words:
            items = []
            for c in w:
                items.append([c.upper(), x])
                x += LETTER_SPACING
            rows.append({"word": w.upper(), "z": 0.0, "letters": items})
            x += WORD_GAP
        allx = [it[1] for r in rows for it in r["letters"]]
        shift = (min(allx) + max(allx)) / 2.0
        for r in rows:
            for it in r["letters"]:
                it[1] -= shift
    for r in rows:
        xs = [it[1] for it in r["letters"]]
        r["cx"] = (min(xs) + max(xs)) / 2.0
        r["letters"] = [(c, x) for c, x in r["letters"]]
    return rows


def _make_letter_object(rig_name, bone_name, ch, x, zc, col):
    """One letter as a mesh object (standing up, facing the front view)."""
    prefix = rig_name.replace("_ARMATURE", "").strip().replace(" ", "_")
    obj_name = "%s_%s_MESH" % (prefix, bone_name.replace("_BONE", ""))     # unique per rig
    curve = bpy.data.curves.new("%s_%s_txt" % (rig_name, bone_name), 'FONT')
    curve.body = ch
    curve.size = LETTER_SIZE
    curve.align_x = 'CENTER'
    curve.align_y = 'CENTER'
    curve.extrude = LETTER_EXTRUDE
    tmp = bpy.data.objects.new(obj_name + "_tmp", curve)
    col.objects.link(tmp)
    loc = (x, 0.0, zc)
    rot = (math.radians(90.0), 0.0, 0.0)
    tmp.location = loc
    tmp.rotation_euler = rot

    # 1) data API: evaluate the text and copy it into a mesh (no selection / mode needed)
    mesh = None
    try:
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(depsgraph), depsgraph=depsgraph)
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError("the text produced an empty mesh")
    except Exception as err:
        print("[Win Anim] text -> mesh (data API) failed: %s - trying Convert operator" % err)
        if mesh is not None:
            try:
                bpy.data.meshes.remove(mesh)
            except Exception:
                pass
            mesh = None

    if mesh is not None:
        bpy.data.objects.remove(tmp, do_unlink=True)
        try:
            bpy.data.curves.remove(curve)
        except Exception:
            pass
        mesh.name = obj_name
        obj = bpy.data.objects.new(obj_name, mesh)
        col.objects.link(obj)
        obj.location = loc
        obj.rotation_euler = rot
        return obj

    # 2) fallback: the classic Convert operator
    try:
        view_layer = bpy.context.view_layer
        for o in view_layer.objects:
            o.select_set(False)
        tmp.select_set(True)
        view_layer.objects.active = tmp
        _run_op(bpy.ops.object.convert, target='MESH')
        tmp.name = obj_name
        return tmp
    except Exception:
        try:
            bpy.data.objects.remove(tmp, do_unlink=True)
            bpy.data.curves.remove(curve)
        except Exception:
            pass
        raise


def _do_create_rig(rig_name, word):
    """Build ONE armature for a whole text, plus the letter meshes (deformed by the armature),
    in their own collection:
        GLOBAL_MASTER > <WORD>_ROOT (one parent bone per word) > one bone per letter
    The first word is on top, the next word below it (MASSIVE over WIN).
    Returns True when the rig exists afterwards. Raises RuntimeError (after cleaning
    up, so no half-built rig stays in the scene) when something goes wrong."""
    existing = bpy.data.objects.get(rig_name)
    if existing is not None:
        if existing.type == 'ARMATURE' and len(existing.data.bones) == 0:
            # empty leftover of an earlier failed attempt -> rebuild it
            old_data = existing.data
            bpy.data.objects.remove(existing, do_unlink=True)
            try:
                bpy.data.armatures.remove(old_data)
            except Exception:
                pass
        else:
            return True

    rows = _layout(word)
    if not rows:
        raise RuntimeError("no letters found in '%s'" % word)

    counts, wcounts = {}, {}
    for r in rows:
        wcounts[r["word"]] = wcounts.get(r["word"], 0) + 1
        for ch, _x in r["letters"]:
            counts[ch] = counts.get(ch, 0) + 1
    seen, wseen, plan = {}, {}, []
    for r in rows:
        items = []
        for ch, x in r["letters"]:
            if counts[ch] > 1:
                seen[ch] = seen.get(ch, 0) + 1
                bone_name = "%s%d_BONE" % (ch, seen[ch])
            else:
                bone_name = "%s_BONE" % ch
            items.append((bone_name, ch, x))
        if wcounts[r["word"]] > 1:
            wseen[r["word"]] = wseen.get(r["word"], 0) + 1
            root_name = "%s%d_ROOT" % (r["word"], wseen[r["word"]])
        else:
            root_name = "%s_ROOT" % r["word"]
        plan.append((root_name, items, r["z"], r["cx"]))
    total_letters = sum(len(p[1]) for p in plan)
    z_master = sum(p[2] for p in plan) / len(plan) + LETTER_CENTER

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    col_name = rig_name.replace("_ARMATURE", "").strip() or rig_name
    col = bpy.data.collections.get(col_name)
    col_is_new = col is None
    if col is None:
        col = bpy.data.collections.new(col_name)
    if not any(c.name == col.name for c in scene.collection.children_recursive):
        scene.collection.children.link(col)

    arm_data = bpy.data.armatures.new(rig_name)
    arm_obj = bpy.data.objects.new(rig_name, arm_data)
    col.objects.link(arm_obj)
    created = [arm_obj]

    prev_active = view_layer.objects.active
    prev_selected = [o for o in view_layer.objects if o.select_get()]

    try:
        _ensure_object_mode()
        for o in prev_selected:
            o.select_set(False)
        arm_obj.select_set(True)
        view_layer.objects.active = arm_obj

        # Bones point along +Y, so their local axes line up with the world axes:
        # X = side, Y = depth, Z = up (this is what the animations expect).
        _run_op(bpy.ops.object.mode_set, mode='EDIT')
        edit_bones = arm_data.edit_bones
        master = edit_bones.new("GLOBAL_MASTER")
        master.head = (0.0, 0.0, z_master)
        master.tail = (0.0, BONE_LENGTH, z_master)
        for root_name, items, z, cx in plan:
            rb = edit_bones.new(root_name)                    # the parent bone of this word
            rb.head = (cx, 0.0, z + LETTER_CENTER)
            rb.tail = (cx, BONE_LENGTH, z + LETTER_CENTER)
            rb.parent = master
            rb.use_connect = False
            for bone_name, ch, x in items:                    # one bone per letter
                b = edit_bones.new(bone_name)
                b.head = (x, 0.0, z)
                b.tail = (x, BONE_LENGTH, z)
                b.parent = rb
                b.use_connect = False
        _run_op(bpy.ops.object.mode_set, mode='OBJECT')
        for pb in arm_obj.pose.bones:          # new bones are QUATERNION - the actions key Euler XYZ
            pb.rotation_mode = 'XYZ'

        gold = _get_or_create_gold_material()
        for root_name, items, z, cx in plan:
            for bone_name, ch, x in items:
                obj = _make_letter_object(rig_name, bone_name, ch, x, z + LETTER_CENTER, col)
                created.append(obj)
                if not obj.data.materials:
                    obj.data.materials.append(gold)
                mod = obj.modifiers.new("Armature", 'ARMATURE')
                mod.object = arm_obj
                vg = obj.vertex_groups.new(name=bone_name)
                vg.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
                obj.parent = arm_obj
                obj.parent_type = 'OBJECT'
                obj.matrix_parent_inverse.identity()
    except Exception as err:
        traceback.print_exc()
        try:
            if arm_obj.mode != 'OBJECT':
                _run_op(bpy.ops.object.mode_set, mode='OBJECT')
        except Exception:
            pass
        for o in reversed(created):
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
        try:
            bpy.data.armatures.remove(arm_data)
        except Exception:
            pass
        if col_is_new:
            try:
                if len(col.objects) == 0 and len(col.children) == 0:
                    bpy.data.collections.remove(col)
            except Exception:
                pass
        _restore_selection(view_layer, prev_selected, prev_active)
        raise RuntimeError(str(err))

    _restore_selection(view_layer, prev_selected, prev_active)
    print("[Win Anim] Created '%s': %d words, %d letters, one armature, in collection '%s'"
          % (rig_name, len(plan), total_letters, col_name))
    return True


def _redraw_ui():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                area.tag_redraw()
    except Exception:
        pass


LAST_ERROR = {"msg": ""}     # shown in the panel until the next successful create


def _set_error(msg):
    LAST_ERROR["msg"] = msg or ""


_CREATE_PROPS = ("create_massive", "create_big", "create_huge")
_RIG_PROPS = ("rig_massive", "rig_big", "rig_huge")


def _schedule_create(idx):
    """Objects can not be created inside a property-update callback, so do it
    a moment later from a timer."""
    def _job():
        try:
            _do_create_rig(RIG_NAMES[idx], RIG_WORDS[idx])
            _set_error("")
            props = getattr(bpy.context.scene, "win_anim", None)
            if props is not None:
                setattr(props, _RIG_PROPS[idx], True)
        except Exception as err:
            _set_error("%s: %s" % (RIG_WORDS[idx], err))
            print("[Win Anim] could not create '%s': %s" % (RIG_NAMES[idx], err))
        _redraw_ui()
        return None
    bpy.app.timers.register(_job, first_interval=0.05)


def _make_create_cb(idx):
    def _cb(self, context):
        try:
            if not getattr(self, _CREATE_PROPS[idx]):
                return
            if _rig_ok(RIG_NAMES[idx]):
                setattr(self, _RIG_PROPS[idx], True)
                return
            _schedule_create(idx)
        except Exception as err:
            print("[Win Anim] create callback:", err)
    return _cb



# ============================================================
# FCURVE / STYLE / BUILD CORE
# ============================================================

FCURVE_PATH = re.compile(r'^pose\.bones\["(.+)"\]\.(location|rotation_euler|scale)$')
ADDON_VERSION = ".".join(str(v) for v in bl_info["version"])


def iter_fcurves(obj):
    """All F-Curves of the object's action (works with the slotted / layered
    Actions of Blender 4.4+ / 5.x and with the old flat Actions)."""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return
    action = ad.action
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        for fc in legacy:
            yield fc
        return
    slot = getattr(ad, "action_slot", None)
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != 'KEYFRAME':
                continue
            if slot is not None:
                bag = strip.channelbag(slot)
                bags = [bag] if bag is not None else []
            else:
                bags = list(strip.channelbags)
            for bag in bags:
                for fc in bag.fcurves:
                    yield fc


def _style_point(kp, style, frame, next_frame, change=0.0):
    interp, ease, period = style
    if interp != 'BEZIER':
        kp.interpolation = interp
    if ease != 'AUTO':
        kp.easing = ease
    if interp == 'ELASTIC':
        # Blender damps the wobble away when amplitude < |change| (amplitude 0 = no wobble at all,
        # the 1.x bug). amplitude == |change| gives the classic elastic curve.
        kp.amplitude = abs(change)
        if next_frame is not None:
            kp.period = max(1.0, period * (next_frame - frame))
    elif interp == 'BACK':
        try:
            kp.back = BACK_OVERSHOOT
        except Exception:
            pass


def _compress(points):
    """Drop keys in the middle of flat runs (same value, same smooth / step style on both
    sides). The curve stays exactly the same; actions get much smaller."""
    out, n = [], len(points)
    for k, p in enumerate(points):
        if 0 < k < n - 1 and p[0] not in _BORDER_SET:
            a, b = points[k - 1], points[k + 1]
            if (p[2] == a[2] and p[2] in (SMOOTH, STEP) and abs(a[1] - p[1]) < 1e-9
                    and abs(b[1] - p[1]) < 1e-9):
                continue
        out.append(p)
    return out


def _new_fcurve(bag, path, idx, group):
    try:
        return bag.fcurves.new(data_path=path, index=idx, group_name=group)
    except TypeError:
        return bag.fcurves.new(path, index=idx)


def _fill_fcurve(fc, points):
    """Write (frame, value, style) points into an empty F-Curve in one bulk call."""
    kps = fc.keyframe_points
    kps.add(len(points))
    kps.foreach_set("co", [c for p in points for c in (float(p[0]), float(p[1]))])
    for i, kp in enumerate(kps):
        kp.interpolation = 'BEZIER'
        kp.handle_left_type = 'AUTO_CLAMPED'
        kp.handle_right_type = 'AUTO_CLAMPED'
        st = points[i][2]
        if st is not SMOOTH and st != SMOOTH:
            nxt = points[i + 1] if i + 1 < len(points) else None
            _style_point(kp, st, points[i][0], nxt[0] if nxt else None,
                         (nxt[1] - points[i][1]) if nxt else 0.0)
    fc.update()


def _write_direct(armature, name):
    """Write every collected key straight into the F-Curves of a new slotted Action.
    This does not depend on the user's keying settings ("Only Insert Needed" etc.)."""
    esc = getattr(bpy.utils, "escape_identifier", None)
    action = bpy.data.actions.new(name)
    try:
        slot = action.slots.new(id_type='OBJECT', name=armature.name)
        layer = action.layers.new("Layer")
        strip = layer.strips.new(type='KEYFRAME')
        bag = strip.channelbag(slot, ensure=True)
        for bkey in sorted(TRACK):
            bone_name, prop = bkey
            frames = TRACK[bkey]
            path = 'pose.bones["%s"].%s' % (esc(bone_name) if esc else bone_name, prop)
            order = sorted(frames)
            for idx in range(3):
                pts = _compress([(f, frames[f][0][idx], frames[f][1]) for f in order])
                _fill_fcurve(_new_fcurve(bag, path, idx, bone_name), pts)
        ad = armature.animation_data or armature.animation_data_create()
        ad.action = action
        try:
            ad.action_slot = slot
        except Exception as err:
            print("[Win Anim] could not assign the action slot:", err)
        if getattr(ad, "action_slot", None) is None:
            raise RuntimeError("the action has no slot assigned")
    except Exception:
        try:
            if armature.animation_data:
                armature.animation_data.action = None
        except Exception:
            pass
        try:
            bpy.data.actions.remove(action)
        except Exception:
            pass
        raise
    return action


def _apply_styles_fallback(armature):
    for fc in iter_fcurves(armature):
        m = FCURVE_PATH.match(fc.data_path)
        if not m:
            continue
        frames = TRACK.get((m.group(1), m.group(2)))
        if not frames:
            continue
        pts = fc.keyframe_points
        for idx in range(len(pts)):
            kp = pts[idx]
            frame = int(round(kp.co[0]))
            item = frames.get(frame)
            if item is None:
                continue
            nxt = int(round(pts[idx + 1].co[0])) if idx + 1 < len(pts) else None
            change = (pts[idx + 1].co[1] - kp.co[1]) if idx + 1 < len(pts) else 0.0
            _style_point(kp, item[1], frame, nxt, change)
        fc.update()


def _write_keyframe_insert(armature, name):
    """Fallback: classic keyframe_insert (used only if _write_direct fails)."""
    pbs = armature.pose.bones
    ad = armature.animation_data
    if ad is not None and ad.action is not None:
        ad.action = None                 # so keyframe_insert starts a fresh Action
    keyed = []
    try:
        for bkey in sorted(TRACK):
            bone_name, prop = bkey
            pb = pbs.get(bone_name)
            if pb is None:
                continue
            keyed.append(pb)
            pb.rotation_mode = 'XYZ'
            frames = TRACK[bkey]
            for f in sorted(frames):
                setattr(pb, prop, frames[f][0])
                pb.keyframe_insert(data_path=prop, frame=f, group=bone_name)
        ad = armature.animation_data
        if ad is None or ad.action is None:
            raise RuntimeError("keyframe_insert did not create an Action")
        action = ad.action
        action.name = name
        _apply_styles_fallback(armature)
    except Exception:
        ad = armature.animation_data
        partial = ad.action if ad is not None else None
        if partial is not None:
            ad.action = None
            try:
                bpy.data.actions.remove(partial)
            except Exception:
                pass
        raise
    reset_pose(keyed)
    return action


def write_action(armature, name):
    """Write the collected keys into a NEW action called `name`.
    Other actions are never touched."""
    try:
        action = _write_direct(armature, name)
    except Exception as err:
        print("[Win Anim] direct F-Curve writing failed (%s) - using keyframe_insert instead" % err)
        traceback.print_exc()
        action = _write_keyframe_insert(armature, name)
    action.use_fake_user = True
    return action


def reset_pose(pose_bones):
    for pb in pose_bones:
        pb.rotation_mode = 'XYZ'
        pb.location = REST_LOC
        pb.rotation_euler = REST_ROT
        pb.scale = REST_SCL


def build_variation(armature, master, letters, action_name, builder):
    bones = [master] + CTX["roots"] + letters
    reset_pose(bones)
    TRACK.clear()
    builder(master, letters)
    ensure_boundaries(bones)      # frames 30 / 31 / 85 / 86 = identical rest pose, all channels present
    return write_action(armature, action_name)


def activate(armature, action):
    ad = armature.animation_data or armature.animation_data_create()
    ad.action = action
    try:
        slots = getattr(action, "slots", None)
        if slots is not None and len(slots) and getattr(ad, "action_slot", None) is None:
            want = "OB" + armature.name
            pick = slots[0]
            for s in slots:
                if s.identifier == want:
                    pick = s
                    break
            ad.action_slot = pick
    except Exception as err:
        print("[Win Anim] could not assign the action slot:", err)
    try:
        for pb in armature.pose.bones:   # Euler channels only work in an Euler rotation mode
            if pb.rotation_mode != 'XYZ':
                pb.rotation_mode = 'XYZ'
    except Exception:
        pass


# ============================================================
# ADDON: PROPERTIES
# ============================================================

_VARIATION_ITEMS = [(str(i), "%d  %s" % (i, v.split("_", 1)[1].replace("_", " ")), VARIATION_DESCRIPTIONS.get(i, ""))
                    for i, (v, _) in enumerate(VARIATIONS, start=1)]


class WinAnimProps(bpy.types.PropertyGroup):
    rig_massive: bpy.props.BoolProperty(name="MASSIVE WIN", default=True)
    rig_big: bpy.props.BoolProperty(name="BIG WIN", default=True)
    rig_huge: bpy.props.BoolProperty(name="HUGE WIN", default=True)

    create_massive: bpy.props.BoolProperty(
        name="MASSIVE WIN", default=False, update=_make_create_cb(0))
    create_big: bpy.props.BoolProperty(
        name="BIG WIN", default=False, update=_make_create_cb(1))
    create_huge: bpy.props.BoolProperty(
        name="HUGE WIN", default=False, update=_make_create_cb(2))

    fps: bpy.props.IntProperty(name="FPS", default=30, min=1, max=120,
                               description="The timings are designed in frames at 30 fps")

    auto_axes: bpy.props.BoolProperty(
        name="Auto-detect bone axes", default=True,
        description="Find out which bone axis points up / sideways / to the camera, so the "
                    "animations also work on rigs whose bones are not aligned with the world axes")

    intensity: bpy.props.FloatProperty(
        name="Intensity", default=1.0, min=0.0, max=2.0, step=10,
        description="Exaggeration: scales overshoots, squash & stretch, tilts and shakes "
                    "(1.0 = as designed). Rebuild to apply")

    stagger: bpy.props.FloatProperty(
        name="Stagger", default=1.0, min=0.0, max=2.0, step=10,
        description="Spacing of the letter cascade (0 = all letters together, 1.0 = as designed). "
                    "Long cascades are squeezed so every letter still finishes in its section. "
                    "Rebuild to apply")

    variation: bpy.props.EnumProperty(name="Variation", items=_VARIATION_ITEMS, default="1")

    show_all: bpy.props.BoolProperty(name="Show all variations", default=False)


# ============================================================
# ADDON: HELPERS
# ============================================================

_ICONS = None


def _icon(name):
    """Return the icon name if this Blender version has it, otherwise no icon."""
    global _ICONS
    if _ICONS is None:
        try:
            _ICONS = set(bpy.types.UILayout.bl_rna.functions["label"]
                         .parameters["icon"].enum_items.keys())
        except Exception:
            _ICONS = set()
    return name if (not _ICONS or name in _ICONS) else 'NONE'


def _enabled_rigs(scene):
    props = scene.win_anim
    out = []
    for name, flag in [(RIG_NAMES[0], props.rig_massive),
                       (RIG_NAMES[1], props.rig_big),
                       (RIG_NAMES[2], props.rig_huge)]:
        if flag:
            rig = bpy.data.objects.get(name)
            if rig and rig.type == 'ARMATURE':
                out.append(rig)
    return out


_MARKERS = (("ENTRY", F_START), ("LOOP", F_LOOP_START), ("EXIT", F_EXIT_START))


def _ensure_scene(scene, fps):
    scene.frame_start = F_START
    scene.frame_end = F_END
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    try:
        names = {m.name for m in scene.timeline_markers}
        for label, frame in _MARKERS:
            if label not in names:
                scene.timeline_markers.new(label, frame=frame)
    except Exception as err:
        print("[Win Anim] could not add timeline markers:", err)


class _RigError(RuntimeError):
    """The rig itself can not be animated (no GLOBAL_MASTER / no letter bones)."""


def _our_action_name(rig, vname):
    return "%s_%s" % (rig.name, vname)


def _our_names(rig):
    return {_our_action_name(rig, v) for v, _ in VARIATIONS}


def _protect_foreign_action(rig):
    """Building replaces the rig's active action. If that is one of YOUR actions, switch its
    fake user on so it stays in the file instead of silently disappearing."""
    ad = rig.animation_data
    act = ad.action if ad is not None else None
    if act is None or act.name in _our_names(rig) or act.name.endswith("__new"):
        return
    if not act.use_fake_user:
        act.use_fake_user = True
        print("[Win Anim] '%s' was the active action - fake user switched on so it is kept." % act.name)


def _action_fcurves(action):
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != 'KEYFRAME':
                continue
            for bag in strip.channelbags:
                out.extend(list(bag.fcurves))
    return out


def _verify_action(action, expected_fcurves):
    """Problems of one action (empty list = fine): number of F-Curves, key range and
    identical poses on the clip borders (entry end / loop start / loop end / exit start)."""
    problems = []
    curves = _action_fcurves(action)
    if not curves:
        problems.append("no F-Curves")
    elif len(curves) != expected_fcurves:
        problems.append("%d F-Curves instead of %d (built with an older version? rebuild)"
                        % (len(curves), expected_fcurves))
    lo = hi = None
    bad = 0
    for fc in curves:
        pts = {int(round(kp.co[0])): kp.co[1] for kp in fc.keyframe_points}
        if not pts:
            bad += 1
            continue
        lo = min(pts) if lo is None else min(lo, min(pts))
        hi = max(pts) if hi is None else max(hi, max(pts))
        if any(f not in pts for f in _BORDER_FRAMES):
            bad += 1
        else:
            vals = [pts[f] for f in _BORDER_FRAMES]
            if max(vals) - min(vals) > 1e-6:
                bad += 1
    if lo is not None and lo < F_START:
        problems.append("keys before frame %d" % F_START)
    if hi is not None and hi > F_END:
        problems.append("keys after frame %d" % F_END)
    if bad:
        problems.append("%d channel(s) without the same pose on frames %d / %d / %d / %d"
                        % ((bad,) + _BORDER_FRAMES))
    return problems


def _verify_rig(rig, repair=True):
    """-> ({variation: [problems]}, [notes]). A variation that was not built shows ['not built']."""
    result, notes = {}, []
    groups = get_word_groups(rig)
    roots = [r for r, _ in groups if r is not None]
    expected = (1 + len(roots) + sum(len(ls) for _, ls in groups)) * 9
    for vname, _ in VARIATIONS:
        action = bpy.data.actions.get(_our_action_name(rig, vname))
        if action is None:
            result[vname] = ["not built"]
            continue
        problems = _verify_action(action, expected)
        if not action.use_fake_user:
            if repair:
                try:
                    action.use_fake_user = True
                except Exception:
                    pass
            if action.use_fake_user:
                notes.append("%s: fake user was off - switched on" % action.name)
            else:
                problems.append("fake user is off (the action would be lost when saving)")
        result[vname] = problems
    return result, notes


def _build_one(rig, var_index, auto_axes=True, exag=1.0, stagger=1.0):
    """Build one variation on one rig. The new action is built under a temporary name and
    replaces the old one only after it succeeded - a failed build never costs you an action."""
    vname, builder = VARIATIONS[var_index - 1]
    master = rig.pose.bones.get("GLOBAL_MASTER")
    if master is None:
        raise _RigError("no GLOBAL_MASTER bone found")
    groups = get_word_groups(rig)
    letters = [b for _r, ls in groups for b in ls]
    if not letters:
        raise _RigError("no letter bones (*_BONE) found")
    final = _our_action_name(rig, vname)
    tmp = final + "__new"
    stale = bpy.data.actions.get(tmp)
    if stale is not None:
        bpy.data.actions.remove(stale)
    _protect_foreign_action(rig)
    ad = rig.animation_data
    previous = ad.action if ad is not None else None
    prepare_context(rig, letters, master, auto_axes, groups=groups, exag=exag, stagger=stagger)
    try:
        action = build_variation(rig, master, letters, tmp, builder)
    except Exception:
        leftover = bpy.data.actions.get(tmp)
        if leftover is not None:
            ad = rig.animation_data
            if ad is not None and ad.action == leftover:
                ad.action = None
            bpy.data.actions.remove(leftover)
        if previous is not None:
            activate(rig, previous)
        raise
    old = bpy.data.actions.get(final)
    if old is not None and old is not action:
        try:
            old.user_remap(action)        # NLA strips / other users now point at the new action
        except Exception as err:
            print("[Win Anim] could not remap the users of '%s': %s" % (final, err))
        old.use_fake_user = False
        bpy.data.actions.remove(old)
    action.name = final
    action.use_fake_user = True
    action["win_anim"] = "v%s intensity=%.2f stagger=%.2f" % (ADDON_VERSION, exag, stagger)
    if action.name != final:
        raise RuntimeError("could not name the action '%s' (the name is already in use)" % final)
    activate(rig, action)
    return True


def _apply_one(rig, var_index):
    vname, _ = VARIATIONS[var_index - 1]
    action = bpy.data.actions.get("%s_%s" % (rig.name, vname))
    if not action:
        return False
    activate(rig, action)
    rig["Animation"] = vname
    return True


def _built_count(vname):
    return sum(1 for n in RIG_NAMES if bpy.data.actions.get("%s_%s" % (n, vname)))


def _rig_total():
    return sum(1 for n in RIG_NAMES if _rig_ok(n))


def _build_args(props):
    return props.auto_axes, props.intensity, props.stagger


# ============================================================
# ADDON: OPERATORS
# ============================================================

class WIN_ANIM_OT_build_all(bpy.types.Operator):
    bl_idname = "win_anim.build_all"
    bl_label = "Build All Variations"
    bl_description = ("Build all %d variations on the enabled rigs, then verify that all actions "
                      "per rig were kept safely" % len(VARIATIONS))
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rigs = _enabled_rigs(context.scene)
        if not rigs:
            self.report({'WARNING'}, "No rigs enabled/found")
            return {'CANCELLED'}
        props = context.scene.win_anim
        _ensure_scene(context.scene, props.fps)
        parts, all_ok = [], True
        for rig in rigs:
            label = rig.name.replace("_ARMATURE", "")
            prev = None
            if rig.animation_data and rig.animation_data.action:
                prev = rig.animation_data.action.name
            for i in range(1, len(VARIATIONS) + 1):
                try:
                    _build_one(rig, i, *_build_args(props))
                except _RigError as err:
                    self.report({'ERROR'}, "%s: %s" % (rig.name, err))
                    break
                except Exception as err:
                    traceback.print_exc()
                    self.report({'ERROR'}, "%s V%d: %s" % (rig.name, i, err))
            # verify, and rebuild anything that is missing / not right (once)
            problems, notes = _verify_rig(rig)
            retry = [i for i, (v, _) in enumerate(VARIATIONS, start=1) if problems[v]]
            if retry and rig.pose.bones.get("GLOBAL_MASTER") is not None and get_letter_bones(rig):
                for i in retry:
                    try:
                        _build_one(rig, i, *_build_args(props))
                    except Exception:
                        traceback.print_exc()
                problems, notes = _verify_rig(rig)
            ok = sum(1 for v, _ in VARIATIONS if not problems[v])
            parts.append("%s %d/%d" % (label, ok, len(VARIATIONS)))
            for v, _ in VARIATIONS:
                if problems[v]:
                    all_ok = False
                    self.report({'WARNING'}, "%s %s: %s" % (label, v, "; ".join(problems[v])))
            # put back the action that was active before
            if prev:
                act = bpy.data.actions.get(prev)
                if act is not None:
                    activate(rig, act)
        if all_ok:
            self.report({'INFO'}, "Build All: " + ", ".join(parts) + " - all actions verified and kept (fake user on)")
        else:
            self.report({'WARNING'}, "Build All: " + ", ".join(parts) + " - see the warnings")
        return {'FINISHED'}


class WIN_ANIM_OT_build_one(bpy.types.Operator):
    bl_idname = "win_anim.build_one"
    bl_label = "Build This Variation"
    bl_description = "(Re)build this variation on the enabled rigs with the current Intensity / Stagger"
    bl_options = {'REGISTER', 'UNDO'}
    variation: bpy.props.IntProperty(default=1, min=1, max=len(VARIATIONS))

    def execute(self, context):
        rigs = _enabled_rigs(context.scene)
        if not rigs:
            self.report({'WARNING'}, "No rigs enabled/found")
            return {'CANCELLED'}
        props = context.scene.win_anim
        _ensure_scene(context.scene, props.fps)
        count = 0
        for rig in rigs:
            try:
                if _build_one(rig, self.variation, *_build_args(props)):
                    rig["Animation"] = VARIATIONS[self.variation - 1][0]
                    count += 1
            except Exception as err:
                traceback.print_exc()
                self.report({'ERROR'}, "%s: %s" % (rig.name, err))
        self.report({'INFO'}, f"Built V{self.variation} on {count} rig(s)")
        return {'FINISHED'}


class WIN_ANIM_OT_apply(bpy.types.Operator):
    bl_idname = "win_anim.apply"
    bl_label = "Apply Variation"
    bl_description = "Apply to enabled rigs (auto-builds if missing)"
    bl_options = {'REGISTER', 'UNDO'}
    variation: bpy.props.IntProperty(default=1, min=1, max=len(VARIATIONS))

    def execute(self, context):
        rigs = _enabled_rigs(context.scene)
        if not rigs:
            self.report({'WARNING'}, "No rigs enabled/found")
            return {'CANCELLED'}
        props = context.scene.win_anim
        _ensure_scene(context.scene, props.fps)
        vname, _ = VARIATIONS[self.variation - 1]
        count = 0
        for rig in rigs:
            try:
                if not bpy.data.actions.get("%s_%s" % (rig.name, vname)):
                    _build_one(rig, self.variation, *_build_args(props))
                if _apply_one(rig, self.variation):
                    count += 1
            except Exception as err:
                traceback.print_exc()
                self.report({'ERROR'}, "%s: %s" % (rig.name, err))
        context.scene.frame_set(F_START)
        self.report({'INFO'}, f"Applied V{self.variation} to {count} rig(s)")
        return {'FINISHED'}


class WIN_ANIM_OT_clear(bpy.types.Operator):
    bl_idname = "win_anim.clear"
    bl_label = "Clear All Actions"
    bl_description = "Remove all Win Animation actions from enabled rigs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rigs = _enabled_rigs(context.scene)
        removed = 0
        for rig in rigs:
            ad = rig.animation_data
            if ad is not None and ad.action is not None and ad.action.name in _our_names(rig):
                ad.action = None          # only our own action is un-assigned (NLA / drivers stay)
            for vname, _ in VARIATIONS:
                action = bpy.data.actions.get(_our_action_name(rig, vname))
                if action:
                    action.use_fake_user = False
                    bpy.data.actions.remove(action)
                    removed += 1
        self.report({'INFO'}, f"Removed {removed} action(s)")
        return {'FINISHED'}


class WIN_ANIM_OT_verify(bpy.types.Operator):
    bl_idname = "win_anim.verify"
    bl_label = "Verify Actions"
    bl_description = ("Check that all actions exist on every enabled rig: F-Curves, fake user, "
                      "and identical poses on frames 30 / 31 / 85 / 86")
    bl_options = {'REGISTER'}

    def execute(self, context):
        rigs = _enabled_rigs(context.scene)
        if not rigs:
            self.report({'WARNING'}, "No rigs enabled/found")
            return {'CANCELLED'}
        parts, all_ok = [], True
        for rig in rigs:
            label = rig.name.replace("_ARMATURE", "")
            problems, notes = _verify_rig(rig, repair=True)
            built = sum(1 for v, _ in VARIATIONS if problems[v] != ["not built"])
            ok = sum(1 for v, _ in VARIATIONS if not problems[v])
            parts.append("%s %d/%d" % (label, ok, len(VARIATIONS)))
            for n in notes:
                self.report({'INFO'}, n)
            for v, _ in VARIATIONS:
                if problems[v] and problems[v] != ["not built"]:
                    all_ok = False
                    self.report({'WARNING'}, "%s %s: %s" % (label, v, "; ".join(problems[v])))
            if built < len(VARIATIONS):
                all_ok = False
        if all_ok:
            self.report({'INFO'}, "Verify: " + ", ".join(parts) + " - all actions OK")
        else:
            self.report({'WARNING'}, "Verify: " + ", ".join(parts) + " (run Build All to fix missing ones)")
        return {'FINISHED'}


class WIN_ANIM_OT_reset_pose(bpy.types.Operator):
    bl_idname = "win_anim.reset_pose"
    bl_label = "Reset Pose"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rigs = _enabled_rigs(context.scene)
        for rig in rigs:
            master = rig.pose.bones.get("GLOBAL_MASTER")
            groups = get_word_groups(rig)
            roots = [r for r, _ in groups if r is not None]
            letters = [b for _r, ls in groups for b in ls]
            if master:
                reset_pose([master] + roots + letters)
        self.report({'INFO'}, f"Reset {len(rigs)} rig(s)")
        return {'FINISHED'}


class WIN_ANIM_OT_create_missing(bpy.types.Operator):
    bl_idname = "win_anim.create_missing"
    bl_label = "Create All Missing"
    bl_description = "Create rigs for all checked Create boxes that don't exist yet"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.win_anim
        created = 0
        _set_error("")
        for idx in range(3):
            if not getattr(props, _CREATE_PROPS[idx]):
                continue
            if _rig_ok(RIG_NAMES[idx]):
                continue
            try:
                if _do_create_rig(RIG_NAMES[idx], RIG_WORDS[idx]):
                    created += 1
                    setattr(props, _RIG_PROPS[idx], True)
            except Exception as err:
                _set_error("%s: %s" % (RIG_WORDS[idx], err))
                self.report({'ERROR'}, "%s: %s" % (RIG_WORDS[idx], err))
        if created:
            self.report({'INFO'}, f"Created {created} rig(s)")
        else:
            self.report({'INFO'}, "No new rigs created")
        return {'FINISHED'}


_SECTIONS = {
    "ENTRY": (F_START, F_ENTRY_END),
    "LOOP": (F_LOOP_START, F_IDLE_END - 1),     # 85 == 31, so play 31..84 for a seamless repeat
    "EXIT": (F_EXIT_START, F_END),
    "ALL": (F_START, F_END),
}


class WIN_ANIM_OT_preview(bpy.types.Operator):
    bl_idname = "win_anim.preview"
    bl_label = "Preview Section"
    bl_description = "Set the playback range to one section and play it (Loop repeats seamlessly)"
    bl_options = {'REGISTER'}
    section: bpy.props.EnumProperty(items=[(k, k.title(), "") for k in _SECTIONS], default="ALL")

    def execute(self, context):
        scene = context.scene
        a, b = _SECTIONS[self.section]
        scene.frame_start, scene.frame_end = a, b
        scene.frame_set(a)
        try:
            screen = context.screen
            if screen is not None and not screen.is_animation_playing:
                override = _find_view3d_override()
                if override:
                    with context.temp_override(**override):
                        bpy.ops.screen.animation_play()
                else:
                    bpy.ops.screen.animation_play()
        except Exception as err:
            print("[Win Anim] could not start playback:", err)
        self.report({'INFO'}, "Playing %s (%d-%d)" % (self.section.title(), a, b))
        return {'FINISHED'}


# ============================================================
# ADDON: PANEL
# ============================================================

def _status_icon(row, vname, rig_total):
    built = _built_count(vname)
    if rig_total > 0 and built == rig_total:
        row.label(text="", icon=_icon('CHECKMARK'))
    elif built > 0:
        row.label(text=f"{built}/{rig_total}")
    else:
        row.label(text="", icon=_icon('BLANK1'))


class VIEW3D_PT_win_animations(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Win Anim"
    bl_label = "Win Animations"

    def draw(self, context):
        layout = self.layout
        props = context.scene.win_anim

        # ==================== Create Rig ====================
        box = layout.box()
        box.label(text="Create Rig", icon=_icon('ADD'))
        sub = box.box()
        sub.label(text="Wins", icon=_icon('FONT_DATA'))
        for idx in range(3):
            row = sub.row(align=True)
            row.prop(props, _CREATE_PROPS[idx], text="")
            ok = _rig_ok(RIG_NAMES[idx])
            row.label(text=RIG_WORDS[idx], icon=_icon('CHECKMARK' if ok else 'ERROR'))
        if LAST_ERROR["msg"]:
            err_box = sub.box()
            err_box.alert = True
            err_box.label(text=LAST_ERROR["msg"][:70], icon=_icon('ERROR'))
            err_box.label(text="(see the System Console for details)")
        sub.operator("win_anim.create_missing", icon=_icon('ADD'))

        # ==================== Animate Rigs ====================
        box = layout.box()
        box.label(text="Animate Rigs", icon=_icon('ARMATURE_DATA'))
        col = box.column(align=True)
        for idx in range(3):
            row = col.row(align=True)
            row.prop(props, _RIG_PROPS[idx], text="")
            if _rig_ok(RIG_NAMES[idx]):
                row.label(text=RIG_WORDS[idx], icon=_icon('BONE_DATA'))
            else:
                row.label(text="%s (missing)" % RIG_WORDS[idx], icon=_icon('ERROR'))

        # ==================== Settings ====================
        box = layout.box()
        box.label(text="Settings", icon=_icon('SETTINGS'))
        box.prop(props, "fps")
        box.prop(props, "auto_axes")
        col = box.column(align=True)
        col.prop(props, "intensity", slider=True)
        col.prop(props, "stagger", slider=True)
        if abs(props.intensity - 1.0) > 1e-6 or abs(props.stagger - 1.0) > 1e-6:
            box.label(text="Rebuild to use the new values", icon=_icon('INFO'))
        box.label(text=f"Frames: {F_START}-{F_END} ({F_END - F_START + 1}f)", icon=_icon('TIME'))
        box.label(text=f"Entry {F_START}-{F_ENTRY_END} | Loop {F_LOOP_START}-{F_IDLE_END} | Exit {F_EXIT_START}-{F_END}")

        # ==================== Build & Manage ====================
        box = layout.box()
        box.label(text="Build & Manage", icon=_icon('TOOL_SETTINGS'))
        col = box.column(align=True)
        col.operator("win_anim.build_all", icon=_icon('ACTION'))
        col.operator("win_anim.verify", icon=_icon('CHECKMARK'))
        col.operator("win_anim.clear", icon=_icon('TRASH'))
        col.operator("win_anim.reset_pose", icon=_icon('LOOP_BACK'))

        # ==================== Variation picker ====================
        box = layout.box()
        box.label(text="Variation", icon=_icon('ANIM_DATA'))
        box.prop(props, "variation", text="")
        i = int(props.variation)
        vname = VARIATIONS[i - 1][0]
        rig_total = _rig_total()
        row = box.row(align=True)
        op = row.operator("win_anim.apply", text="Apply", icon=_icon('PLAY'))
        op.variation = i
        op = row.operator("win_anim.build_one", text="Build", icon=_icon('FILE_REFRESH'))
        op.variation = i
        _status_icon(row, vname, rig_total)
        box.label(text=VARIATION_DESCRIPTIONS.get(i, ""), icon=_icon('INFO'))
        row = box.row(align=True)
        row.label(text="Preview:")
        for sec in ("ENTRY", "LOOP", "EXIT", "ALL"):
            row.operator("win_anim.preview", text=sec.title()).section = sec

        # ==================== All variations ====================
        box = layout.box()
        box.prop(props, "show_all", text="All %d variations" % len(VARIATIONS),
                 icon=_icon('DISCLOSURE_TRI_DOWN' if props.show_all else 'DISCLOSURE_TRI_RIGHT'), emboss=False)
        if props.show_all:
            col = box.column(align=True)
            for k, (vn, _) in enumerate(VARIATIONS, start=1):
                row = col.row(align=True)
                row.label(text="%d  %s" % (k, vn.split("_", 1)[1].replace("_", " ")))
                _status_icon(row, vn, rig_total)
                op = row.operator("win_anim.apply", text="", icon=_icon('PLAY'))
                op.variation = k
                op = row.operator("win_anim.build_one", text="", icon=_icon('FILE_REFRESH'))
                op.variation = k


# ============================================================
# REGISTER
# ============================================================

_classes = (
    WinAnimProps,
    WIN_ANIM_OT_build_all,
    WIN_ANIM_OT_build_one,
    WIN_ANIM_OT_apply,
    WIN_ANIM_OT_clear,
    WIN_ANIM_OT_reset_pose,
    WIN_ANIM_OT_create_missing,
    WIN_ANIM_OT_verify,
    WIN_ANIM_OT_preview,
    VIEW3D_PT_win_animations,
)


def _unregister_existing():
    """Clean up an older copy of this add-on that is still registered
    (for example when the script is run again from the Text Editor)."""
    if hasattr(bpy.types.Scene, "win_anim"):
        try:
            del bpy.types.Scene.win_anim
        except Exception:
            pass
    for cls in reversed(_classes):
        old = getattr(bpy.types, cls.__name__, None)
        if old is not None:
            try:
                bpy.utils.unregister_class(old)
            except Exception:
                pass


def register():
    _unregister_existing()
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.win_anim = bpy.props.PointerProperty(type=WinAnimProps)


def unregister():
    if hasattr(bpy.types.Scene, "win_anim"):
        del bpy.types.Scene.win_anim
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
