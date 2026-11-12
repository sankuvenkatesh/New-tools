"""2D -> 3D: use the depth model to solve a real depth for every joint, then map to the armature scale with a perspective camera.

How it works:
  1. The depth model (depth.py, Depth Anything V2) predicts a relative depth for the whole image; it is sampled at the keypoint
     pixels to give every joint a depth d (relative depth, affine invariant: true depth = alpha*d + beta);
  2. Camera: X=(u-cx)*Y/f, Z=(cy-v)*Y/f, Y=beta+alpha*n (n is the normalised relative depth, f is the focal length);
  3. (alpha, beta) come from one **global** least-squares fit (Gauss-Newton) of "length of every bone segment in the image = armature bone length",
     so there is no longer any joint-by-joint solving from bone lengths and no heuristic about "elbow forward or backward":
     under perspective, "who is closer to the camera" changes the pixel scale, so the depth direction is identifiable;
  4. The bone-length solve (old method) is only a fallback: used when the depth model is missing, the fit fails, or the fit error is clearly larger;
  5. Finally the whole set of coordinates is rotated into the armature rest frame according to the person's facing (the character faces -Y).

Every joint, projected back into the image, agrees with the original keypoint; the depth direction comes from real pixels, not from guesses.
"""
from __future__ import annotations

import math

import numpy as np
from mathutils import Quaternion, Vector

from . import coco, depth as depth_model, utils

# Person facing (camera space: +Y points into the image, i.e. away from the camera)
FACING_DIRS = {
    "FRONT": np.array((0.0, -1.0, 0.0)),   # facing the camera
    "BACK": np.array((0.0, 1.0, 0.0)),     # back to the camera
    "FACE_RIGHT": np.array((1.0, 0.0, 0.0)),  # the person faces the right of the image (side view)
    "FACE_LEFT": np.array((-1.0, 0.0, 0.0)),
}
FACING_LABELS = {
    "FRONT": "Front facing the camera",
    "BACK": "Back facing the camera",
    "FACE_RIGHT": "Facing the right of the image (side view)",
    "FACE_LEFT": "Facing the left of the image (side view)",
}

# Depth source:
#   DEPTH = use the depth model to solve a depth for every joint (default; front/back direction comes from pixels, no guessing)
#   BONE  = old method: bone-length constraint dz=±√(L²-d²), sign from fixed anatomical assumptions (fallback)
#   FLAT  = no depth, the whole pose is flattened into the image plane
DEPTH_MODES = ("DEPTH", "BONE", "FLAT")
DEPTH_MODE_LABELS = {
    "DEPTH": "Depth model (recommended)",
    "BONE": "Bone-length solve (old method)",
    "FLAT": "Flatten into the image plane",
}

DEFAULT_OPTIONS = {
    "score_thr": 0.3,
    "facing": "FRONT",
    "depth_mode": "DEPTH",
    "depth_focal": 1.2,     # focal factor: f = factor x max(image width, image height), in pixels
    "depth_flip": False,    # tick only if the automatically fitted depth direction is reversed
    # Depth semantics given by the physical criterion (the person is closer than the background): True = large value = near. It is an independent line of evidence from the bone-length fit;
    # when the fit cannot tell front from back (frontal standing pose, body flat in depth) it decides the sign, see _apply_direction_hint.
    "depth_near_large": None,
    "swap_lr": False,
    "fingers": False,
    # Hand: fraction by which the hand (palm + fingers) is pulled toward the forearm direction. 0 = follow the hand keypoints exactly (only clamped to the upper limit),
    # 1 = hand fully along the forearm. Hand keypoints only reach the finger bases, so a fist / side-on palm / misplaced fingertips give a very large deviation,
    # and taking it as-is would leave the palm sticking out sideways from the wrist, see _align_hands.
    "hand_align": 0.5,
    "scale_hint": 1.0,
}

# Fixed "front/back direction" assumptions in the bone-length fallback mode (a tunable switch in the old version, now only a fallback constant)
FALLBACK_ELBOW_BACK = True
FALLBACK_KNEE_FRONT = True
FALLBACK_SPINE_LEAN_FRONT = True

# Two parameterisations of the depth fit: the depth model may be linear in "depth" or linear in "disparity" (1/depth).
# Both are fitted and the one with the smaller residual wins, so we do not have to bet on the model's output semantics.
DEPTH_TRANSFORMS = ("linear", "inverse")

# Only when the bone-length residual (metres) of the depth solve exceeds this value is the depth considered untrustworthy and the bone-length solve used instead.
# Note: it is deliberately **not** compared with the residual of the bone-length solve - that solve forces the bone lengths by construction
# (the bone lengths are almost always satisfied, at the price of guessing the depth direction), so comparing residuals would always favour it.
MAX_DEPTH_BONE_ERROR = 0.12

# When the fit costs of the two depth directions (sign of alpha) differ by less than this ratio, "the bone lengths cannot tell front from back"
# and the physical criterion decides the sign. In real photos the two directions measured only 0.1% apart (ratio 1.001),
# i.e. the direction was a coin toss - and a pose flipped front-to-back looks like "the hand is behind the body".
DEPTH_SIGN_TIE_RATIO = 1.10

# Tolerance of the segment-level "depth repair": a segment is only repaired when |actual bone length - armature bone length| exceeds max(this value, ratio x bone length).
DEPTH_REPAIR_TOL = 0.02
DEPTH_REPAIR_RATIO = 0.12

# If the front/back depth difference given by the depth map is smaller than this value (metres), the model is treated as "silent" and the sign falls back to the anatomical default
# (monocular depth models often smear thin limbs onto the torso depth, and the elbow / wrist difference is often under 1 cm).
DEPTH_SIGN_MIN_DELTA = 0.01

# Maximum angle of the hand away from the forearm: beyond it the hand is clamped into this forearm cone (whatever hand_align is).
# Hand keypoints only reach the finger bases; a fist / side-on palm / fingertips misplaced to the other side of the palm base can give a 40~90 degree deviation,
# and taking it as-is would leave the palm sticking out sideways from the wrist; in real photos the hand almost always follows the forearm.
HAND_ALIGN_MAX_DEG = 30.0

# --------------------------------------------------------------------------- facial features / head
# "Head base" (skull base / upper neck opening, i.e. the start of the head bone) estimated from the facial features: the ear midpoint lies about
# 0.15 x (eyes -> chin) below the eye line (adult head proportions: the eye line is at the middle of the head height, the ear canal is about 2 cm below the eye line).
# Ears are body-side branches and very unreliable when the head is turned / hair covers them / only one is detected - the old approach then fell back to "midpoint of the eyes",
# which sits on top of the head and pulls the neck short and crooked, so the proportions of the face itself are used instead.
FACE_HEAD_BASE_BELOW_EYE = 0.15
# The same scale converted to the "top of the head": ear line to skull top = about 1.1 x (eyes -> chin).
FACE_HEAD_TIP_ABOVE_BASE = 1.1
# When the chin is not detected, derive "eyes -> chin" from (eyes -> mouth) (adult about 0.4).
FACE_EYE_TO_MOUTH_RATIO = 0.4
# If the ear midpoint is farther than this ratio (x eyes -> chin) from the head base estimated from the facial features, the ears are considered unreliable.
FACE_EAR_TOLERANCE = 0.9
# Lower limit of the "agreement" of paired facial points: if |sum of unit vectors| / sum of weights is below it, those pairs contradict each other
# (most likely the depth map is wrong on the face); then only the in-image component is kept and the front/back component is dropped - better not to
# turn the head at all than let depth noise twist the head round.
FACE_AXIS_MIN_CONSISTENCY = 0.7


def default_options(**overrides) -> dict:
    opts = dict(DEFAULT_OPTIONS)
    opts.update(overrides)
    return opts

def _pix(keypoints, scores, index, thr: float):
    if index is None:
        return None
    if float(scores[index]) < thr:
        return None
    x, y = float(keypoints[index][0]), float(keypoints[index][1])
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0.0 or y < 0.0:
        return None
    return np.array((x, y), dtype=np.float64)


def _mean(points):
    points = [p for p in points if p is not None]
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def _resample_polyline(points: list[np.ndarray], count: int) -> list[np.ndarray]:
    """Resample a polyline into count+1 points by arc length."""
    if len(points) < 2 or count < 1:
        return list(points)
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    seg = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    total = float(sum(seg))
    if total <= 1e-9:
        return [pts[0].copy() for _ in range(count + 1)]

    cumulative = [0.0]
    for length in seg:
        cumulative.append(cumulative[-1] + length)

    out = []
    for k in range(count + 1):
        target = total * k / count
        index = 0
        while index < len(seg) - 1 and cumulative[index + 1] < target:
            index += 1
        span = max(cumulative[index + 1] - cumulative[index], 1e-12)
        t = (target - cumulative[index]) / span
        out.append(pts[index] * (1.0 - t) + pts[index + 1] * t)
    return out


def _safe_cross(a: np.ndarray, b: np.ndarray):
    axis = np.cross(a, b)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


# --------------------------------------------------------------------------- facial features -> head frame
def _confidence(scores, index) -> float:
    """Confidence of a single keypoint (used to weight facial point pairs); counts as 0 when unavailable / not finite."""
    try:
        value = float(np.asarray(scores, dtype=float)[index])
    except (IndexError, TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _pair_direction(points: dict, weights: dict, pairs, plane=None):
    """Direction (unit vector) voted by paired points + the pairs used + the agreement; when there are not enough points returns ``(None, [], 0.0)``.

    When ``plane`` is given, the points are first moved into 3D with it (``plane(pixel) -> (x, z)`` in orthographic mode), otherwise the vote happens in the pixel
    plane. Each pair is weighted by "anatomical weight x confidence of both ends x baseline length of the pair": the longer the baseline the more reliable the direction,
    and low-confidence points (eye corners / mouth corners hidden when the head is turned) make themselves lighter. The returned agreement
    (``|sum of unit vectors| / sum of weights``) tells whether these pairs **contradict each other**.
    """
    total = np.zeros(3 if plane is not None else 2, dtype=np.float64)
    used = []
    weight_sum = 0.0
    for key_a, key_b, weight in pairs:
        point_a, point_b = points.get(key_a), points.get(key_b)
        if point_a is None or point_b is None:
            continue
        if plane is not None:
            flat_a, flat_b = plane(point_a), plane(point_b)
            a = np.array((flat_a[0], 0.0, flat_a[1]), dtype=np.float64)
            b = np.array((flat_b[0], 0.0, flat_b[1]), dtype=np.float64)
        else:
            a = np.asarray(point_a, dtype=np.float64)
            b = np.asarray(point_b, dtype=np.float64)
        vector = b - a
        span = float(np.linalg.norm(vector))
        if span < 1e-9:
            continue
        current = float(weight) * min(weights.get(key_a, 1.0), weights.get(key_b, 1.0)) * span
        if current <= 0.0:
            continue
        total += vector / span * current
        weight_sum += current
        used.append((key_a, key_b, float(weight)))
    norm = float(np.linalg.norm(total))
    if norm < 1e-9 or weight_sum <= 0.0:
        return None, [], 0.0
    return total / norm, used, norm / weight_sum


def _face_frame(keypoints, scores, thr: float) -> dict:
    """Estimate a "head frame" from the 68-point face, for the head / neck.

    Why not settle for "ears + eyebrows/chin": ears are body-side branches - when the head is turned, hair covers them or only one is detected
    they give a very skewed position, and the neck direction is "upper chest -> head base", so as soon as the head base is off the neck tilts; estimating the head direction from
    just the eyebrows and chin is also dragged around by chin detection errors. The 68-point face is **made specifically for head pose**:

      * **side axis** (eye-line direction, see :data:`coco.FACE_SIDE_PAIRS`): eye corners / mouth corners / brow tips /
        jaw corners averaged in pairs. It is the roll reference axis of ``head`` in retarget, so head turn and head tilt both rely on this;
      * **up axis** (chin -> eyes, see :data:`coco.FACE_UP_PAIRS`): nodding / looking up / tilting rely on it,
        and it is also the direction the ``head`` bone aims at (the old approach used only the two points "eyebrows - chin");
      * **face scale** (eyes -> chin = about 11.5 cm): converts "how far below the eye line the head base is" and "how far above the head base the top of the head is"
        into pixels, used to place the head base and head top when the ears are unusable.

    The returned ``points`` are pixel points (keys ``face_<group name>``, plus two midpoints eye_mid / brow_mid),
    ``side_pairs`` / ``up_pairs`` record "which pairs produced the direction": each of the two solve modes moves these points to 3D with its own
    camera model and votes there - every pair has its own real depth, so the voted direction
    **carries a front/back component** (the front/back component of the eye line is head turn, that of the up axis is nodding).
    """
    points: dict = {}
    weights: dict = {}
    for key, indexes in coco.FACE_GROUPS.items():
        found, levels = [], []
        for index in indexes:
            point = _pix(keypoints, scores, index, thr)
            if point is None:
                continue
            found.append(point)
            levels.append(_confidence(scores, index))
        if not found:
            continue
        points[key] = np.mean(np.stack(found, axis=0), axis=0)
        weights[key] = sum(levels) / len(levels)

    # Midpoints: mean of the left and right groups (used by both the eye line and the up axis; when one side is missing the matching pairs are simply absent)
    for name, key_a, key_b in (("eye_mid", "eye_l", "eye_r"),
                               ("brow_mid", "brow_l", "brow_r"),
                               ("jaw_mid", "jaw_l", "jaw_r")):
        if key_a in points and key_b in points:
            points[name] = (points[key_a] + points[key_b]) * 0.5
            weights[name] = min(weights.get(key_a, 0.0), weights.get(key_b, 0.0))

    side_dir, side_pairs, side_consistency = _pair_direction(points, weights,
                                                             coco.FACE_SIDE_PAIRS)
    up_dir, up_pairs, up_consistency = _pair_direction(points, weights, coco.FACE_UP_PAIRS)

    # Face scale: eyes -> chin. If the chin is not detected, derive it from (eyes -> mouth).
    face_len = None
    eye_mid, mouth_mid, chin = points.get("eye_mid"), points.get("mouth_mid"), points.get("chin")
    if eye_mid is not None and chin is not None:
        face_len = float(np.linalg.norm(chin - eye_mid))
    elif eye_mid is not None and mouth_mid is not None:
        face_len = float(np.linalg.norm(eye_mid - mouth_mid)) / FACE_EYE_TO_MOUTH_RATIO
    if face_len is not None and (not math.isfinite(face_len) or face_len < 1e-6):
        face_len = None

    # Head base: 0.15 face lengths below the eye line (= about the height of the ear canal / upper neck opening)
    base = None
    if eye_mid is not None and up_dir is not None and face_len:
        base = eye_mid - np.asarray(up_dir, dtype=np.float64) * (
            FACE_HEAD_BASE_BELOW_EYE * face_len)

    eye_span = None
    if "eye_l" in points and "eye_r" in points:
        eye_span = float(np.linalg.norm(points["eye_r"] - points["eye_l"]))

    return {"points": points, "weights": weights,
            "side_dir": side_dir, "side_pairs": side_pairs, "side_consistency": side_consistency,
            "up_dir": up_dir, "up_pairs": up_pairs, "up_consistency": up_consistency,
            "face_len": face_len, "eye_span": eye_span, "base": base,
            "count": len(points),
            "reliable": bool(side_dir is not None and up_dir is not None and face_len)}


def _ear_plausible(ears, base, face_len) -> bool:
    """Whether the midpoint of the two ears is far from the head base estimated from the facial features (far = the ears are probably hidden / misdetected)."""
    if ears is None or base is None or not face_len:
        return True
    return float(np.linalg.norm(np.asarray(ears, dtype=float)
                                - np.asarray(base, dtype=float))) <= FACE_EAR_TOLERANCE * face_len


def _face_axis_3d(place, face, key: str, consistency: float = FACE_AXIS_MIN_CONSISTENCY):
    """Direction (unit vector) voted in 3D by paired facial points; returns None when there are not enough points / the direction is degenerate.

    ``place`` moves the ``face_<group name>`` points to 3D (the depth solve uses real depth, orthographic mode uses the image plane),
    so the voted result carries a front/back component. When the pairs contradict each other (agreement below ``consistency``) the front/back component is
    dropped: the depth map is most likely wrong on the face, so only the in-image direction is kept (head tilt stays accurate, head turn does not go wild).
    """
    if not face:
        return None
    total = np.zeros(3, dtype=np.float64)
    weight_sum = 0.0
    for key_a, key_b, weight in (face.get(key) or ()):
        point_a = place("face_%s" % key_a)
        point_b = place("face_%s" % key_b)
        if point_a is None or point_b is None:
            continue
        vector = np.asarray(point_b, dtype=np.float64) - np.asarray(point_a, dtype=np.float64)
        span = float(np.linalg.norm(vector))
        if span < 1e-9:
            continue
        current = float(weight) * span
        total += vector / span * current
        weight_sum += current
    norm = float(np.linalg.norm(total))
    if norm < 1e-9 or weight_sum <= 0.0:
        return None
    if norm / weight_sum < float(consistency):
        total = np.array((total[0], 0.0, total[2]), dtype=np.float64)
        norm = float(np.linalg.norm(total))
        if norm < 1e-9:
            return None
    return total / norm


def _collect_2d(keypoints, scores, opts) -> dict:
    """Collect the required 2D pixel points; missing ones are filled by mirroring / interpolation, and which ones were filled in is recorded."""
    thr = float(opts["score_thr"])
    swap = bool(opts["swap_lr"])

    def kp(index):
        return _pix(keypoints, scores, index, thr)

    pairs = {
        "eye_L": (coco.LEFT_EYE, coco.RIGHT_EYE), "eye_R": (coco.RIGHT_EYE, coco.LEFT_EYE),
        "ear_L": (coco.LEFT_EAR, coco.RIGHT_EAR), "ear_R": (coco.RIGHT_EAR, coco.LEFT_EAR),
        "sh_L": (coco.LEFT_SHOULDER, coco.RIGHT_SHOULDER),
        "sh_R": (coco.RIGHT_SHOULDER, coco.LEFT_SHOULDER),
        "el_L": (coco.LEFT_ELBOW, coco.RIGHT_ELBOW), "el_R": (coco.RIGHT_ELBOW, coco.LEFT_ELBOW),
        "wr_L": (coco.LEFT_WRIST, coco.RIGHT_WRIST), "wr_R": (coco.RIGHT_WRIST, coco.LEFT_WRIST),
        "hip_L": (coco.LEFT_HIP, coco.RIGHT_HIP), "hip_R": (coco.RIGHT_HIP, coco.LEFT_HIP),
        "kn_L": (coco.LEFT_KNEE, coco.RIGHT_KNEE), "kn_R": (coco.RIGHT_KNEE, coco.LEFT_KNEE),
        "an_L": (coco.LEFT_ANKLE, coco.RIGHT_ANKLE), "an_R": (coco.RIGHT_ANKLE, coco.LEFT_ANKLE),
        "big_L": (coco.LEFT_BIG_TOE, coco.RIGHT_BIG_TOE),
        "big_R": (coco.RIGHT_BIG_TOE, coco.LEFT_BIG_TOE),
    }
    pts = {name: kp(a if not swap else b) for name, (a, b) in pairs.items()}
    synth: set[str] = set()

    pts["nose"] = kp(coco.NOSE)
    pts["chin"] = kp(coco.FACE_CHIN)
    pts["brow"] = _mean([kp(i) for i in coco.FACE_BROWS])

    # Facial features: the 68-point face is made for head pose; the head / neck orientation is voted from it (see _face_frame)
    face = _face_frame(keypoints, scores, thr)
    for key, point in face["points"].items():
        pts["face_%s" % key] = point
    pts["eye_a"] = pts.get("face_eye_l")
    if pts["eye_a"] is None:
        pts["eye_a"] = kp(coco.FACE_EYE_L_END)     # outer eye corner on the left of the image
    pts["eye_b"] = pts.get("face_eye_r")
    if pts["eye_b"] is None:
        pts["eye_b"] = kp(coco.FACE_EYE_R_END)     # outer eye corner on the right of the image

    left_base = coco.HAND_LEFT_START if not swap else coco.HAND_RIGHT_START
    right_base = coco.HAND_RIGHT_START if not swap else coco.HAND_LEFT_START
    hands = {}
    for side, base in (("L", left_base), ("R", right_base)):
        hands[side] = {local: kp(base + local) for local in range(coco.HAND_COUNT)}

    if pts["sh_L"] is None and pts["sh_R"] is None:
        return {"error": "No shoulder keypoints were detected, cannot reconstruct the pose"}

    if pts["sh_L"] is not None and pts["sh_R"] is not None:
        pts["chest"] = (pts["sh_L"] + pts["sh_R"]) * 0.5
    elif pts["sh_L"] is not None:
        pts["chest"] = pts["sh_L"].copy()
    else:
        pts["chest"] = pts["sh_R"].copy()

    if pts["sh_L"] is None:
        pts["sh_L"] = (2.0 * pts["chest"] - pts["sh_R"]) if pts["sh_R"] is not None \
            else pts["chest"].copy()
        synth.add("sh_L")
    if pts["sh_R"] is None:
        pts["sh_R"] = (2.0 * pts["chest"] - pts["sh_L"]) if pts["sh_L"] is not None \
            else pts["chest"].copy()
        synth.add("sh_R")

    # Hips
    if pts["hip_L"] is None and pts["hip_R"] is None:
        anchors = [p for p in (pts["an_L"], pts["an_R"], pts["kn_L"], pts["kn_R"])
                   if p is not None]
        bottom = _mean(anchors) if anchors else None
        pelvis = pts["chest"] + (bottom - pts["chest"]) * 0.45 if bottom is not None \
            else pts["chest"].copy()
        half = max(abs(pts["sh_L"][0] - pts["sh_R"][0]) * 0.4, 8.0)
        pts["hip_L"] = np.array((pelvis[0] + half, pelvis[1]))
        pts["hip_R"] = np.array((pelvis[0] - half, pelvis[1]))
        synth.update(("hip_L", "hip_R"))
    elif pts["hip_L"] is None:
        pts["hip_L"] = np.array((2.0 * pts["chest"][0] - pts["hip_R"][0], pts["hip_R"][1]))
        synth.add("hip_L")
    elif pts["hip_R"] is None:
        pts["hip_R"] = np.array((2.0 * pts["chest"][0] - pts["hip_L"][0], pts["hip_L"][1]))
        synth.add("hip_R")
    pts["pelvis"] = (pts["hip_L"] + pts["hip_R"]) * 0.5

    # Interpolate / extrapolate missing limb joints
    for side in ("L", "R"):
        for name, above, below in (("kn_%s" % side, "hip_%s" % side, "an_%s" % side),
                                   ("an_%s" % side, "kn_%s" % side, None),
                                   ("el_%s" % side, "sh_%s" % side, "wr_%s" % side),
                                   ("wr_%s" % side, "el_%s" % side, None),
                                   ("big_%s" % side, "an_%s" % side, None)):
            if pts.get(name) is not None:
                continue
            top = pts.get(above)
            bottom = pts.get(below) if below else None
            if top is None and bottom is None:
                continue
            if top is None:
                top = bottom
            if bottom is not None:
                pts[name] = (top + bottom) * 0.5 if name not in ("an_%s" % side, "wr_%s" % side) \
                    else bottom
            else:
                if name.startswith("big_"):
                    span = 0.45 * float(np.linalg.norm(pts["an_%s" % side]
                                                       - pts["kn_%s" % side])) \
                        if pts.get("kn_%s" % side) is not None else 12.0
                    pts[name] = top + np.array((0.0, max(span, 6.0)))
                else:
                    span = float(np.linalg.norm(pts["pelvis"] - pts["chest"]))
                    pts[name] = top + np.array((0.0, max(span * 0.45, 8.0)))
            synth.add(name)

    # Head reference point (skull base / upper neck opening):
    #   the midpoint of the two ears is the best direct observation, but ears are body-side branches - if only one is detected, or it is wildly different from the head
    #   base estimated from the facial features (common with a turned head + hair occlusion), it cannot be used, or the neck would tilt to the wrong side.
    #   In that case estimate from the facial features: 0.15 x (eyes -> chin) below the eye line = about the height of the ear canal. Then fall back to the eye midpoint /
    #   nose / torso extrapolation (the old approach).
    face_base = face.get("base")
    ears = None
    if pts.get("ear_L") is not None and pts.get("ear_R") is not None:
        ears = (pts["ear_L"] + pts["ear_R"]) * 0.5
        if not _ear_plausible(ears, face_base, face.get("face_len")):
            ears = None
    if ears is not None:
        pts["head_base"] = ears.copy()
        face["head_base_source"] = "ears"
    elif face_base is not None:
        pts["head_base"] = np.asarray(face_base, dtype=float).copy()
        synth.add("head_base")
        face["head_base_source"] = "face"
    else:
        anchor = _mean([pts.get("eye_L"), pts.get("eye_R")])
        if anchor is None:
            anchor = pts.get("nose")
        if anchor is not None:
            pts["head_base"] = anchor.copy()
            face["head_base_source"] = "eye"
        else:
            pts["head_base"] = pts["chest"] + (pts["pelvis"] - pts["chest"]) * -0.35
            synth.add("head_base")
            face["head_base_source"] = "torso"

    # Head top (the direction the head bone aims at): prefer the up axis of the facial features (voted by several pairs); the depth sampling point still lands on the head
    if face.get("up_dir") is not None and face.get("face_len"):
        pts["head_tip"] = pts["head_base"] + np.asarray(face["up_dir"], dtype=float) * (
            FACE_HEAD_TIP_ABOVE_BASE * float(face["face_len"]))
    elif pts["brow"] is not None and pts["chin"] is not None:
        pts["head_tip"] = pts["brow"] + (pts["brow"] - pts["chin"]) * 0.25
    elif pts["nose"] is not None:
        base = pts["head_base"]
        pts["head_tip"] = pts["nose"] + (pts["nose"] - base) * 0.3
    else:
        pts["head_tip"] = None
        synth.add("head_tip")

    return {"points": pts, "hands": hands, "synth": synth, "face": face}


# --------------------------------------------------------------------------- shared helpers
def _orthographic_scale(pts, metrics, image_size, opts) -> float:
    """Fit "pixels per metre" (the global scale of orthographic mode) from "keypoint pixel distances / armature bone lengths"."""
    spine_span = float((metrics["chest"] - metrics["pelvis"]).length)
    samples = []
    for side in ("L", "R"):
        samples.extend([
            (pts.get("sh_%s" % side), pts.get("el_%s" % side),
             metrics["upper_arm_len"][side]),
            (pts.get("el_%s" % side), pts.get("wr_%s" % side),
             metrics["forearm_len"][side]),
            (pts.get("hip_%s" % side), pts.get("kn_%s" % side),
             metrics["thigh_len"][side]),
            (pts.get("kn_%s" % side), pts.get("an_%s" % side),
             metrics["shin_len"][side]),
        ])
    samples.append((pts.get("pelvis"), pts.get("chest"), spine_span))
    samples = [item for item in samples
               if item[0] is not None and item[1] is not None and item[2] > 1e-6
               and float(np.linalg.norm(item[1] - item[0])) > 1.0]

    if samples:
        # Keypoint distances can only get smaller through "perspective foreshortening", so the true value should be a larger quantile,
        # otherwise most bones would get no depth (equivalent to flattening the pose).
        ratios = sorted(float(np.linalg.norm(b - a)) / L for a, b, L in samples)
        selected = ratios[-max(1, len(ratios) // 4):]
        scale = float(sum(selected) / len(selected))
    else:
        scale = float(image_size[1]) / 1.7
    scale *= float(opts.get("scale_hint", 1.0) or 1.0)
    return float(max(scale, 1e-3))


def _bone_constraints(metrics) -> list[dict]:
    """Bone-segment constraints: point names are used for pixel/depth sampling, joint names for the 3D bone-length error, lengths in metres."""
    items = []
    for side in ("L", "R"):
        items.extend([
            {"point_a": "sh_%s" % side, "point_b": "el_%s" % side,
             "joint_a": "shoulder.%s" % side, "joint_b": "elbow.%s" % side,
             "length": float(metrics["upper_arm_len"][side] or 0.0)},
            {"point_a": "el_%s" % side, "point_b": "wr_%s" % side,
             "joint_a": "elbow.%s" % side, "joint_b": "wrist.%s" % side,
             "length": float(metrics["forearm_len"][side] or 0.0)},
            {"point_a": "hip_%s" % side, "point_b": "kn_%s" % side,
             "joint_a": "hip.%s" % side, "joint_b": "knee.%s" % side,
             "length": float(metrics["thigh_len"][side] or 0.0)},
            {"point_a": "kn_%s" % side, "point_b": "an_%s" % side,
             "joint_a": "knee.%s" % side, "joint_b": "ankle.%s" % side,
             "length": float(metrics["shin_len"][side] or 0.0)},
            {"point_a": "an_%s" % side, "point_b": "big_%s" % side,
             "joint_a": "ankle.%s" % side, "joint_b": "ball.%s" % side,
             "length": float(metrics["foot_len"][side] or 0.0)},
        ])
    spine_span = float((metrics["chest"] - metrics["pelvis"]).length)
    items.append({"point_a": "pelvis", "point_b": "chest",
                  "joint_a": "pelvis", "joint_b": "chest",
                  "length": float(metrics.get("pelvis_to_shoulders") or spine_span)})
    return [item for item in items if item["length"] > 1e-6]


def _bone_error_rms(joints, constraints) -> float:
    """Root-mean-square error (metres) between the actual 3D bone lengths and the armature bone lengths; both solve modes are compared with the same ruler."""
    errors = []
    for item in constraints:
        head = joints.get(item["joint_a"])
        tail = joints.get(item["joint_b"])
        if head is None or tail is None:
            continue
        errors.append(float(np.linalg.norm(np.asarray(tail) - np.asarray(head)))
                      - item["length"])
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errors))))


def _perspective_point(point, depth_axis, cx, cy, focal):
    """Pixel + depth -> camera-space 3D (X right, Y into the image, Z up, in metres)."""
    y = float(depth_axis)
    return np.array(((float(point[0]) - cx) * y / focal, y,
                     (cy - float(point[1])) * y / focal))


def _huber(error: float, delta: float) -> float:
    """Huber residual: small errors stay linear, large errors (keypoint jitter / bone-length mismatch) are squashed to sqrt."""
    magnitude = abs(error)
    if magnitude <= delta:
        return error
    return math.copysign(math.sqrt(2.0 * delta * magnitude - delta * delta), error)


def _gauss_newton(residual_fn, theta0, iterations: int = 40) -> tuple[np.ndarray, float]:
    """Two-parameter Gauss-Newton + backtracking; returns (parameters, sum of squared residuals)."""
    theta = np.asarray(theta0, dtype=np.float64)
    residual = residual_fn(theta)
    cost = float(residual @ residual)
    for _ in range(iterations):
        jacobian = np.zeros((residual.size, theta.size), dtype=np.float64)
        for index in range(theta.size):
            step = 1e-4 * max(1.0, abs(float(theta[index])))
            shifted = theta.copy()
            shifted[index] += step
            jacobian[:, index] = (residual_fn(shifted) - residual) / step
        try:
            delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        accepted = False
        for attempt in range(8):
            trial = theta + delta * (0.5 ** attempt)
            trial_residual = residual_fn(trial)
            trial_cost = float(trial_residual @ trial_residual)
            if trial_cost < cost - 1e-12:
                theta, residual, cost = trial, trial_residual, trial_cost
                accepted = True
                break
        if not accepted:
            break
    return theta, cost


# --------------------------------------------------------------------------- bone-length solve (fallback)
def _solve_bone_lengths(pts, hands, metrics, opts, image_size, use_depth: bool,
                        signs: dict = None, face: dict = None) -> dict:
    """Old method: orthographic projection + bone-length constraint dz=±√(L²-d²), the sign uses fixed anatomical assumptions.

    Only used when the depth model is unavailable, the fit fails or the error is clearly larger (see reconstruct).

    ``signs`` is the sign table "which one of each segment is in front" given by the depth map (``(point A, point B) -> B is closer to the camera``).
    When a depth map exists it is preferred - in a single picture the elbow forward/backward project identically, so the depth map is the only line of evidence,
    and the fixed assumption (elbow backward) is only a fallback for segments with no depth information. "The hand is in front in the depth map but behind the body after applying"
    was caused by it not being used.
    """
    image_w, image_h = float(image_size[0]), float(image_size[1])
    scale = _orthographic_scale(pts, metrics, image_size, opts)
    cx, cy = image_w * 0.5, image_h * 0.5
    front = FACING_DIRS.get(opts["facing"], FACING_DIRS["FRONT"])
    front_y_sign = 1.0 if front[1] > 0.0 else -1.0
    signs = signs or {}

    def plane(point):
        return np.array(((point[0] - cx) / scale, (cy - point[1]) / scale))

    def want_front(point_a: str, point_b: str, default):
        """Front/back direction of this segment: use the depth map's if it gave one; ``None`` = the depth map has no opinion (stay in the plane);
        the fixed assumption is only used when nothing was given at all (no depth map)."""
        if (point_a, point_b) not in signs:
            return default
        return signs[(point_a, point_b)]

    def solve(parent3, point2, length, want_front):
        """Given the bone length and the in-plane position, find the child joint position; the depth sign is decided by want_front."""
        xy = plane(point2)
        dx, dy = xy[0] - parent3[0], xy[1] - parent3[2]
        distance = math.hypot(dx, dy)
        if not use_depth or length <= 1e-6 or distance >= length or want_front is None:
            depth = 0.0
        else:
            depth = math.sqrt(max(length * length - distance * distance, 0.0))
        offset = 0.0 if depth == 0.0 else (front_y_sign * depth if want_front
                                           else -front_y_sign * depth)
        return np.array((xy[0], parent3[1] + offset, xy[1]))

    joints: dict = {}
    pelvis_xy = plane(pts["pelvis"])
    pelvis_offset = float(metrics.get("pelvis_offset_z", 0.0) or 0.0)
    chest_offset = float(metrics.get("chest_offset_z", 0.0) or 0.0)
    # The depth solve uses the "unshifted" pelvis point; the shift is only used for later directions (otherwise it would contaminate the bone-length constraints)
    pelvis_plain = np.array((pelvis_xy[0], 0.0, pelvis_xy[1]))
    joints["pelvis"] = pelvis_plain + np.array((0.0, 0.0, pelvis_offset))

    spine_span = float((metrics["chest"] - metrics["pelvis"]).length)
    chest_len = float(metrics.get("pelvis_to_shoulders") or spine_span)
    joints["chest"] = solve(pelvis_plain, pts["chest"], chest_len,
                            want_front("pelvis", "chest",
                                       _default_front("pelvis", "chest")))
    joints["chest"][2] += chest_offset
    spine_points = _resample_polyline([joints["pelvis"], joints["chest"]], 4)

    # Rigid offsets on the torso: they rotate together with the torso and fix the 3D relation of shoulders / hips / neck / head base
    rest_pelvis = np.asarray(metrics["pelvis"], dtype=float)
    rest_chest = np.asarray(metrics["chest"], dtype=float)
    q_torso = Vector(rest_chest - rest_pelvis).normalized().rotation_difference(
        Vector(joints["chest"] - joints["pelvis"]).normalized())

    def torso_offset(vec):
        return np.asarray(q_torso @ Vector(np.asarray(vec, dtype=float)))

    for side in ("L", "R"):
        hip_xy = plane(pts["hip_%s" % side])
        hip_y = joints["pelvis"][1] + torso_offset(metrics["hip_offsets"][side])[1]
        joints["hip.%s" % side] = np.array((hip_xy[0], hip_y, hip_xy[1]))

    for side in ("L", "R"):
        sh_xy = plane(pts["sh_%s" % side])
        shoulder_y = joints["chest"][1] + torso_offset(
            metrics["shoulder_offsets"][side])[1]
        joints["shoulder.%s" % side] = np.array((sh_xy[0], shoulder_y, sh_xy[1]))
        clavicle = (metrics.get("clavicle_offsets") or {}).get(side)
        joints["shoulder_origin.%s" % side] = joints["chest"] + (
            torso_offset(clavicle) if clavicle is not None else 0.0)

    neck_offset = metrics.get("neck_offset")
    joints["neck_start"] = joints["chest"] + (torso_offset(neck_offset)
                                              if neck_offset is not None else 0.0)

    for side in ("L", "R"):
        shoulder = joints["shoulder.%s" % side]
        # A single image cannot tell whether the elbow is in front of or behind the body (both project to exactly the same image):
        # with a depth map use the sign it gives, otherwise guess with the anatomical assumption "elbow backward, knee forward".
        elbow = solve(shoulder, pts["el_%s" % side], metrics["upper_arm_len"][side],
                      want_front("sh_%s" % side, "el_%s" % side,
                                 _default_front("sh_%s" % side, "el_%s" % side)))
        joints["elbow.%s" % side] = elbow
        wrist = solve(elbow, pts["wr_%s" % side], metrics["forearm_len"][side],
                      want_front("el_%s" % side, "wr_%s" % side,
                                 _default_front("el_%s" % side, "wr_%s" % side)))
        joints["wrist.%s" % side] = wrist
        # Hand: keypoints only reach the finger bases, so infer from "in-plane direction + wrist depth" to avoid bone-length mismatches
        hand_point = hands[side].get(coco.HAND_MIDDLE_MCP)
        if hand_point is None:
            hand_point = pts["wr_%s" % side]
        hand_xy = plane(hand_point)
        joints["hand_tip.%s" % side] = np.array((hand_xy[0], wrist[1], hand_xy[1]))

        hip = joints["hip.%s" % side]
        knee = solve(hip, pts["kn_%s" % side], metrics["thigh_len"][side],
                     want_front("hip_%s" % side, "kn_%s" % side,
                                _default_front("hip_%s" % side, "kn_%s" % side)))
        joints["knee.%s" % side] = knee
        ankle = solve(knee, pts["an_%s" % side], metrics["shin_len"][side],
                      want_front("kn_%s" % side, "an_%s" % side,
                                 _default_front("kn_%s" % side, "an_%s" % side)))
        joints["ankle.%s" % side] = ankle
        ball = solve(ankle, pts["big_%s" % side], metrics["foot_len"][side],
                     want_front("an_%s" % side, "big_%s" % side,
                                _default_front("an_%s" % side, "big_%s" % side)))
        joints["ball.%s" % side] = ball
        toe_dir = ball - ankle
        norm = float(np.linalg.norm(toe_dir))
        toe_len = metrics["toe_len"][side] or 0.05
        joints["toe_tip.%s" % side] = ball + (toe_dir / norm if norm > 1e-9
                                              else np.array((0.0, -1.0, 0.0))) * toe_len

    # Head: the "up axis" of the facial features only gives the in-plane direction, the depth follows the head base (orthographic mode cannot determine pitch)
    head_len = metrics["head_len"] or 0.2
    head_xy = plane(pts["head_base"])
    head_offset = metrics.get("head_offset")
    head_y = joints["chest"][1] + (torso_offset(head_offset)[1]
                                   if head_offset is not None else 0.0)
    joints["head_base"] = np.array((head_xy[0], head_y, head_xy[1]))
    # Take the direction directly from the up axis voted by the facial features rather than "head base -> head top pixel": the head base is probably taken as the ear
    # midpoint (which shifts sideways when the head is turned), and using it as the start would pull the head direction off by more than ten degrees.
    up = _face_axis_plane(face, coco.FACE_UP_PAIRS, plane)
    if up is not None:
        joints["head_tip"] = joints["head_base"] + up * head_len
    elif pts["head_tip"] is not None:
        tip_xy = plane(pts["head_tip"])
        joints["head_tip"] = np.array((tip_xy[0], joints["head_base"][1], tip_xy[1]))
    else:
        joints["head_tip"] = joints["head_base"] + np.array((0.0, 0.0, head_len))

    refs = _build_refs(joints, _eye_line_from_plane(pts, face, plane))
    fingers = _build_fingers_bone(pts, hands, joints, opts, plane)
    _align_hands(joints, metrics, opts, fingers)
    return {"joints": joints, "refs": refs, "spine_points": spine_points,
            "fingers": fingers, "scale": scale,
            "mode": "BONE" if use_depth else "FLAT"}


def _face_axis_plane(face, pairs, plane):
    """In-plane facial feature direction (used by orthographic mode): points are first moved to (x, z) with ``plane``, depth is always 0."""
    if not face:
        return None
    direction = _pair_direction(face.get("points") or {}, face.get("weights") or {},
                                pairs, plane=plane)[0]
    if direction is None:
        return None
    return np.array((direction[0], 0.0, direction[2]), dtype=np.float64)


def _eye_line_from_plane(pts, face, plane):
    """Eye line in orthographic mode: of the side axis voted by the facial features only the in-plane direction is taken (depth follows the head).

    Using only the in-image direction is intentional - orthographic mode has no depth to begin with, and forcing pixels to act as "front/back" would give a random
    head-turn angle; what can be guaranteed here is the **head tilt / lean** (the in-image slope of the eye line), which is enough.
    """
    side = _face_axis_plane(face, coco.FACE_SIDE_PAIRS, plane)
    if side is not None:
        return side
    if pts.get("eye_a") is None or pts.get("eye_b") is None:
        return None
    a, b = plane(pts["eye_a"]), plane(pts["eye_b"])
    candidate = np.array((b[0] - a[0], 0.0, b[1] - a[1]))
    return candidate if float(np.linalg.norm(candidate)) > 1e-6 else None


def _build_refs(joints, eye_line=None) -> dict:
    """Reference axes (for roll / twist alignment); the eye line is computed by the caller with its own camera model.

    The sign of the eye line must be **unified with the shoulder line**: the eye line from the facial features is "image left -> image right"; when the person has their back to the camera it
    is exactly opposite to the body's left/right, whereas the rest-side reference axis in retarget (eye_line in rigify_map) uses
    "person right -> person left". Without unifying, a photo with the person's back to the camera would twist the head by 180 degrees.
    """
    shoulder_line = joints["shoulder.L"] - joints["shoulder.R"]
    refs = {"hip_line": joints["hip.L"] - joints["hip.R"],
            "shoulder_line": shoulder_line}
    eye = None
    if eye_line is not None:
        eye = np.asarray(eye_line, dtype=float)
        if float(np.dot(eye, shoulder_line)) < 0.0:
            eye = -eye
    if eye is None or float(np.linalg.norm(eye)) <= 1e-6:
        refs["eye_line"] = shoulder_line
    else:
        refs["eye_line"] = eye

    for side in ("L", "R"):
        upper_arm = joints["elbow.%s" % side] - joints["shoulder.%s" % side]
        forearm = joints["wrist.%s" % side] - joints["elbow.%s" % side]
        thigh = joints["knee.%s" % side] - joints["hip.%s" % side]
        shin = joints["ankle.%s" % side] - joints["knee.%s" % side]
        # Bend axes use a fixed-order cross product (upper segment x lower segment), consistent with the rest-side algorithm in rigify_map:
        # only then can roll alignment tell "limb bent forward" from "limb folded backward".
        axis = _safe_cross(upper_arm, forearm)
        if axis is not None:
            refs["armbend.%s" % side] = axis
        axis = _safe_cross(thigh, shin)
        if axis is not None:
            refs["kneebend.%s" % side] = axis
    return refs


# --------------------------------------------------------------------------- hand alignment
def _bend_toward(direction, reference, blend: float, max_deg: float) -> Quaternion:
    """The rotation (quaternion) that "turns direction towards reference".

    The deviation angle is first scaled by ``blend`` (0 = leave alone, 1 = fully aligned with reference) and then clamped to within ``max_deg``.
    Returns the identity quaternion when the direction is degenerate (length ~ 0) or no rotation is needed.
    """
    identity = Quaternion((1.0, 0.0, 0.0, 0.0))
    direction = np.asarray(direction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    span = float(np.linalg.norm(direction))
    ref_span = float(np.linalg.norm(reference))
    if span < 1e-9 or ref_span < 1e-9:
        return identity
    cosine = float(np.dot(direction, reference)) / (span * ref_span)
    angle = math.acos(min(max(cosine, -1.0), 1.0))
    keep = min(max(float(blend), 0.0), 1.0)
    # The deviation to keep after turning: first scaled by the ratio (blend=1 => 0 degrees = fully aligned), then clamped to the upper limit
    left = min(angle * (1.0 - keep), math.radians(max_deg))
    amount = angle - left                    # the amount actually to turn away
    if amount <= 1e-9:
        return identity
    turn = Vector(direction).rotation_difference(Vector(reference))
    if turn.angle < 1e-9:
        return identity
    # A minimal rotation is unique (when two directions are nearly opposite rotation_difference picks an axis by itself),
    # so it is enough to scale the angle by amount/angle.
    return Quaternion(turn.axis, turn.angle * (amount / angle))


def _align_hands(joints, metrics, opts, fingers=None) -> None:
    """Make the hand (palm + fingers) follow the forearm as much as possible: turn the whole hand about the wrist.

    Only the orientation of the hand relative to the forearm changes; the hand's shape (fist / open) and length are kept (the hand length still uses the armature hand length,
    and the rotation is rigid, so the distance between fingers and wrist does not change). Reason: hand keypoints only reach the finger bases, and with a fist / side-on palm /
    misplaced fingertips they give a 40~90 degree deviation, which taken as-is leaves the palm sticking out sideways from the wrist; `hand_align` controls how far to pull toward the
    forearm direction (0 = only clamp to the upper limit, no proportional pull, 1 = fully along the forearm), and whatever its value the result is clamped to
    within :data:`HAND_ALIGN_MAX_DEG`.
    """
    blend = float(opts.get("hand_align", 0.5) or 0.0)
    for side in ("L", "R"):
        elbow = joints.get("elbow.%s" % side)
        wrist = joints.get("wrist.%s" % side)
        tip = joints.get("hand_tip.%s" % side)
        if elbow is None or wrist is None or tip is None:
            continue
        wrist = np.asarray(wrist, dtype=np.float64)
        hand = np.asarray(tip, dtype=np.float64) - wrist
        span = float(np.linalg.norm(hand))
        if span < 1e-9:
            continue
        hand_len = float(metrics["hand_len"][side] or 0.0) or span
        # The reference is the forearm direction "elbow -> wrist" (the hand should extend outward along it), not "wrist -> elbow"
        forearm = wrist - np.asarray(elbow, dtype=np.float64)
        turn = _bend_toward(hand, forearm, blend, HAND_ALIGN_MAX_DEG)
        joints["hand_tip.%s" % side] = wrist + np.asarray(
            turn @ Vector(hand / span)) * hand_len
        for chain in (fingers or {}).get(side, {}).values():
            for index, point in enumerate(chain):
                chain[index] = wrist + np.asarray(
                    turn @ Vector(np.asarray(point, dtype=np.float64) - wrist))


def _build_fingers_bone(pts, hands, joints, opts, plane) -> dict:
    """Fingers in the bone-length fallback mode: in-plane direction + wrist depth."""
    fingers: dict = {}
    if not opts.get("fingers"):
        return fingers
    for side in ("L", "R"):
        base_y = joints["wrist.%s" % side][1]
        per_hand = {}
        for finger, indexes in coco.HAND_GROUPS.items():
            chain = []
            for local in indexes:
                point = hands[side].get(local)
                if point is None:
                    chain = []
                    break
                xy = plane(point)
                chain.append(np.array((xy[0], base_y, xy[1])))
            if chain:
                per_hand[finger] = chain
        if per_hand:
            fingers[side] = per_hand
    return fingers


# --------------------------------------------------------------------------- depth model solve
def _depth_variants(depths: dict, reference: str = "pelvis", span_keys=None) -> dict:
    """Convert the sampled relative depths into "normalised relative depths".

    The depth model may be linear in "depth" or in "disparity" (1/depth); both are produced,
    and the fit residual decides which one is used (we do not bet on the model's output semantics).
    span_keys selects the points used to compute the "span" (default: the constraint points): when an individual joint samples the background it does not
    skew the overall scale; the span is taken at the 80% quantile to tolerate a single outlier sample as well.
    Returns {parameterisation name: {point name: normalised value}}.
    """
    base = depths.get(reference)
    if base is None or not math.isfinite(base):
        return {}
    out = {}
    for kind in DEPTH_TRANSFORMS:
        values = {}
        for name, value in depths.items():
            if value is None or not math.isfinite(value):
                continue
            if kind == "inverse":
                if abs(value) < 1e-6 or abs(base) < 1e-6:
                    continue
                value = 1.0 / value - 1.0 / base
            else:
                value = value - base
            values[name] = float(value)
        span_values = sorted(abs(values[name]) for name in (span_keys or values)
                             if name in values)
        if not span_values:
            span_values = sorted(abs(item) for item in values.values())
        if not span_values:
            continue
        span = span_values[min(len(span_values) - 1, int(len(span_values) * 0.8))]
        if span < 1e-6:
            continue
        out[kind] = {name: item / span for name, item in values.items()}
    return out


def _depth_near_is_large(kind: str, alpha: float) -> bool:
    """The fitted depth semantics: True = large depth value = close to the camera."""
    return (alpha < 0.0) if kind == "linear" else (alpha > 0.0)


def _default_front(point_a: str, point_b: str) -> bool:
    """When the depth map has nothing to say, the front/back direction of this segment follows the anatomical default (the same assumptions as the bone-length fallback).

    Monocular depth models often "smear" thin limbs onto the torso depth (elbow and wrist differ by under 1 cm), and then the sign can only come from common sense:
    elbow backward, hand forward, knee forward - this is also the one from the old fixed assumptions.
    """
    if point_b.startswith("el_"):
        return not FALLBACK_ELBOW_BACK
    if point_b.startswith("wr_"):
        return FALLBACK_ELBOW_BACK
    if point_b.startswith("kn_"):
        return FALLBACK_KNEE_FRONT
    if point_b.startswith("an_"):
        return not FALLBACK_KNEE_FRONT
    return True          # torso leans forward / foot points forward / hand in front of the wrist


def _apply_direction_hint(solutions, best, opts: dict):
    """Use the physical criterion (the person is closer than the background) to pin the depth direction; returns unchanged when there is no hint or the fit really trusts one side.

    Whether the bone-length constraints can tell front from back depends on how strong the perspective is: when the person stands frontally and the body is flat in depth, the two
    signs of alpha give almost the same residual (only 0.1% apart on real photos), so "the direction" is really luck; and flipping the whole pose
    front-to-back looks like "the hand is in front in the depth map, but jumps behind the body after applying".

    ``opts["depth_near_large"]`` is the line of evidence used to colour the depth preview (it depends on the pixels only: whether the depth values inside the person's box
    are overall larger or smaller than outside it), and is fully independent of the bone-length fit. When the cost difference is below :data:`DEPTH_SIGN_TIE_RATIO`
    switch to the candidate solution that agrees with it; when the difference is large the fit really trusts one side, and the fit wins.
    """
    hint = opts.get("depth_near_large")
    if hint is None:
        return best
    cost, kind, _normalized, theta = best
    if _depth_near_is_large(kind, float(theta[0])) == bool(hint):
        return best
    return _pick_direction(solutions, best, bool(hint), DEPTH_SIGN_TIE_RATIO)


def _pick_direction(solutions, best, near_large: bool, ratio: float):
    """Among the candidates pick the best solution with "depth direction = near_large"; if the cost exceeds ratio times, do not take it (return best)."""
    candidates = [item for item in solutions
                  if _depth_near_is_large(item[1], float(item[3][0])) == bool(near_large)]
    if not candidates:
        return best
    alternative = candidates[0]          # solutions are already sorted by cost, so the first one is the best in that direction
    return alternative if alternative[0] <= best[0] * ratio else best


def _segment_signs(kind: str, alpha: float, normalized: dict, constraints, extra=()) -> dict:
    """Front/back sign of every bone segment: ``(point A, point B) -> is B closer to the camera than A`` (from the depth map).

    A value of ``None`` means **the depth map has no opinion** (the depth difference is too small to mean anything, below ``DEPTH_SIGN_MIN_DELTA``
    metres): monocular depth models often smear thin limbs onto the torso depth, with elbow / wrist differences often under 1 cm; then it must neither be taken as "behind"
    (that is exactly what throws the hand behind the body) nor guessed blindly, so the segment is treated as having no front/back information and stays in the image plane.
    """
    extras = [(item["point_a"], item["point_b"]) for item in constraints] + list(extra)
    signs = {}
    for point_a, point_b in extras:
        value_a = normalized.get(point_a)
        value_b = normalized.get(point_b)
        if value_a is None or value_b is None:
            continue
        delta = (value_b - value_a) * alpha
        signs[(point_a, point_b)] = (None if abs(delta) < DEPTH_SIGN_MIN_DELTA
                                     else bool(delta < 0.0))
    return signs


def _ray_depth(parent, u: float, v: float, length: float, cx: float, cy: float,
               focal: float, current_y: float, child_nearer: bool):
    """Find the depth along the child joint's own pixel ray so that the "parent -> child" distance equals the bone length exactly.

    Returning None means this segment cannot be repaired (the in-image distance already exceeds the bone length); the caller falls back on its own.
    """
    ray = np.array(((float(u) - cx) / focal, 1.0, (cy - float(v)) / focal))
    quad_a = float(ray @ ray)
    quad_b = -2.0 * float(ray @ parent)
    quad_c = float(parent @ parent) - length * length
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    if disc <= 0.0 or quad_a <= 1e-12:
        return None
    root = math.sqrt(disc)
    roots = [(-quad_b - root) / (2.0 * quad_a), (-quad_b + root) / (2.0 * quad_a)]
    roots = [value for value in roots if value > 0.1]
    if not roots:
        return None
    same_side = [value for value in roots
                 if ((value - float(parent[1])) < 0.0) == bool(child_nearer)]
    return min(same_side or roots, key=lambda value: abs(value - current_y))


def _repair_depths(joints, pixels, segments, signs, cx: float, cy: float,
                   focal: float) -> int:
    """Segment-level repair: bone lengths from the armature, front/back direction from the depth map (returns the number of repaired segments).

    The depth model gives **relative** depth, whose magnitude is often an order of magnitude smaller than the bone length (measured on a real photo: forearm 26 cm,
    the depth map only gives a 1~2 cm front/back difference). The joint **direction** is then dragged around by the pixels - when the elbow and wrist pixels differ by only a few pixels
    (limb pointing at the camera) the direction is decided almost entirely by those few pixels, and the hand gets thrown behind the body.

    Only repaired when |actual bone length - armature bone length| is clearly too large (more than max(2 cm, 12%)): along the pixel ray of that joint we
    solve for the depth that makes this bone length hold, the sign (who is in front) comes from ``signs`` given by the depth map, and the pixel position does not move,
    so the in-image pose still looks like the photo.
    """
    repaired: set = set()
    for _round in range(len(segments)):
        changed = 0
        for item in segments:
            a = joints.get(item["joint_a"])
            b = joints.get(item["joint_b"])
            pixel = pixels.get(item["point_b"])
            length = float(item["length"])
            if a is None or b is None or pixel is None or length <= 1e-6:
                continue
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if abs(float(np.linalg.norm(b - a)) - length) <= max(DEPTH_REPAIR_TOL,
                                                                 DEPTH_REPAIR_RATIO * length):
                continue
            child_nearer = signs.get((item["point_a"], item["point_b"]), None)
            if child_nearer is None:
                # The depth map has no opinion on this segment (flattened / keypoint not sampled): do not force a front/back direction,
                # stay in the image plane - better to give no depth than to push it backward
                continue
            y = _ray_depth(a, float(pixel[0]), float(pixel[1]), length, cx, cy, focal,
                           float(b[1]), child_nearer)
            if y is None:
                # The in-image distance already exceeds the bone length: no depth adjustment is enough, use the same depth (consistent with the bone-length solve)
                y = float(a[1])
            if abs(y - float(b[1])) < 1e-9:
                continue
            joints[item["joint_b"]] = np.array(((float(pixel[0]) - cx) * y / focal, y,
                                                (cy - float(pixel[1])) * y / focal))
            repaired.add(item["joint_b"])
            changed += 1
        if not changed:
            break
    return len(repaired)


def _solve_depth(pts, hands, metrics, opts, image_size, depth_map, depth_size, face=None):
    """Depth-model-driven solve: the depth of every joint comes straight from the depth map.

    The only unknowns are two **global** parameters: (alpha, beta). alpha is the scale of "normalised relative depth -> metres" (signed,
    so the depth direction is decided by the fit itself), and beta is the distance from the reference joint (pelvis) to the camera. Both are obtained
    from one least-squares fit (Gauss-Newton) of the constraint "3D length of every bone segment in the image = armature bone length",
    not by solving joint by joint from bone lengths, so the "elbow forward or backward" ambiguity does not exist.

    Returning None means the fit is not trustworthy (the caller falls back to the bone-length solve).
    """
    image_w, image_h = float(image_size[0]), float(image_size[1])
    cx, cy = image_w * 0.5, image_h * 0.5
    focal = max(float(opts.get("depth_focal", 1.2) or 1.2) * max(image_w, image_h), 1.0)
    unit = max(float(metrics.get("pelvis_to_shoulders") or 0.5), 0.05)
    constraints = _bone_constraints(metrics)
    if not constraints:
        return None
    # The constraint points decide the "body depth range": when an individual joint samples the background (top of the head misses, etc.) the scale is not skewed
    span_keys = sorted({item["point_a"] for item in constraints}
                       | {item["point_b"] for item in constraints})

    # ---------------- 1) Sample the depth of every joint ----------------
    sample_list = []
    for name in ("pelvis", "chest", "hip_L", "hip_R", "sh_L", "sh_R", "el_L", "el_R",
                 "wr_L", "wr_R", "kn_L", "kn_R", "an_L", "an_R", "big_L", "big_R",
                 "head_base", "head_tip", "eye_a", "eye_b"):
        sample_list.append((name, pts.get(name)))
    # Facial feature groups: the head / neck orientation is voted from them (every pair has its own real depth, see _face_axis_3d).
    # They do not take part in the bone-length fit (they are not in constraints), so they only cost a few extra depth samples.
    for name, point in pts.items():
        if name.startswith("face_") and point is not None:
            sample_list.append((name, point))
    for side in ("L", "R"):
        sample_list.append(("mcp_%s" % side, hands[side].get(coco.HAND_MIDDLE_MCP)))
        if opts.get("fingers"):
            for local in range(coco.HAND_COUNT):
                sample_list.append(("h_%s_%02d" % (side, local), hands[side].get(local)))
    sampled = depth_model.sample_many(depth_map, depth_size, image_size,
                                      [point for _name, point in sample_list])
    depths = {name: float(value) for (name, _point), value in zip(sample_list, sampled)}
    # place() only looks at this one table: the hand keypoints used to exist only in hands, so `place("mcp_L")` always returned None,
    # and the depth of the hand / fingers (even the finger keypoints) was dropped as a whole - not a single finger was generated. They are merged into the same table here.
    anchors = {name: point for name, point in sample_list}

    variants = _depth_variants(depths, "pelvis", span_keys)
    if not variants:
        return None

    # ---------------- 2) Fit (alpha, beta) ----------------
    scale = _orthographic_scale(pts, metrics, image_size, opts)
    beta0 = focal / max(scale, 1e-6)

    def constraint_range(normalized):
        """Normalised depth range of the constraint points (decides the "body depth span")."""
        values = [normalized[name] for name in span_keys if name in normalized]
        if not values:
            return -1.0, 1.0
        return min(values), max(values)

    def residuals(theta, normalized, n_range):
        alpha, beta = float(theta[0]), float(theta[1])
        out = []
        for item in constraints:
            na = normalized.get(item["point_a"])
            nb = normalized.get(item["point_b"])
            point_a = pts.get(item["point_a"])
            point_b = pts.get(item["point_b"])
            if na is None or nb is None or point_a is None or point_b is None:
                continue
            ya, yb = beta + alpha * na, beta + alpha * nb
            if ya <= 1e-3 or yb <= 1e-3:      # a joint went behind the camera: reject
                out.append(1.0)
                continue
            head = _perspective_point(point_a, ya, cx, cy, focal)
            tail = _perspective_point(point_b, yb, cx, cy, focal)
            error = float(np.linalg.norm(tail - head)) - item["length"]
            out.append(_huber(error, max(0.02, 0.1 * item["length"])))
        # Weak regulariser: when the pose is almost flat (depth carries no information) it keeps the depth span from running away (independent of the normalisation scale)
        out.append(3e-3 * math.sqrt(max(len(constraints), 1)) * alpha * n_range / unit)
        return np.asarray(out, dtype=np.float64)

    solutions = []
    for kind, normalized in variants.items():
        low, high = constraint_range(normalized)
        n_range = max(high - low, 1e-6)
        for target in (0.3 * unit, 0.6 * unit, 1.0 * unit):
            # The initial value is based on the "body depth span", independent of the normalisation method
            alpha0 = target / n_range
            for sign in (1.0, -1.0):
                theta, cost = _gauss_newton(
                    lambda value, data=normalized, span=n_range:
                        residuals(value, data, span),
                    np.array((sign * alpha0, beta0), dtype=np.float64))
                if math.isfinite(cost):
                    solutions.append((cost, kind, normalized, theta))
                # Explicitly add the "mirror solution" (alpha negated, beta unchanged = the whole pose flipped front-to-back): the normalisation uses the pelvis as the
                # reference, so mirroring is alpha -> -alpha. Gauss-Newton converges to the same solution from positive and negative starts; without this
                # the candidates would contain only one direction - and both "Flip Depth Direction" and "use the physical criterion to decide the sign" would stop working
                # (that is exactly where the old version broke: "Flip" was ticked but the pose was not actually flipped).
                if abs(float(theta[0])) <= 1e-9:
                    continue
                mirrored = np.array((-float(theta[0]), float(theta[1])), dtype=np.float64)
                mirror_residual = residuals(mirrored, normalized, n_range)
                mirror_cost = float(mirror_residual @ mirror_residual)
                if math.isfinite(mirror_cost):
                    solutions.append((mirror_cost, kind, normalized, mirrored))
    if not solutions:
        return None
    solutions.sort(key=lambda item: item[0])
    best = _apply_direction_hint(solutions, solutions[0], opts)
    if bool(opts.get("depth_flip")):
        # Manual flip: switch to the best candidate with the "opposite depth direction" (the user's explicit choice, highest priority)
        best = _pick_direction(solutions, best,
                              not _depth_near_is_large(best[1], float(best[3][0])),
                              float("inf"))
    cost, kind, normalized, theta = best
    alpha, beta = float(theta[0]), float(theta[1])
    if not (math.isfinite(alpha) and math.isfinite(beta)) or beta <= 0.05 or alpha == 0.0:
        return None
    low, high = constraint_range(normalized)
    depth_span = abs(alpha) * max(high - low, 1e-6)
    if depth_span > 4.0 * unit:            # depth span too large to be a human: judged a failed fit
        return None
    # Whether "a large value" in the depth map is near or far (for the false-colour preview): with linear parameterisation alpha<0 means large value = near
    near_is_large = (alpha < 0.0) if kind == "linear" else (alpha > 0.0)

    # ---------------- 3) Assemble the joints ----------------
    # Joint depths are clamped to "body depth range ±30%": when an individual joint samples the background (top of the head misses, etc.)
    # it is not thrown several metres away
    margin = 0.3 * max(high - low, 1e-6)
    n_low, n_high = low - margin, high + margin

    def place(name, fallback_y=None):
        point = anchors.get(name)
        if point is None:
            return None
        value = normalized.get(name)
        if value is None:
            y = beta if fallback_y is None else float(fallback_y)
        else:
            value = min(max(value, n_low), n_high)
            y = beta + alpha * value
        return _perspective_point(point, max(y, 1e-3), cx, cy, focal)

    joints: dict = {}
    pelvis_offset = float(metrics.get("pelvis_offset_z", 0.0) or 0.0)
    chest_offset = float(metrics.get("chest_offset_z", 0.0) or 0.0)
    joints["pelvis"] = place("pelvis")
    joints["pelvis"][2] += pelvis_offset
    joints["chest"] = place("chest")
    joints["chest"][2] += chest_offset
    spine_points = _resample_polyline([joints["pelvis"], joints["chest"]], 4)

    for side in ("L", "R"):
        joints["hip.%s" % side] = place("hip_%s" % side)
        joints["shoulder.%s" % side] = place("sh_%s" % side)
        joints["elbow.%s" % side] = place("el_%s" % side)
        joints["wrist.%s" % side] = place("wr_%s" % side)
        joints["knee.%s" % side] = place("kn_%s" % side)
        joints["ankle.%s" % side] = place("an_%s" % side)
        joints["ball.%s" % side] = place("big_%s" % side)

        ankle = np.asarray(joints["ankle.%s" % side], dtype=float)
        ball = np.asarray(joints["ball.%s" % side], dtype=float)
        toe_dir = ball - ankle
        span = float(np.linalg.norm(toe_dir))
        toe_len = metrics["toe_len"][side] or 0.05
        joints["toe_tip.%s" % side] = ball + (
            toe_dir / span if span > 1e-9 else np.array((0.0, -1.0, 0.0))) * toe_len

        # Hand: keypoints only reach the finger bases, but they have real depth, which gives a more reliable direction than "reusing the wrist depth"
        hand_tip = place("mcp_%s" % side)
        if hand_tip is None:
            wrist = np.asarray(joints["wrist.%s" % side], dtype=float)
            elbow = np.asarray(joints["elbow.%s" % side], dtype=float)
            direction = wrist - elbow
            span = float(np.linalg.norm(direction))
            hand_len = float(metrics["hand_len"][side] or 0.08)
            hand_tip = wrist + (direction / span if span > 1e-9
                                else np.array((0.0, -1.0, 0.0))) * hand_len
        joints["hand_tip.%s" % side] = np.asarray(hand_tip, dtype=float)

    # ---------------- 4) Segment-level repair: bone lengths from the armature, front/back direction from the depth map ----------------
    # Only repaired when |actual bone length - armature bone length| is clearly too large (when the difference is small the magnitude given by the depth map is kept).
    hand_pairs = [("wr_%s" % side, "mcp_%s" % side) for side in ("L", "R")]
    signs = _segment_signs(kind, alpha, normalized, constraints, hand_pairs)
    repairs = list(constraints)
    for side in ("L", "R"):
        repairs.append({"point_a": "wr_%s" % side, "point_b": "mcp_%s" % side,
                        "joint_a": "wrist.%s" % side, "joint_b": "hand_tip.%s" % side,
                        "length": float(metrics["hand_len"][side] or 0.0)})
    bone_error = _bone_error_rms(joints, constraints)    # residual before the repair: used to judge "whether the depth can be trusted"
    repaired = _repair_depths(joints, anchors, repairs, signs, cx, cy, focal)
    # The repair moves the depth of chest, so the spine polyline must be resampled
    spine_points = _resample_polyline([joints["pelvis"], joints["chest"]], 4)

    # Clavicle and neck have no keypoints of their own and can only follow the torso (consistent with the bone-length mode)
    rest_pelvis = np.asarray(metrics["pelvis"], dtype=float)
    rest_chest = np.asarray(metrics["chest"], dtype=float)
    q_torso = Vector(rest_chest - rest_pelvis).normalized().rotation_difference(
        Vector(joints["chest"] - joints["pelvis"]).normalized())

    def torso_offset(vec):
        return np.asarray(q_torso @ Vector(np.asarray(vec, dtype=float)))

    for side in ("L", "R"):
        clavicle = (metrics.get("clavicle_offsets") or {}).get(side)
        joints["shoulder_origin.%s" % side] = joints["chest"] + (
            torso_offset(clavicle) if clavicle is not None else 0.0)
    neck_offset = metrics.get("neck_offset")
    joints["neck_start"] = joints["chest"] + (torso_offset(neck_offset)
                                              if neck_offset is not None else 0.0)

    head_len = metrics["head_len"] or 0.2
    head_base = place("head_base")
    if head_base is None:
        head_offset = metrics.get("head_offset")
        head_base = joints["chest"] + (torso_offset(head_offset)
                                       if head_offset is not None else 0.0)
    joints["head_base"] = np.asarray(head_base, dtype=float)
    # Head direction: the up axis of the facial features (chin -> eyes / eyebrows / mouth corners, each with its own real depth).
    # The old approach sampled the depth of "the pixel above the forehead": that point often lands on hair / background, which drags the head direction (especially
    # the pitch) around; voting over several pairs of facial points gives nodding / looking up a real front/back component.
    up_axis = _face_axis_3d(place, face, "up_pairs")
    if up_axis is not None:
        joints["head_tip"] = joints["head_base"] + up_axis * head_len
    else:
        head_tip = place("head_tip")
        joints["head_tip"] = (np.asarray(head_tip, dtype=float) if head_tip is not None
                              else joints["head_base"] + np.array((0.0, 0.0, head_len)))

    # Eye line (the roll reference axis of head): paired facial features vote together, each with its own real depth - the in-image component is head tilt,
    # the front/back component is **head turn**, which is the hardest thing to get from a single picture and the easiest to throw off with a sample or two on the eye corners
    refs = _build_refs(joints, _face_axis_3d(place, face, "side_pairs"))
    fingers = _build_fingers_depth(opts, place)
    _align_hands(joints, metrics, opts, fingers)   # last step: the hand (including fingers) follows the forearm
    return {"joints": joints, "refs": refs, "spine_points": spine_points,
            "fingers": fingers, "scale": focal / max(beta, 1e-6), "mode": "DEPTH",
            "depth": {"transform": kind, "alpha": alpha, "beta": beta, "focal": focal,
                      "near_is_large": near_is_large, "cost": float(cost),
                      "bone_error": bone_error, "bone_error_fallback": None,
                      "repaired": repaired, "signs": signs}}


def _build_fingers_depth(opts, place) -> dict:
    """Fingers in depth mode: every finger segment uses the depth of its own pixel (``place`` already understands hand keypoints)."""
    fingers: dict = {}
    if not opts.get("fingers"):
        return fingers
    for side in ("L", "R"):
        per_hand = {}
        for finger, indexes in coco.HAND_GROUPS.items():
            chain = [place("h_%s_%02d" % (side, local)) for local in indexes]
            if all(point is not None for point in chain):
                per_hand[finger] = [np.asarray(point, dtype=float) for point in chain]
        if per_hand:
            fingers[side] = per_hand
    return fingers


# --------------------------------------------------------------------------- main entry
def reconstruct(keypoints, scores, metrics, image_size, options=None,
                depth_map=None, depth_size=None) -> dict:
    """Main entry: returns {'ok', 'joints', 'refs', 'spine_points', ...} (unit = metres, armature scale).

    With depth_mode="DEPTH" and a depth_map given, the "depth model + perspective camera" solve is used (the depth of every joint
    comes from the depth map, and (alpha, beta) are fitted globally over all bone segments of the image); otherwise it falls back to the bone-length solve (old behaviour).
    It also falls back automatically when the bone-length error of the depth solve is clearly larger, and says so in notes.
    """
    opts = default_options(**(options or {}))
    collected = _collect_2d(keypoints, scores, opts)
    if "error" in collected:
        return {"ok": False, "message": collected["error"]}
    pts = collected["points"]
    hands = collected["hands"]
    synth = collected["synth"]
    face = collected.get("face") or {}

    mode = str(opts.get("depth_mode", "DEPTH")).upper()
    if mode not in DEPTH_MODES:
        mode = "DEPTH"
    constraints = _bone_constraints(metrics)
    notes: list[str] = []
    solved = None
    rejected_depth = None
    # Announce when the head base / neck direction changes its source: the panel can then show directly whether the ears or the facial features were used
    if face.get("head_base_source") == "face":
        notes.append("Head base / neck direction estimated from the facial features instead (only one ear detected, or it does not match the facial features)")
    elif face.get("head_base_source") in ("eye", "torso"):
        notes.append("No usable ears or facial features, the head base was extrapolated the old way")

    if mode == "DEPTH":
        if depth_map is None or depth_size is None:
            notes.append("No depth map, fell back to the bone-length solve")
        else:
            solved = _solve_depth(pts, hands, metrics, opts, image_size,
                                  depth_map, depth_size, face=face)
            if solved is None:
                notes.append("The depth solve failed, fell back to the bone-length solve")
            else:
                # The residual of the bone-length solve is for reference only (it forces the bone lengths by construction); the criterion is the residual of the depth fit **before** the repair
                # (after the repair every segment is exactly the armature bone length, which would be meaningless for judging "whether the depth map can be trusted")
                signs = solved["depth"].get("signs") or None
                fallback = _solve_bone_lengths(pts, hands, metrics, opts, image_size,
                                               use_depth=True, signs=signs, face=face)
                depth_error = float(solved["depth"]["bone_error"])
                fallback_error = _bone_error_rms(fallback["joints"], constraints)
                solved["depth"]["bone_error_fallback"] = fallback_error
                if depth_error > MAX_DEPTH_BONE_ERROR:
                    notes.append("The bone-length residual of the depth solve is large (%.3f m > %.3f m), fell back to the bone-length solve"
                                 % (depth_error, MAX_DEPTH_BONE_ERROR))
                    rejected_depth = solved["depth"]
                    solved = fallback
    if solved is None:
        solved = _solve_bone_lengths(pts, hands, metrics, opts, image_size,
                                     use_depth=(mode != "FLAT"), face=face)

    joints = solved["joints"]
    refs = solved["refs"]
    spine_points = solved["spine_points"]
    fingers = solved["fingers"]
    scale = float(solved["scale"])

    # ---------------- Rotate into the armature rest frame according to the person's facing (the character faces -Y) ----------------
    front = FACING_DIRS.get(opts["facing"], FACING_DIRS["FRONT"])
    yaw = math.atan2(float(front[0]), -float(front[1]))
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)

    def rotate_vector(vec):
        return np.array((vec[0] * cos_y - vec[1] * sin_y,
                         vec[0] * sin_y + vec[1] * cos_y, vec[2]))

    pivot = np.asarray(joints["pelvis"], dtype=float).copy()

    def rotate_point(point):
        return rotate_vector(np.asarray(point, dtype=float) - pivot) + pivot

    for name in list(joints.keys()):
        joints[name] = rotate_point(joints[name])
    spine_points = [rotate_point(point) for point in spine_points]
    for name in list(refs.keys()):
        refs[name] = rotate_vector(refs[name])
    for side, per_hand in fingers.items():
        for finger in list(per_hand.keys()):
            per_hand[finger] = [rotate_point(point) for point in per_hand[finger]]

    result = {
        "ok": True,
        "scale": scale,
        "yaw_deg": math.degrees(yaw),
        "mode": solved.get("mode", "BONE"),
        "joints": joints,
        "refs": refs,
        "spine_points": spine_points,
        "fingers": fingers,
        "points2d": pts,
        "synth": sorted(synth),
        "notes": notes,
    }
    if "depth" in solved:
        result["depth"] = solved["depth"]
    elif rejected_depth is not None:
        # Fitted but judged untrustworthy: the parameters are still given (the panel / log shows why it fell back)
        result["depth"] = dict(rejected_depth, rejected=True)
    return result