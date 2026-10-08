# ##### BEGIN GPL LICENSE BLOCK #####
#
# Perfect Overlap Addon
# Version 2.6.0 - Anim Layer (NLA Add track) + key safety nets.
#
# Copyright 2021-2026 CaptainHansode, sakaiden.com
# Copyright 2026 Sanku Venkatesh
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####
#
# 2.6.0
# * Anim Layer option: bake onto a separate NLA track (Add blend) instead of
#   writing into the existing Action (the active Action is pushed down first).
# * Direct bake safety nets: skipped keys are re-inserted, key reduction is
#   checked against the real curve, NLA-animated bones get an explanation.

bl_info = {
    "name": "Perfect Overlap",
    "author": "Sanku Venkatesh",
    "version": (2, 6, 0),
    "blender": (4, 4, 0),
    "location": "3D Viewport > Sidebar > Perfect Overlap (Pose Mode)",
    "description": (
        "Overlap and follow-through animation for bone chains. "
        "Simulates a spring-lagged bone tip and re-aims the bone at it."
    ),
    "category": "Animation",
}

import bpy
import json
import math
import mathutils
import traceback

__author__ = "Sanku Venkatesh"
__copyright__ = (
    "Copyright 2021-2026 CaptainHansode, sakaiden.com; "
    "2026 Sanku Venkatesh"
)
__license__ = "GPL"
__status__ = "Production"

REQUIRED_BLENDER = (4, 4, 0)

EPS = 1.0e-6
ZERO_V = mathutils.Vector((0.0, 0.0, 0.0))
ONE_V = mathutils.Vector((1.0, 1.0, 1.0))
IDENTITY_Q = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))

# The lagged tip / head may drift at most this many bone lengths from where
# the bone would be if it simply followed its parent (1.0 = 90 degrees swing).
MAX_TIP_LAG_RATIO = 1.0
MAX_HEAD_LAG_RATIO = 1.0
MAX_SWING_ANGLE = math.radians(150.0)

# Model check: evaluated pose vs. the solver's own model of the chain.
MODEL_TOLERANCE_LENGTH = 0.02
MODEL_TOLERANCE_ANGLE = math.radians(1.0)

# Below this the result is considered "no visible overlap".
MIN_VISIBLE_OVERLAP = math.radians(0.05)

CYCLE_MODIFIER_NAME = "Perfect Overlap Cycle"

NON_CONTROL_PREFIXES = (
    "def-", "def_", "def.",
    "org-", "org_", "org.",
    "mch-", "mch_", "mch.",
    "wgt-", "wgt_", "wgt.",
    "vis-", "vis_", "vis.",
)

ROTATION_PATHS = (
    "rotation_quaternion",
    "rotation_euler",
    "rotation_axis_angle",
)

# Anim Layer (NLA) settings
LAYER_NAME = "Perfect Overlap"
LAYER_ACTION_PROP = "perfect_overlap_layer"
LAYER_STATIC_PROP = "perfect_overlap_static"
LAYER_VERIFY_ANGLE = math.radians(1.0)
STATIC_TOLERANCE = 1.0e-4

# What an NLA stack evaluates to for a channel that nothing below animates.
DEFAULT_CHANNEL_VALUES = {
    "rotation_quaternion": (1.0, 0.0, 0.0, 0.0),
    "rotation_euler": (0.0, 0.0, 0.0),
    "rotation_axis_angle": (0.0, 0.0, 1.0, 0.0),
    "location": (0.0, 0.0, 0.0),
}


# ============================================================================
# Math helpers
# ============================================================================

def is_finite_vector(v):
    try:
        return all(math.isfinite(float(x)) for x in v)
    except (TypeError, ValueError, AttributeError):
        return False


def is_finite_quaternion(q):
    try:
        return all(math.isfinite(float(x)) for x in q)
    except (TypeError, ValueError, AttributeError):
        return False


def safe_vector(v, fallback=None):
    if fallback is None:
        fallback = ZERO_V
    try:
        out = v.copy()
        if not is_finite_vector(out):
            return fallback.copy()
        return out
    except Exception:
        return fallback.copy()


def safe_quaternion(q, fallback=None):
    """Normalize a quaternion. Uses .magnitude - mathutils.Quaternion has no
    .length (that was the crash bug in 2.3.x). Never raises."""
    if fallback is None:
        fallback = IDENTITY_Q
    try:
        out = q.copy()
        mag = float(out.magnitude)
        if not math.isfinite(mag) or mag < EPS or not is_finite_quaternion(out):
            return fallback.copy()
        out.normalize()
        return out
    except Exception:
        return fallback.copy()


def safe_invert(m):
    """Matrix inverse that never raises (a singular matrix becomes identity)."""
    try:
        return m.inverted()
    except (ValueError, ArithmeticError):
        return mathutils.Matrix.Identity(len(m))


def quat_angle_between(a, b):
    """Smallest rotation angle between two quaternions (handles q == -q)."""
    qa = safe_quaternion(a)
    qb = safe_quaternion(b)
    dot = min(1.0, abs(float(qa.dot(qb))))
    return 2.0 * math.acos(dot)


def axis_angle_to_quaternion(aa):
    try:
        angle = float(aa[0])
        axis = mathutils.Vector((float(aa[1]), float(aa[2]), float(aa[3])))
        if axis.length < EPS or not math.isfinite(angle):
            return IDENTITY_Q.copy()
        axis.normalize()
        return safe_quaternion(mathutils.Quaternion(axis, angle))
    except (TypeError, ValueError, IndexError):
        return IDENTITY_Q.copy()


def current_subframe(scene):
    try:
        return float(scene.frame_subframe)
    except (AttributeError, TypeError, ValueError):
        return 0.0


# ============================================================================
# Solver
# ============================================================================

class PerfectOverlapSolver:
    """Spring-lagged tip overlap solver.

    Pipeline:
        1. build_chains()    pick the chains to overlap from the selection
        2. sample_source()   read the source animation        (read-only)
        3. solve()           simulate the overlap in pure maths
        4. delete_keys()     remove old keys in the range
        5. write_results()   write the new rotation keys

    Model: for a bone with parent P

        pose = P.pose @ (P.rest^-1 @ bone.rest) @ LocRotScale(loc, rot, scale)

    The tail of every selected bone is a spring-lagged point. The target of
    that point is where the tail would be if the bone simply followed its
    (already overlapped) parent with its authored local pose. The bone is then
    re-aimed at the lagged tail and the aim change is converted back into a
    plain local rotation. Parents are solved before children, so the lag
    accumulates down the chain.
    """

    def __init__(self):
        self.sf = 0
        self.ef = 100
        self.delay = 3.0
        self.recursion = 5.0
        self.strength = 1.0
        self.threshold = 0.001
        self.animate_translate = False
        self.debug = False

        self.key_failures = set()
        self.lookup_failures = set()
        self.pose_failures = set()
        self.translate_skipped = []
        self.model_mismatch = []
        self.max_overlap_angle = 0.0
        self.filled_keys = 0
        self.layer_static = []
        self._written = {}

    # ------------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------------

    def configure(self, sf, ef, delay, recursion, strength, threshold,
                  animate_translate, debug=False):
        self.sf = int(sf)
        self.ef = int(ef)
        self.delay = float(delay)
        self.recursion = float(recursion)
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.animate_translate = bool(animate_translate)
        self.debug = bool(debug)
        return self

    def solver_params(self):
        # k is the per-frame pull toward the target. Larger delay = lower k.
        k = 1.0 / max(min(self.delay, 10.0), 1.0)

        # Damping from recursion: 0 -> heavy damping, 10 -> light damping.
        rec = max(0.0, min(10.0, self.recursion))
        damping = 0.76 - (rec / 10.0) * 0.34
        damping = max(0.38, min(0.76, damping))

        # Strength speeds the response up, which adds follow-through.
        strength = max(1.0, min(10.0, self.strength))
        amp = 1.0 + (strength - 1.0) / 9.0
        return k, damping, amp

    def _log(self, msg):
        if self.debug:
            print("[Perfect Overlap] {}".format(msg))

    @staticmethod
    def _fail(collection, message):
        if message:
            collection.add(str(message))

    # ------------------------------------------------------------------------
    # Rig inspection
    # ------------------------------------------------------------------------

    def _visible(self, pbn):
        try:
            if pbn.bone.hide:
                return False
        except (AttributeError, ReferenceError):
            return False
        colls = getattr(pbn.bone, "collections", None)
        if not colls:
            return True
        for c in colls:
            vis = getattr(c, "is_visible_effectively", None)
            if vis is None:
                vis = getattr(c, "is_visible_with_ancestors", None)
            if vis is None:
                vis = getattr(c, "is_visible", True)
            if vis:
                return True
        return False

    def _is_control(self, pbn):
        try:
            if pbn.name.lower().startswith(NON_CONTROL_PREFIXES):
                return False
        except (AttributeError, ReferenceError):
            return False
        return self._visible(pbn)

    def _control_children(self, pbn, depth=0):
        if depth >= 8:
            return []
        result = []
        try:
            children = pbn.children
        except (AttributeError, ReferenceError):
            return result
        for c in children:
            if self._is_control(c):
                result.append(c)
            else:
                result.extend(self._control_children(c, depth + 1))
        return result

    def _next_in_chain(self, pbn, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        for c in candidates:
            try:
                if c.bone.use_connect:
                    return c
            except (AttributeError, ReferenceError):
                pass
        try:
            tail = pbn.tail
            direction = (pbn.tail - pbn.head).normalized()
            scale = max(float(pbn.length), EPS)
        except Exception:
            return candidates[0]
        best, best_score = None, None
        for c in candidates:
            try:
                dist = (c.head - tail).length / scale
                c_vec = c.tail - c.head
                c_dir = c_vec.normalized() if c_vec.length > EPS else -direction
                align = float(c_dir.dot(direction))
                score = dist - align
                if best_score is None or score < best_score:
                    best_score, best = score, c
            except Exception:
                continue
        return best if best is not None else candidates[0]

    @staticmethod
    def _depth(pbn):
        d = 0
        cur = pbn
        while cur is not None:
            d += 1
            cur = cur.parent
        return d

    # ------------------------------------------------------------------------
    # Chain building
    # ------------------------------------------------------------------------

    def build_chains(self):
        """Return a list of chains. Each chain is a list of pose bones ordered
        parents-first. Chains are sorted parents-first as well."""
        obj = bpy.context.active_object
        if obj is None or obj.type != 'ARMATURE':
            return []

        selected = []
        for pbn in (bpy.context.selected_pose_bones or []):
            try:
                if pbn.id_data == obj:
                    selected.append(pbn)
            except (AttributeError, ReferenceError):
                pass
        if not selected:
            return []

        names = {p.name for p in selected}
        ordered = sorted(selected, key=self._depth)
        used = set()
        chains = []

        for pbn in ordered:
            if pbn.name in used:
                continue
            if pbn.parent is None:
                # A parentless control is a driver, not part of a chain.
                continue
            chain = [pbn]
            used.add(pbn.name)
            cur = pbn
            while True:
                cands = [
                    c for c in self._control_children(cur)
                    if c.name in names and c.name not in used
                ]
                nxt = self._next_in_chain(cur, cands)
                if nxt is None:
                    break
                chain.append(nxt)
                used.add(nxt.name)
                cur = nxt
            chains.append(chain)

        chains.sort(key=lambda ch: self._depth(ch[0]))
        return chains

    @staticmethod
    def iter_bones(chains):
        """Unique bones across all chains, parents-first."""
        seen = set()
        for chain in chains:
            for pbn in chain:
                if pbn.name in seen:
                    continue
                seen.add(pbn.name)
                yield pbn

    # ------------------------------------------------------------------------
    # Phase 1 - sampling (read-only)
    # ------------------------------------------------------------------------

    @staticmethod
    def _active_quat(pbn):
        """Quaternion of the bone's ACTIVE rotation channel."""
        mode = pbn.rotation_mode
        try:
            if mode == 'QUATERNION':
                return safe_quaternion(pbn.rotation_quaternion)
            if mode == 'AXIS_ANGLE':
                return axis_angle_to_quaternion(pbn.rotation_axis_angle)
            return safe_quaternion(pbn.rotation_euler.to_quaternion())
        except (ValueError, ArithmeticError, TypeError):
            return IDENTITY_Q.copy()

    def sample_source(self, chains):
        """Read the source animation for every frame in the range.

        Returns a dict:
            frames   {frame: {"world": M4,
                              bone: {"matrix", "local_loc", "local_quat",
                                     "local_scale"}   (chain bones)
                              driver: {"matrix"}}}    (parents outside chain)
            rest_rel {bone: parent.rest^-1 @ bone.rest}
            parent   {bone: parent name}
            length   {bone: rest length}
        or None when there is nothing to read. Only the current frame moves.
        """
        scene = bpy.context.scene
        obj = bpy.context.active_object
        if obj is None:
            return None
        original_frame = scene.frame_current
        original_subframe = current_subframe(scene)

        bones = list(self.iter_bones(chains))
        if not bones:
            return None
        chain_names = {pbn.name for pbn in bones}

        rest_rel = {}
        parent_of = {}
        length = {}
        drivers = {}
        for pbn in bones:
            parent = pbn.parent
            if parent is None:
                continue
            rest_rel[pbn.name] = (
                safe_invert(parent.bone.matrix_local) @ pbn.bone.matrix_local
            )
            parent_of[pbn.name] = parent.name
            length[pbn.name] = max(float(pbn.bone.length), EPS)
            if parent.name not in chain_names:
                drivers[parent.name] = parent
        if not rest_rel:
            return None

        cache = {}
        try:
            for f in range(self.sf, self.ef + 1):
                scene.frame_set(f)
                bpy.context.view_layer.update()
                frame_data = {"world": obj.matrix_world.copy()}
                for pbn in bones:
                    frame_data[pbn.name] = {
                        "matrix": pbn.matrix.copy(),
                        "local_loc": safe_vector(pbn.location),
                        "local_quat": self._active_quat(pbn),
                        "local_scale": safe_vector(pbn.scale, ONE_V),
                        "raw_rotation": tuple(
                            float(v)
                            for v in getattr(pbn, self._rotation_path(pbn))
                        ),
                    }
                for name, pbn in drivers.items():
                    frame_data[name] = {"matrix": pbn.matrix.copy()}
                cache[f] = frame_data
        finally:
            scene.frame_set(original_frame, subframe=original_subframe)
            bpy.context.view_layer.update()

        return {
            "frames": cache,
            "rest_rel": rest_rel,
            "parent": parent_of,
            "length": length,
        }

    def check_model(self, source):
        """Names of bones whose real pose differs from the solver's model.

        The solver assumes pose = parent_pose @ rest_offset @ local_channels.
        That is exact for normal FK controls but not for bones with
        constraints on them or non-default Inherit Rotation / Inherit Scale.
        """
        worst = {}
        for frame_data in source["frames"].values():
            for name, rest_rel in source["rest_rel"].items():
                data = frame_data.get(name)
                parent_data = frame_data.get(source["parent"][name])
                if not data or not parent_data:
                    continue
                parent_pose = parent_data.get("matrix")
                actual = data.get("matrix")
                if parent_pose is None or actual is None:
                    continue
                predicted = (
                    parent_pose
                    @ rest_rel
                    @ mathutils.Matrix.LocRotScale(
                        data["local_loc"],
                        data["local_quat"],
                        data["local_scale"],
                    )
                )
                pos_err = (
                    (predicted.translation - actual.translation).length
                    / source["length"][name]
                )
                ang_err = quat_angle_between(
                    predicted.to_quaternion(), actual.to_quaternion()
                )
                prev = worst.get(name, (0.0, 0.0))
                worst[name] = (max(prev[0], pos_err), max(prev[1], ang_err))
        return sorted(
            n for n, (pos, ang) in worst.items()
            if pos > MODEL_TOLERANCE_LENGTH or ang > MODEL_TOLERANCE_ANGLE
        )

    # ------------------------------------------------------------------------
    # Phase 2 - solving (pure maths)
    # ------------------------------------------------------------------------

    def _translate_axes(self, pbn):
        if not self.animate_translate:
            return []
        try:
            if pbn.bone.use_connect:
                return []
            return [i for i in range(3) if not pbn.lock_location[i]]
        except (AttributeError, ReferenceError):
            return []

    @staticmethod
    def _spring(pos, vel, target, k, damping, amp):
        v = vel * damping + (target - pos) * k
        if not is_finite_vector(v):
            v = ZERO_V.copy()
        return pos + v * amp, v

    @staticmethod
    def _aim_quaternion(pose_x, quat, head, tail, wanted_tail):
        """Local rotation that turns the bone from ``tail`` toward
        ``wanted_tail`` (both seen from ``head``).

        With pose = pose_x @ basis, a world-space turn ``delta`` of the bone
        needs the basis rotation  q' = conj(qx) @ delta @ qx @ q  where qx is
        the rotation of pose_x. Twist about the bone axis stays as authored.
        """
        aim_now = tail - head
        aim_new = wanted_tail - head
        if aim_now.length < EPS or aim_new.length < EPS:
            return quat
        try:
            delta = aim_now.rotation_difference(aim_new)
            if delta.w < 0.0:
                delta.negate()
            angle = float(delta.angle)
            if angle > MAX_SWING_ANGLE:
                delta = IDENTITY_Q.slerp(delta, MAX_SWING_ANGLE / angle)
            qx = pose_x.to_quaternion()
            result = qx.conjugated() @ delta @ qx @ quat
            return safe_quaternion(result, quat)
        except (ValueError, ArithmeticError, TypeError):
            return quat

    def _lag_head(self, st, pose_x, loc, meta, world, world_inv, ctx,
                  snap, reset_velocity):
        """Spring-lag the bone head (optional Translation) as local location."""
        head_world = world @ (pose_x @ loc)

        if snap or st["head"] is None:
            st["head"] = head_world.copy()
            if reset_velocity:
                st["head_vel"] = ZERO_V.copy()
            return loc

        new_head, vel = self._spring(
            st["head"], st["head_vel"], head_world,
            ctx["k"], ctx["damping"], ctx["amp"],
        )
        world_length = (
            world.to_3x3() @ mathutils.Vector((0.0, meta["length"], 0.0))
        ).length
        limit = max(world_length, EPS) * MAX_HEAD_LAG_RATIO
        lag = new_head - head_world
        if lag.length > limit:
            new_head = head_world + lag.normalized() * limit
            vel = ZERO_V.copy()

        # location that puts the head on the lagged point
        wanted = safe_invert(pose_x) @ (world_inv @ new_head)
        delta = wanted - loc
        for ax in range(3):
            if ax not in meta["axes"]:
                delta[ax] = 0.0
        result = loc + delta

        # keep the spring on what was really applied (no wind-up on locks)
        st["head"] = world @ (pose_x @ result)
        st["head_vel"] = vel
        return result

    def _step_frame(self, frame_data, ctx, snap=False, reset_velocity=False):
        """Advance every chain bone by one frame. Pure maths.

        snap=True puts every spring exactly on its target (no lag) and returns
        the authored pose; reset_velocity=True also clears the momentum.
        """
        world = ctx["identity"] if ctx["ignore_world"] else frame_data["world"]
        world_inv = safe_invert(world)
        solved = {}
        out = {}

        for name in ctx["order"]:
            meta = ctx["info"][name]
            src = frame_data.get(name)
            parent_pose = solved.get(meta["parent"])
            if parent_pose is None:
                parent_data = frame_data.get(meta["parent"])
                parent_pose = parent_data.get("matrix") if parent_data else None
            if src is None or parent_pose is None:
                continue

            st = ctx["state"][name]
            loc = src["local_loc"]
            quat = src["local_quat"]
            scale = src["local_scale"]
            pose_x = parent_pose @ meta["rest_rel"]

            use_loc = loc
            if meta["axes"]:
                use_loc = self._lag_head(
                    st, pose_x, loc, meta, world, world_inv, ctx,
                    snap, reset_velocity,
                )

            pose_t = pose_x @ mathutils.Matrix.LocRotScale(use_loc, quat, scale)
            head = pose_t.translation.copy()
            tail = pose_t @ mathutils.Vector((0.0, meta["length"], 0.0))
            tail_world = world @ tail

            new_quat = quat
            if snap or st["tip"] is None:
                st["tip"] = tail_world.copy()
                if reset_velocity:
                    st["tip_vel"] = ZERO_V.copy()
            else:
                new_tip, vel = self._spring(
                    st["tip"], st["tip_vel"], tail_world,
                    ctx["k"], ctx["damping"], ctx["amp"],
                )
                limit = max((tail_world - world @ head).length, EPS)
                limit *= MAX_TIP_LAG_RATIO
                lag = new_tip - tail_world
                if lag.length > limit:
                    new_tip = tail_world + lag.normalized() * limit
                    vel = ZERO_V.copy()
                st["tip"] = new_tip
                st["tip_vel"] = vel
                new_quat = self._aim_quaternion(
                    pose_x, quat, head, tail, world_inv @ new_tip
                )

            out[name] = {
                "rotation": new_quat.copy(),
                "location": use_loc.copy() if meta["axes"] else None,
            }
            # the solved pose of this bone drives its children
            solved[name] = pose_x @ mathutils.Matrix.LocRotScale(
                use_loc, new_quat, scale
            )
        return out

    @staticmethod
    def _copy_frame_result(frame_result):
        return {
            name: {
                "rotation": res["rotation"].copy(),
                "location": (
                    res["location"].copy() if res["location"] is not None else None
                ),
            }
            for name, res in frame_result.items()
        }

    def solve(self, chains, source, cycle=False, preroll=0):
        """Simulate the overlap for the whole range.

        Returns {frame: {bone: {"rotation": Quaternion, "location": Vector or
        None}}} or None when there is nothing to solve. Touches no Blender
        data, so a failure here cannot damage the animation.

        Cycle mode treats the last frame as the same pose as the first,
        settles the momentum with pre-roll passes and records one periodic
        pass. The armature object's own motion is ignored in Cycle mode (a
        loop is assumed to be in place) so the wrap-around is not a teleport.
        """
        frames = list(range(self.sf, self.ef + 1))
        cache = source["frames"]
        if frames[0] not in cache:
            return None

        k, damping, amp = self.solver_params()
        self.translate_skipped = []

        info = {}
        for pbn in self.iter_bones(chains):
            name = pbn.name
            parent_name = source["parent"].get(name)
            rest_rel = source["rest_rel"].get(name)
            if parent_name is None or rest_rel is None:
                continue
            axes = self._translate_axes(pbn)
            if self.animate_translate and not axes:
                self.translate_skipped.append(name)
            info[name] = {
                "parent": parent_name,
                "rest_rel": rest_rel,
                "length": source["length"][name],
                "axes": axes,
                "depth": self._depth(pbn),
            }
        order = sorted(info, key=lambda n: info[n]["depth"])
        if not order:
            return None

        ctx = {
            "order": order,
            "info": info,
            "state": {
                n: {
                    "tip": None, "tip_vel": ZERO_V.copy(),
                    "head": None, "head_vel": ZERO_V.copy(),
                }
                for n in order
            },
            "k": k,
            "damping": damping,
            "amp": amp,
            "ignore_world": bool(cycle),
            "identity": mathutils.Matrix.Identity(4),
        }

        # The first frame is the exact source pose.
        first = self._step_frame(
            cache[frames[0]], ctx, snap=True, reset_velocity=True
        )

        if not cycle:
            results = {frames[0]: first}
            for f in frames[1:]:
                results[f] = self._step_frame(cache[f], ctx)
        else:
            for _ in range(max(1, int(preroll))):
                for f in frames[1:]:
                    self._step_frame(cache[f], ctx)
            results = {}
            for f in frames[1:]:
                results[f] = self._step_frame(cache[f], ctx)
            # last frame == first frame in a loop
            results[frames[0]] = self._copy_frame_result(results[frames[-1]])

        worst = 0.0
        for f, frame_result in results.items():
            for name, res in frame_result.items():
                src = cache[f].get(name)
                if src is not None:
                    worst = max(
                        worst,
                        quat_angle_between(res["rotation"], src["local_quat"]),
                    )
        self.max_overlap_angle = worst

        self._log("solved {} bone(s) over {} frame(s); max offset {:.2f} deg".format(
            len(order), len(frames), math.degrees(worst),
        ))
        return results

    # ------------------------------------------------------------------------
    # Pose capture / restore
    # ------------------------------------------------------------------------

    def capture_pose(self, chains):
        snap = {}
        for pbn in self.iter_bones(chains):
            snap[pbn.name] = {
                "location": pbn.location.copy(),
                "rotation_quaternion": pbn.rotation_quaternion.copy(),
                "rotation_euler": pbn.rotation_euler.copy(),
                "rotation_axis_angle": tuple(
                    float(v) for v in pbn.rotation_axis_angle
                ),
                "scale": pbn.scale.copy(),
            }
        return snap

    def restore_pose(self, chains, snap):
        if not snap:
            return
        for pbn in self.iter_bones(chains):
            vals = snap.get(pbn.name)
            if vals is None:
                continue
            pbn.location = vals["location"].copy()
            pbn.rotation_quaternion = vals["rotation_quaternion"].copy()
            e = vals["rotation_euler"]
            pbn.rotation_euler = mathutils.Euler((e[0], e[1], e[2]), e.order)
            pbn.rotation_axis_angle = vals["rotation_axis_angle"]
            pbn.scale = vals["scale"].copy()
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------------
    # Slotted-Action / F-Curve access
    # ------------------------------------------------------------------------

    def _ensure_action_slot(self, id_data):
        """Make sure the assigned Action has a slot assigned to this ID."""
        if id_data is None:
            return True
        try:
            adt = id_data.animation_data
        except AttributeError:
            return False
        if adt is None or adt.action is None:
            return True

        current = getattr(adt, "action_slot", None)
        if current is not None:
            try:
                if current.target_id_type in ('UNSPECIFIED', id_data.id_type):
                    return True
            except (AttributeError, ReferenceError):
                pass

        try:
            suitable = adt.action_suitable_slots
            if suitable:
                adt.action_slot = suitable[0]
                return True
        except (AttributeError, RuntimeError, TypeError):
            pass

        try:
            adt.action_slot = adt.action.slots.new(
                id_type=id_data.id_type,
                name=id_data.name,
            )
            return True
        except (AttributeError, RuntimeError, TypeError):
            return False

    def _fcurve_container(self, id_data):
        if id_data is None:
            return None
        try:
            adt = id_data.animation_data
        except AttributeError:
            return None
        if adt is None or adt.action is None:
            return None
        if not self._ensure_action_slot(id_data):
            return None

        slot = getattr(adt, "action_slot", None)
        if slot is None:
            return None

        try:
            from bpy_extras import anim_utils
            helper = getattr(
                anim_utils, "animdata_get_channelbag_for_assigned_slot", None,
            )
            channelbag = helper(adt) if helper is not None else None
            if channelbag is None:
                fallback = getattr(
                    anim_utils, "action_get_channelbag_for_slot", None,
                )
                if fallback is not None:
                    channelbag = fallback(adt.action, slot)
            if channelbag is None:
                # No F-Curves for this slot yet. Not an error.
                return None
            return channelbag.fcurves
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            self._fail(self.lookup_failures, "channelbag: {}".format(exc))
            return None

    def _bones_fcurve_map(self, chains):
        obj = bpy.context.active_object
        if obj is None:
            return []
        container = self._fcurve_container(obj)
        if container is None:
            return []
        by_prefix = {
            pbn.path_from_id(): pbn for pbn in self.iter_bones(chains)
        }
        result = []
        for fc in list(container):
            for prefix, pbn in by_prefix.items():
                if fc.data_path.startswith(prefix + "."):
                    result.append((fc, pbn, fc.data_path[len(prefix) + 1:]))
                    break
        return result

    @staticmethod
    def _rotation_path(pbn):
        if pbn.rotation_mode == 'QUATERNION':
            return 'rotation_quaternion'
        if pbn.rotation_mode == 'AXIS_ANGLE':
            return 'rotation_axis_angle'
        return 'rotation_euler'

    def _baked_curves(self, chains):
        """(fcurve, pbn, channel) for the channels this add-on writes."""
        result = []
        for fc, pbn, channel in self._bones_fcurve_map(chains):
            if channel == self._rotation_path(pbn):
                result.append((fc, pbn, channel))
            elif (
                channel == 'location'
                and fc.array_index in self._translate_axes(pbn)
            ):
                result.append((fc, pbn, channel))
        return result

    # ------------------------------------------------------------------------
    # Phase 3 - writing
    # ------------------------------------------------------------------------

    def _apply_rotation(self, pbn, quat, tracker):
        """Write the local rotation in the bone's active rotation mode, keeping
        it continuous with the previous frame's representation."""
        q = safe_quaternion(quat)
        mode = pbn.rotation_mode
        try:
            if mode == 'QUATERNION':
                prev = tracker.get("last_quat")
                if prev is not None and q.dot(prev) < 0.0:
                    q.negate()
                pbn.rotation_quaternion = q
                tracker["last_quat"] = q.copy()
                return

            if mode == 'AXIS_ANGLE':
                angle = float(q.angle)
                axis = q.axis.copy()
                prev = tracker.get("last_axis_angle")
                if prev is not None:
                    prev_angle = float(prev[0])
                    prev_axis = mathutils.Vector(prev[1:4])
                    if prev_axis.length >= EPS:
                        prev_axis.normalize()
                        if axis.dot(prev_axis) < 0.0:
                            axis.negate()
                            angle = -angle
                    angle += math.tau * round((prev_angle - angle) / math.tau)
                pbn.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
                tracker["last_axis_angle"] = (angle, axis.x, axis.y, axis.z)
                return

            euler = q.to_euler(mode)
            prev = tracker.get("last_euler")
            if prev is not None:
                try:
                    euler.make_compatible(prev)
                except (ValueError, ArithmeticError):
                    pass
            pbn.rotation_euler = euler
            tracker["last_euler"] = euler.copy()
        except Exception as exc:
            self._fail(
                self.pose_failures, "{}: rotation {}".format(pbn.name, exc),
            )

    def write_results(self, chains, results, start_pose):
        """Write the solved rotations (and optional locations) as keys."""
        scene = bpy.context.scene
        obj = bpy.context.active_object

        bones = {pbn.name: pbn for pbn in self.iter_bones(chains)}

        # one continuity tracker per bone, seeded from the authored start pose
        trackers = {}
        for name in bones:
            t = {}
            seed = start_pose.get(name) if start_pose else None
            if seed:
                try:
                    t["last_quat"] = safe_quaternion(seed["rotation_quaternion"])
                    t["last_euler"] = seed["rotation_euler"].copy()
                    t["last_axis_angle"] = tuple(
                        float(v) for v in seed["rotation_axis_angle"]
                    )
                except (KeyError, AttributeError, TypeError, ValueError):
                    t = {}
            trackers[name] = t

        self._ensure_action_slot(obj)

        for f in range(self.sf, self.ef + 1):
            frame_result = results.get(f)
            if not frame_result:
                continue
            scene.frame_set(f)
            for name, pbn in bones.items():
                res = frame_result.get(name)
                if res is None:
                    continue
                self._apply_rotation(pbn, res["rotation"], trackers[name])
                if res["location"] is not None:
                    try:
                        pbn.location = res["location"].copy()
                    except Exception as exc:
                        self._fail(
                            self.pose_failures,
                            "{}: location {}".format(name, exc),
                        )
                self._key_bone(pbn)
            bpy.context.view_layer.update()

        if not self._ensure_action_slot(obj):
            self._fail(self.key_failures, "Action slot")

    def _key_bone(self, pbn):
        f = bpy.context.scene.frame_current

        path = self._rotation_path(pbn)
        self._written[(pbn.name, path, f)] = tuple(
            float(v) for v in getattr(pbn, path)
        )
        if self._translate_axes(pbn):
            self._written[(pbn.name, "location", f)] = tuple(
                float(v) for v in pbn.location
            )

        for ax in self._translate_axes(pbn):
            try:
                r = pbn.keyframe_insert(data_path='location', index=ax, frame=f)
                if r is False:
                    self._fail(
                        self.key_failures, "{} loc[{}]".format(pbn.name, ax),
                    )
            except Exception as exc:
                self._fail(
                    self.key_failures,
                    "{} loc[{}]: {}".format(pbn.name, ax, exc),
                )

        try:
            r = pbn.keyframe_insert(
                data_path=self._rotation_path(pbn), frame=f,
            )
            if r is False:
                self._fail(self.key_failures, "{} rotation".format(pbn.name))
        except Exception as exc:
            self._fail(
                self.key_failures, "{} rotation: {}".format(pbn.name, exc),
            )

    # ------------------------------------------------------------------------
    # Key deletion / reduction
    # ------------------------------------------------------------------------

    def delete_keys(self, chains):
        """Remove rotation keys (and location keys on the unlocked axes when
        Translation is on) inside [start, end]. Returns the number removed."""
        obj = bpy.context.active_object
        container = self._fcurve_container(obj)
        emptied = []
        removed = 0

        for fc, pbn, channel in self._bones_fcurve_map(chains):
            if channel == 'location':
                if fc.array_index not in self._translate_axes(pbn):
                    continue
            elif channel not in ROTATION_PATHS:
                continue

            indices = [
                i for i, k in enumerate(fc.keyframe_points)
                if self.sf <= float(k.co[0]) <= self.ef
            ]
            for i in reversed(indices):
                try:
                    fc.keyframe_points.remove(fc.keyframe_points[i], fast=True)
                    removed += 1
                except (RuntimeError, ReferenceError):
                    pass
            try:
                fc.update()
            except Exception:
                pass
            if len(fc.keyframe_points) == 0:
                emptied.append(fc)

        if container is not None:
            for fc in emptied:
                try:
                    container.remove(fc)
                except (RuntimeError, ReferenceError):
                    pass
        return removed

    @staticmethod
    def _max_deviation(values, frames, left, right):
        if right - left <= 1:
            return 0.0, None
        t0, t1 = frames[left], frames[right]
        span = t1 - t0
        if abs(span) < EPS:
            return 0.0, None
        worst, worst_i = -1.0, None
        for i in range(left + 1, right):
            ratio = (frames[i] - t0) / span
            err = 0.0
            for comp in values:
                pred = comp[left] + (comp[right] - comp[left]) * ratio
                err = max(err, abs(comp[i] - pred))
            if err > worst:
                worst, worst_i = err, i
        return worst, worst_i

    def _keep_indices(self, values, frames, tol):
        """Douglas-Peucker: every dropped key stays within ``tol`` of the
        straight line between the keys that are kept."""
        n = len(frames)
        if n <= 2:
            return set(range(n))
        keep = {0, n - 1}
        stack = [(0, n - 1)]
        while stack:
            left, right = stack.pop()
            err, idx = self._max_deviation(values, frames, left, right)
            if idx is None or err <= tol:
                continue
            keep.add(idx)
            stack.append((left, idx))
            stack.append((idx, right))
        return keep

    def reduce_keys(self, chains, curves=None):
        """Key reduction on the channels this add-on wrote (or on ``curves``, a
        list of (fcurve, pbn, channel)). Returns the number of keys removed."""
        if curves is None:
            curves = self._baked_curves(chains)
        groups = {}
        for fc, pbn, channel in curves:
            groups.setdefault((pbn.name, channel), []).append((fc, pbn))

        removed = 0
        for (name, channel), members in groups.items():
            fcurves = [fc for fc, _ in members]
            pbn = members[0][1]
            if channel == 'location':
                tol = max(self.threshold * max(float(pbn.length), EPS), 1.0e-8)
            else:
                tol = max(self.threshold, 1.0e-6)

            frames = None
            for fc in fcurves:
                cf = [
                    float(k.co[0]) for k in fc.keyframe_points
                    if self.sf <= float(k.co[0]) <= self.ef
                ]
                if frames is None:
                    frames = cf
                else:
                    common = set(cf)
                    frames = [x for x in frames if x in common]
            if not frames or len(frames) <= 2:
                self._smooth(fcurves)
                continue

            values = []
            complete = True
            for fc in fcurves:
                lookup = {
                    float(k.co[0]): float(k.co[1]) for k in fc.keyframe_points
                }
                try:
                    values.append([lookup[x] for x in frames])
                except KeyError:
                    complete = False
                    break
            if not complete:
                self._fail(
                    self.lookup_failures,
                    "{} {} incomplete key set".format(name, channel),
                )
                self._smooth(fcurves)
                continue

            keep = self._keep_indices(values, frames, tol)
            drop = {frames[i] for i in range(len(frames)) if i not in keep}
            for fc in fcurves:
                kps = fc.keyframe_points
                idx = [
                    n for n, k in enumerate(kps) if float(k.co[0]) in drop
                ]
                for n in reversed(idx):
                    try:
                        kps.remove(kps[n], fast=True)
                        removed += 1
                    except (RuntimeError, ReferenceError):
                        pass
                try:
                    fc.update()
                except Exception:
                    pass
            self._smooth(fcurves)
            removed -= self._ensure_fidelity(fcurves, frames, values, tol)
        return removed

    def _ensure_fidelity(self, fcurves, frames, values, tol):
        """Blender evaluates the reduced keys with Bezier handles, not with the
        straight lines the reduction assumed. Check the real curve against the
        dense bake and put back the key with the largest error until it holds.
        Returns how many keys were restored."""
        if any(len(fc.modifiers) for fc in fcurves):
            return 0
        limit = tol * 2.0 + 1.0e-9
        restored = 0
        for _ in range(len(frames)):
            worst, worst_i = 0.0, None
            for i, f in enumerate(frames):
                for fc, comp in zip(fcurves, values):
                    try:
                        err = abs(float(fc.evaluate(f)) - comp[i])
                    except Exception:
                        return restored
                    if err > worst:
                        worst, worst_i = err, i
            if worst_i is None or worst <= limit:
                break
            for fc, comp in zip(fcurves, values):
                try:
                    fc.keyframe_points.insert(
                        frames[worst_i], comp[worst_i], options={'FAST'},
                    )
                    fc.update()
                except Exception:
                    return restored
            self._smooth(fcurves)
            restored += 1
        return restored

    def _smooth(self, fcurves):
        for fc in fcurves:
            for k in fc.keyframe_points:
                fr = float(k.co[0])
                if self.sf <= fr <= self.ef:
                    k.interpolation = 'BEZIER'
                    k.handle_left_type = 'AUTO_CLAMPED'
                    k.handle_right_type = 'AUTO_CLAMPED'
            try:
                fc.update()
            except Exception:
                pass

    # ------------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------------

    def _seam_keys(self, fc):
        first = last = None
        for k in fc.keyframe_points:
            fr = float(k.co[0])
            if abs(fr - self.sf) < EPS:
                first = k
            elif abs(fr - self.ef) < EPS:
                last = k
        return first, last

    def close_cycle_seam(self, chains, curves=None):
        """Make the end key equal the start key on every baked curve."""
        for fc, _pbn, _ch in (curves if curves is not None else self._baked_curves(chains)):
            first, last = self._seam_keys(fc)
            if first is None or last is None or first is last:
                continue
            last.co[1] = first.co[1]
            try:
                fc.update()
            except Exception:
                pass

    def smooth_cycle_seam(self, chains, curves=None):
        """Give the seam keys a shared tangent so the loop point is smooth."""
        for fc, _pbn, _ch in (curves if curves is not None else self._baked_curves(chains)):
            first, last = self._seam_keys(fc)
            if first is None or last is None or first is last:
                continue
            keys = sorted(fc.keyframe_points, key=lambda k: float(k.co[0]))
            inner = [k for k in keys if self.sf < float(k.co[0]) < self.ef]
            if len(inner) < 2:
                continue
            after, before = inner[0], inner[-1]
            dt_after = float(after.co[0]) - float(self.sf)
            dt_before = float(self.ef) - float(before.co[0])
            if dt_after < EPS or dt_before < EPS:
                continue
            slope = (
                (float(after.co[1]) - float(before.co[1]))
                / (dt_after + dt_before)
            )
            for key in (first, last):
                key.interpolation = 'BEZIER'
                key.handle_left_type = 'FREE'
                key.handle_right_type = 'FREE'
            first.handle_left = (
                float(self.sf) - dt_before / 3.0,
                float(first.co[1]) - slope * dt_before / 3.0,
            )
            first.handle_right = (
                float(self.sf) + dt_after / 3.0,
                float(first.co[1]) + slope * dt_after / 3.0,
            )
            last.handle_left = (
                float(self.ef) - dt_before / 3.0,
                float(last.co[1]) - slope * dt_before / 3.0,
            )
            last.handle_right = (
                float(self.ef) + dt_after / 3.0,
                float(last.co[1]) + slope * dt_after / 3.0,
            )
            try:
                fc.update()
            except Exception:
                pass

    def add_cycle_modifiers(self, chains):
        for fc, _pbn, _ch in self._baked_curves(chains):
            if any(m.type == 'CYCLES' for m in fc.modifiers):
                continue
            try:
                mod = fc.modifiers.new('CYCLES')
            except (RuntimeError, ReferenceError):
                continue
            try:
                mod.name = CYCLE_MODIFIER_NAME
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                fc.update()
            except Exception:
                pass

    def remove_cycle_modifiers(self, chains):
        """Remove this add-on's Cycles modifiers from the chain's curves."""
        for fc, _pbn, _ch in self._bones_fcurve_map(chains):
            for m in list(fc.modifiers):
                if (
                    m.type == 'CYCLES'
                    and getattr(m, "name", None) == CYCLE_MODIFIER_NAME
                ):
                    try:
                        fc.modifiers.remove(m)
                    except (RuntimeError, ReferenceError):
                        pass


    # ------------------------------------------------------------------------
    # Direct bake: safety nets
    # ------------------------------------------------------------------------

    def fill_missing_keys(self, chains):
        """Re-insert baked keys that keyframe_insert silently skipped (locked
        curves, keying preferences, ...). Returns how many were added."""
        obj = bpy.context.active_object
        container = self._fcurve_container(obj)
        existing = {
            (fc.data_path, int(fc.array_index)): fc
            for fc, _pbn, _ch in self._baked_curves(chains)
        }
        frames = list(range(self.sf, self.ef + 1))
        filled = 0
        for pbn in self.iter_bones(chains):
            path = self._rotation_path(pbn)
            count = 4 if pbn.rotation_mode in ('QUATERNION', 'AXIS_ANGLE') else 3
            channels = [(path, list(range(count)))]
            axes = self._translate_axes(pbn)
            if axes:
                channels.append(("location", axes))
            for channel, indices in channels:
                data_path = pbn.path_from_id() + "." + channel
                for idx in indices:
                    fc = existing.get((data_path, idx))
                    if fc is None and container is not None:
                        try:
                            fc = container.new(
                                data_path, index=idx, group_name=pbn.name,
                            )
                        except (RuntimeError, TypeError, AttributeError):
                            fc = None
                    if fc is None:
                        continue
                    have = {round(float(k.co[0]), 3) for k in fc.keyframe_points}
                    added = False
                    for f in frames:
                        if round(float(f), 3) in have:
                            continue
                        vals = self._written.get((pbn.name, channel, f))
                        if vals is None or idx >= len(vals):
                            continue
                        fc.keyframe_points.insert(f, vals[idx], options={'FAST'})
                        filled += 1
                        added = True
                    if added:
                        try:
                            fc.update()
                        except Exception:
                            pass
        return filled

    def nla_animates_chain(self, obj, chains):
        """True when NLA strips (not our own layers) animate the chain bones."""
        adt = getattr(obj, "animation_data", None)
        if adt is None:
            return False
        prefixes = tuple(
            pbn.path_from_id() + "." for pbn in self.iter_bones(chains)
        )
        try:
            for track in adt.nla_tracks:
                for strip in track.strips:
                    act = strip.action
                    if act is None or act.get(LAYER_ACTION_PROP):
                        continue
                    cb = self._channelbag_for(
                        act, getattr(strip, "action_slot", None),
                    )
                    if cb is None:
                        continue
                    for fc in cb.fcurves:
                        if fc.data_path.startswith(prefixes):
                            return True
        except (AttributeError, RuntimeError, TypeError):
            pass
        return False

    # ------------------------------------------------------------------------
    # Anim Layer (NLA track with Add blending)
    #
    # NLA "Add" adds the strip's values onto what the stack below evaluates to.
    # So the layer stores  new - base  for every channel, where base is what the
    # stack below gives (the sampled source) or, for a channel nothing below
    # animates, Blender's default value (identity quaternion, 0 location, ...).
    # ------------------------------------------------------------------------

    def _channelbag_for(self, action, slot):
        if action is None:
            return None
        try:
            from bpy_extras import anim_utils
        except ImportError:
            return None
        try:
            if slot is not None:
                cb = anim_utils.action_get_channelbag_for_slot(action, slot)
                if cb is not None:
                    return cb
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            for layer in action.layers:
                for strip in layer.strips:
                    for cb in strip.channelbags:
                        return cb
        except (AttributeError, RuntimeError, TypeError):
            pass
        return None

    def check_layer_ready(self, obj):
        adt = getattr(obj, "animation_data", None)
        if adt is not None and getattr(adt, "use_tweak_mode", False):
            raise RuntimeError(
                "Leave NLA tweak mode (Tab in the NLA editor) before baking "
                "an Anim Layer"
            )

    def animated_channels(self, obj):
        """{(data_path, index)} animated by the stack that will sit below the
        layer: the active Action and every NLA strip that is not our own."""
        paths = set()
        adt = getattr(obj, "animation_data", None)
        if adt is None:
            return paths
        sources = []
        if adt.action is not None:
            sources.append((adt.action, getattr(adt, "action_slot", None)))
        for track in adt.nla_tracks:
            for strip in track.strips:
                act = strip.action
                if act is None or act.get(LAYER_ACTION_PROP):
                    continue
                sources.append((act, getattr(strip, "action_slot", None)))
        for act, slot in sources:
            cb = self._channelbag_for(act, slot)
            if cb is None:
                continue
            for fc in cb.fcurves:
                if getattr(fc, "mute", False):
                    continue
                paths.add((fc.data_path, int(fc.array_index)))
        return paths

    def _layer_strips(self, obj):
        adt = getattr(obj, "animation_data", None)
        if adt is None:
            return
        for track in list(adt.nla_tracks):
            for strip in list(track.strips):
                act = strip.action
                if act is not None and act.get(LAYER_ACTION_PROP):
                    yield track, strip

    def layer_previous_curves(self, chains):
        """F-Curves of the chain bones inside earlier Perfect Overlap layers."""
        obj = bpy.context.active_object
        prefixes = tuple(
            pbn.path_from_id() + "." for pbn in self.iter_bones(chains)
        )
        items = []
        for track, strip in self._layer_strips(obj):
            cb = self._channelbag_for(
                strip.action, getattr(strip, "action_slot", None),
            )
            if cb is None:
                continue
            for fc in list(cb.fcurves):
                if fc.data_path.startswith(prefixes):
                    items.append({
                        "fc": fc, "cb": cb, "track": track, "strip": strip,
                        "action": strip.action, "mute": bool(fc.mute),
                    })
        return items

    def _apply_static_values(self, items, chains):
        """Put the un-keyed channels the old layer(s) drove back to the pose they
        had before the layer existed. Without this they keep whatever the layer
        last evaluated to once the layer is muted or removed."""
        obj = bpy.context.active_object
        prefixes = tuple(
            pbn.path_from_id() + "." for pbn in self.iter_bones(chains)
        )
        seen = set()
        for it in items:
            action = it["action"]
            if id(action) in seen:
                continue
            seen.add(id(action))
            try:
                data = json.loads(action.get(LAYER_STATIC_PROP) or "{}")
            except (TypeError, ValueError):
                continue
            for key, value in data.items():
                data_path, _sep, idx = key.rpartition("|")
                if not data_path.startswith(prefixes):
                    continue
                try:
                    name = data_path.split('"')[1]
                    channel = data_path.rsplit(".", 1)[1]
                    getattr(obj.pose.bones.get(name), channel)[int(idx)] = float(value)
                except (IndexError, KeyError, AttributeError, TypeError, ValueError):
                    continue
        bpy.context.view_layer.update()

    def layer_mute_previous(self, chains):
        """Mute the old layer curves so sampling sees the pure base animation."""
        items = self.layer_previous_curves(chains)
        for it in items:
            it["fc"].mute = True
        if items:
            self._apply_static_values(items, chains)
        return items

    @staticmethod
    def layer_unmute(items):
        for it in items:
            try:
                it["fc"].mute = it["mute"]
            except (ReferenceError, AttributeError, RuntimeError):
                pass

    def layer_commit_previous(self, items):
        """Delete the old layer curves (and any strip/track/action left empty)."""
        obj = bpy.context.active_object
        adt = obj.animation_data
        for it in items:
            try:
                it["cb"].fcurves.remove(it["fc"])
            except (RuntimeError, ReferenceError, TypeError):
                pass
        done = set()
        for it in items:
            key = id(it["strip"])
            if key in done:
                continue
            done.add(key)
            try:
                if len(it["cb"].fcurves) > 0:
                    continue
                track, action = it["track"], it["action"]
                track.strips.remove(it["strip"])
                if len(track.strips) == 0:
                    adt.nla_tracks.remove(track)
                if action.users == 0:
                    bpy.data.actions.remove(action)
            except (RuntimeError, ReferenceError, TypeError, AttributeError):
                pass

    @staticmethod
    def rename_layer_track(track):
        """Blender numbers duplicate names; once the old layer is gone take the
        plain name back."""
        adt = bpy.context.active_object.animation_data
        try:
            if track.name != LAYER_NAME and not any(
                t.name == LAYER_NAME for t in adt.nla_tracks
            ):
                track.name = LAYER_NAME
        except (AttributeError, RuntimeError, ReferenceError):
            pass

    def layer_remove(self, chains):
        """Delete Keys with Anim Layer on: take the bones out of the layer(s)."""
        items = self.layer_previous_curves(chains)
        if items:
            self._apply_static_values(items, chains)
        self.layer_commit_previous(items)
        return len(items)

    @staticmethod
    def _make_tracker(seed):
        """Continuity tracker seeded from a capture_pose() entry."""
        tracker = {}
        if not seed:
            return tracker
        try:
            tracker["last_quat"] = safe_quaternion(seed["rotation_quaternion"])
            tracker["last_euler"] = seed["rotation_euler"].copy()
            tracker["last_axis_angle"] = tuple(
                float(v) for v in seed["rotation_axis_angle"]
            )
        except (KeyError, AttributeError, TypeError, ValueError):
            return {}
        return tracker

    def _layer_rotation_values(self, pbn, quat, raw, anim, tracker):
        """New rotation in the bone's own representation, aligned with the base
        value (``raw``) so that  new - base  stays small."""
        q = safe_quaternion(quat)
        mode = pbn.rotation_mode
        any_anim = any(anim)

        if mode == 'QUATERNION':
            ref = None
            if any_anim and len(raw) == 4:
                r = mathutils.Quaternion(raw)
                if r.magnitude > EPS:
                    ref = r
            if ref is None:
                ref = tracker.get("last_quat")
            if ref is not None and q.dot(ref) < 0.0:
                q.negate()
            tracker["last_quat"] = q.copy()
            return tuple(float(v) for v in q)

        if mode == 'AXIS_ANGLE':
            angle = float(q.angle)
            axis = q.axis.copy()
            prev = tuple(raw) if any_anim else tracker.get("last_axis_angle")
            if prev is not None:
                prev_axis = mathutils.Vector(prev[1:4])
                if prev_axis.length >= EPS:
                    prev_axis.normalize()
                    if axis.dot(prev_axis) < 0.0:
                        axis.negate()
                        angle = -angle
                angle += math.tau * round((float(prev[0]) - angle) / math.tau)
            tracker["last_axis_angle"] = (angle, axis.x, axis.y, axis.z)
            return (angle, axis.x, axis.y, axis.z)

        euler = q.to_euler(mode)
        ref = (
            mathutils.Euler(tuple(raw), mode) if any_anim
            else tracker.get("last_euler")
        )
        if ref is not None:
            try:
                euler.make_compatible(ref)
            except (ValueError, ArithmeticError):
                pass
        values = [float(euler[0]), float(euler[1]), float(euler[2])]
        for i in range(3):
            if anim[i]:
                values[i] += math.tau * round((raw[i] - values[i]) / math.tau)
        tracker["last_euler"] = mathutils.Euler(tuple(values), mode)
        return tuple(values)

    def build_layer_curves(self, chains, source, results, start_pose, animated):
        """Pure maths: {(bone, data_path, index): {frame: value}} for the layer,
        plus the bones whose un-keyed pose is not the default pose."""
        curves = {}
        statics = []
        static_values = {}
        frames = range(self.sf, self.ef + 1)
        for pbn in self.iter_bones(chains):
            name = pbn.name
            path = self._rotation_path(pbn)
            data_path = pbn.path_from_id() + "." + path
            loc_path = pbn.path_from_id() + ".location"
            default = DEFAULT_CHANNEL_VALUES[path]
            loc_default = DEFAULT_CHANNEL_VALUES["location"]
            n = len(default)
            anim = [(data_path, i) in animated for i in range(n)]
            loc_anim = [(loc_path, i) in animated for i in range(3)]
            axes = self._translate_axes(pbn)
            tracker = self._make_tracker(start_pose.get(name) if start_pose else None)
            is_static = False
            for f in frames:
                res = results.get(f, {}).get(name)
                src = source["frames"][f].get(name)
                if res is None or src is None:
                    continue
                raw = src["raw_rotation"]
                values = self._layer_rotation_values(
                    pbn, res["rotation"], raw, anim, tracker,
                )
                for i in range(n):
                    base = raw[i] if anim[i] else default[i]
                    if not anim[i]:
                        static_values.setdefault("{}|{}".format(data_path, i), raw[i])
                    if not anim[i] and abs(raw[i] - default[i]) > STATIC_TOLERANCE:
                        is_static = True
                    curves.setdefault((name, data_path, i), {})[f] = values[i] - base
                if res["location"] is not None:
                    for ax in axes:
                        raw_loc = float(src["local_loc"][ax])
                        base = raw_loc if loc_anim[ax] else loc_default[ax]
                        if not loc_anim[ax]:
                            static_values.setdefault("{}|{}".format(loc_path, ax), raw_loc)
                        if not loc_anim[ax] and abs(raw_loc - loc_default[ax]) > STATIC_TOLERANCE:
                            is_static = True
                        curves.setdefault((name, loc_path, ax), {})[f] = (
                            float(res["location"][ax]) - base
                        )
            if is_static:
                statics.append(name)
        return curves, statics, static_values

    def _restore_action(self, adt, action, slot):
        adt.action = action
        if slot is not None:
            try:
                adt.action_slot = slot
            except (AttributeError, RuntimeError, TypeError):
                pass

    def _push_down_active_action(self, adt, undo):
        """Move the active Action into its own NLA track (Blender's Push Down)
        so a track above it can add to it."""
        action = adt.action
        if action is None:
            return None
        slot = getattr(adt, "action_slot", None)
        try:
            start = float(action.frame_range[0])
        except (AttributeError, TypeError, IndexError):
            start = float(self.sf)

        track = adt.nla_tracks.new()
        undo.append(lambda: adt.nla_tracks.remove(track))
        try:
            track.name = action.name
        except (AttributeError, RuntimeError):
            pass
        strip = track.strips.new(action.name, int(math.floor(start)), action)
        if slot is not None and hasattr(strip, "action_slot"):
            try:
                strip.action_slot = slot
            except (AttributeError, RuntimeError, TypeError):
                pass
            if strip.action_slot != slot:
                raise RuntimeError(
                    "Could not hand the Action slot to the pushed-down strip"
                )
        for attr_strip, attr_adt in (
            ("blend_type", "action_blend_type"),
            ("extrapolation", "action_extrapolation"),
        ):
            try:
                setattr(strip, attr_strip, getattr(adt, attr_adt))
            except (AttributeError, RuntimeError, TypeError):
                pass
        try:
            if abs(float(strip.frame_start) - start) > 1.0e-6:
                strip.frame_start = start
        except (AttributeError, RuntimeError, TypeError):
            pass
        adt.action = None
        undo.append(lambda: self._restore_action(adt, action, slot))
        return strip

    def _verify_layer(self, chains, results):
        """Evaluate the NLA and compare with what was baked."""
        scene = bpy.context.scene
        span = self.ef - self.sf
        frames = sorted({
            min(self.ef, self.sf + 1),
            self.sf + span // 2,
            max(self.sf, self.ef - 1),
        })
        for f in frames:
            scene.frame_set(f)
            bpy.context.view_layer.update()
            frame_result = results.get(f) or {}
            for pbn in self.iter_bones(chains):
                res = frame_result.get(pbn.name)
                if res is None:
                    continue
                err = quat_angle_between(self._active_quat(pbn), res["rotation"])
                if err > LAYER_VERIFY_ANGLE:
                    raise RuntimeError(
                        "Anim Layer check failed on '{}' at frame {} "
                        "({:.1f} deg off)".format(pbn.name, f, math.degrees(err))
                    )
                if res["location"] is not None:
                    limit = 1.0e-3 * max(float(pbn.length), EPS) + 1.0e-4
                    for ax in self._translate_axes(pbn):
                        if abs(float(pbn.location[ax]) - float(res["location"][ax])) > limit:
                            raise RuntimeError(
                                "Anim Layer check failed on '{}' location at "
                                "frame {}".format(pbn.name, f)
                            )

    def write_layer(self, chains, source, results, start_pose, cycle=False):
        """Bake onto a NEW NLA track with Add blending.

        Transactional: if anything fails, every change made here is undone
        (tracks and the layer Action removed, the original Action re-assigned)
        and the error is re-raised."""
        from bpy_extras import anim_utils

        obj = bpy.context.active_object
        animated = self.animated_channels(obj)
        curves, statics, static_values = self.build_layer_curves(
            chains, source, results, start_pose, animated,
        )
        if not curves:
            raise RuntimeError("Nothing to write to the layer")
        self.layer_static = statics

        undo = []
        try:
            adt = obj.animation_data
            if adt is None:
                adt = obj.animation_data_create()
                undo.append(obj.animation_data_clear)
            self._push_down_active_action(adt, undo)

            action = bpy.data.actions.new(
                "{}_{}".format(obj.name, LAYER_NAME.replace(" ", ""))
            )
            undo.append(lambda: bpy.data.actions.remove(action))
            action[LAYER_ACTION_PROP] = 1
            action[LAYER_STATIC_PROP] = json.dumps(static_values)
            slot = action.slots.new(id_type='OBJECT', name=obj.name)
            channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)

            by_name = {pbn.name: pbn for pbn in self.iter_bones(chains)}
            frames = list(range(self.sf, self.ef + 1))
            made = []
            for (bone, data_path, idx), series in sorted(curves.items()):
                fc = channelbag.fcurves.new(data_path, index=idx, group_name=bone)
                for f in frames:
                    if f in series:
                        fc.keyframe_points.insert(f, series[f], options={'FAST'})
                fc.update()
                made.append((fc, by_name[bone], data_path.rsplit(".", 1)[1]))

            self.reduce_keys(chains, curves=made)
            if cycle:
                self.close_cycle_seam(chains, curves=made)
                self.smooth_cycle_seam(chains, curves=made)

            track = adt.nla_tracks.new()
            undo.append(lambda: adt.nla_tracks.remove(track))
            track.name = LAYER_NAME
            strip = track.strips.new(LAYER_NAME, int(self.sf), action)
            try:
                if hasattr(strip, "action_slot"):
                    strip.action_slot = slot
            except (AttributeError, RuntimeError, TypeError):
                pass
            strip.blend_type = 'ADD'
            strip.extrapolation = 'NOTHING'
            bpy.context.view_layer.update()

            self._verify_layer(chains, results)
        except Exception:
            for fn in reversed(undo):
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
            raise
        return {"track": track, "strip": strip, "action": action}


# ============================================================================
# UI properties
# ============================================================================

class PERFECTOVERLAP_PG_props(bpy.types.PropertyGroup):
    start_frame: bpy.props.IntProperty(
        name="Start", default=0, min=-1048574, max=1048574,
    )
    end_frame: bpy.props.IntProperty(
        name="End", default=100, min=-1048574, max=1048574,
    )
    delay: bpy.props.FloatProperty(
        name="Delay", default=3.0, min=1.0, max=10.0,
        description="Lag time. Higher values make the chain trail further",
    )
    recursion: bpy.props.FloatProperty(
        name="Recursion", default=5.0, min=0.0, max=10.0,
        description="Momentum. 0 settles cleanly, 10 keeps swinging",
    )
    strength: bpy.props.FloatProperty(
        name="Strength", default=1.0, min=1.0, max=10.0,
        description="Adds follow-through (faster response, more swing)",
    )
    threshold: bpy.props.FloatProperty(
        name="Threshold", default=0.001, min=0.00001, max=0.1,
        step=0.01, precision=4,
        description="Key reduction tolerance",
    )
    animate_translate: bpy.props.BoolProperty(
        name="Translation", default=False,
        description=(
            "Also lag the location of unlocked, unconnected bones. "
            "Delete Keys then removes their location keys as well"
        ),
    )
    cycle: bpy.props.BoolProperty(
        name="Cycle", default=False,
        description=(
            "Treat the range as a loop (last frame = first frame): "
            "settle with pre-roll and close the seam"
        ),
    )
    cycle_preroll: bpy.props.IntProperty(
        name="Pre-roll", default=2, min=1, max=20,
        description="Number of hidden passes used to settle cycle momentum",
    )
    use_anim_layer: bpy.props.BoolProperty(
        name="Anim Layer", default=False,
        description=(
            "Bake the overlap onto a separate NLA track (Add blending) instead "
            "of writing into the existing Action. The active Action is pushed "
            "down into the NLA first so the track can sit on top of it. "
            "Re-baking replaces the previous layer; Delete Keys removes it"
        ),
    )
    debug: bpy.props.BoolProperty(
        name="Debug", default=False,
        description="Print solver details to the system console",
    )


# ============================================================================
# Operator helpers
# ============================================================================

def _blender_ok():
    return bpy.app.version >= REQUIRED_BLENDER


def _armature_ok(context):
    obj = context.active_object
    return (
        obj is not None
        and obj.type == 'ARMATURE'
        and context.mode == 'POSE'
    )


def _action_editable(obj):
    try:
        if not obj.is_editable:
            return False, "Armature is read-only"
    except AttributeError:
        if obj.library is not None and obj.override_library is None:
            return False, "Armature is linked and read-only"

    adt = obj.animation_data
    action = adt.action if adt else None
    if action is not None:
        try:
            if not action.is_editable:
                return False, "Action is read-only"
        except AttributeError:
            if action.library is not None and action.override_library is None:
                return False, "Action is linked and read-only"
    return True, ""


def _apply_props(solver, props):
    solver.configure(
        sf=props.start_frame,
        ef=props.end_frame,
        delay=props.delay,
        recursion=props.recursion,
        strength=props.strength,
        threshold=props.threshold,
        animate_translate=props.animate_translate,
        debug=props.debug,
    )
    return solver


def _short_list(names, limit=6):
    names = list(names)
    text = ", ".join(names[:limit])
    if len(names) > limit:
        text += " (+{} more)".format(len(names) - limit)
    return text


def _format_failures(label, failures, limit=8):
    if not failures:
        return ""
    vals = sorted(str(v) for v in failures)
    if len(vals) > limit:
        return "{}: {} (+{} more)".format(
            label, ", ".join(vals[:limit]), len(vals) - limit,
        )
    return "{}: {}".format(label, ", ".join(vals))


def _report_failures(operator, solver):
    for label, failures in (
        ("Keying", solver.key_failures),
        ("Channels", solver.lookup_failures),
        ("Pose", solver.pose_failures),
    ):
        msg = _format_failures(label, failures)
        if msg:
            operator.report({'WARNING'}, msg)


def _restore_scene(scene, frame, subframe):
    try:
        scene.frame_set(frame, subframe=subframe)
        bpy.context.view_layer.update()
    except Exception:
        traceback.print_exc()


# ============================================================================
# Operators
# ============================================================================

class PERFECTOVERLAP_OT_calculate(bpy.types.Operator):
    bl_idname = "perfect_overlap.calculate"
    bl_label = "Calculate"
    bl_description = "Bake overlap / follow-through on the selected chain"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        if not _blender_ok():
            self.report(
                {'ERROR'},
                "Blender {}.{}.{} or newer required".format(*REQUIRED_BLENDER),
            )
            return {'CANCELLED'}
        if not _armature_ok(context):
            self.report({'ERROR'}, "Must be in Pose Mode on an armature")
            return {'CANCELLED'}

        obj = context.active_object
        ok, why = _action_editable(obj)
        if not ok:
            self.report({'ERROR'}, why)
            return {'CANCELLED'}

        props = context.scene.perfect_overlap_props
        if props.start_frame >= props.end_frame:
            self.report({'ERROR'}, "Start frame must be less than end frame")
            return {'CANCELLED'}

        use_layer = bool(props.use_anim_layer)
        scene = context.scene
        original_frame = scene.frame_current
        original_subframe = current_subframe(scene)
        solver = _apply_props(PerfectOverlapSolver(), props)

        chains = None
        snapshot = None
        touched = False
        muted = []
        nla_warn = False
        result = {'FINISHED'}

        try:
            chains = solver.build_chains()
            if not chains:
                self.report(
                    {'WARNING'},
                    "No chain detected. Select the child controls; each "
                    "chain needs a parent that drives it.",
                )
                return {'CANCELLED'}

            snapshot = solver.capture_pose(chains)

            if use_layer:
                solver.check_layer_ready(obj)
                # earlier layer of these bones must not be part of the source
                muted = solver.layer_mute_previous(chains)

            # ---- Phase 1: read (nothing is modified) ----
            source = solver.sample_source(chains)
            if not source:
                self.report({'ERROR'}, "Could not read the source animation")
                return {'CANCELLED'}
            solver.model_mismatch = solver.check_model(source)

            # ---- Phase 2: solve (pure maths, nothing is modified) ----
            results = solver.solve(
                chains,
                source,
                cycle=bool(props.cycle),
                preroll=int(props.cycle_preroll) if props.cycle else 0,
            )
            if not results:
                self.report({'ERROR'}, "Solver produced no output")
                return {'CANCELLED'}

            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = solver.capture_pose(chains)

            # ---- Phase 3: write ----
            if use_layer:
                # transactional: leaves nothing behind if it raises
                layer = solver.write_layer(
                    chains, source, results, start_pose,
                    cycle=bool(props.cycle),
                )
                solver.layer_commit_previous(muted)
                muted = []
                solver.rename_layer_track(layer["track"])
            else:
                nla_warn = solver.nla_animates_chain(obj, chains)
                touched = True
                solver.delete_keys(chains)
                solver.restore_pose(chains, start_pose)
                solver.write_results(chains, results, start_pose)

                try:
                    solver.filled_keys = solver.fill_missing_keys(chains)
                except Exception as exc:
                    traceback.print_exc()
                    self.report({'WARNING'}, "Key check skipped: {}".format(exc))

                # ---- Post-processing (best effort) ----
                try:
                    solver.reduce_keys(chains)
                except Exception as exc:
                    traceback.print_exc()
                    self.report({'WARNING'}, "Key reduction skipped: {}".format(exc))

                try:
                    if props.cycle:
                        solver.close_cycle_seam(chains)
                        solver.smooth_cycle_seam(chains)
                        solver.add_cycle_modifiers(chains)
                    else:
                        solver.remove_cycle_modifiers(chains)
                except Exception as exc:
                    traceback.print_exc()
                    self.report({'WARNING'}, "Cycle step skipped: {}".format(exc))

            bone_count = sum(1 for _ in solver.iter_bones(chains))
            self.report(
                {'INFO'},
                "Baked overlap on {} bone(s){}, frames {} to {} "
                "(largest offset {:.1f} deg)".format(
                    bone_count,
                    " onto the '{}' NLA layer (Add)".format(LAYER_NAME) if use_layer else "",
                    props.start_frame, props.end_frame,
                    math.degrees(solver.max_overlap_angle),
                ),
            )

            if solver.max_overlap_angle < MIN_VISIBLE_OVERLAP:
                self.report(
                    {'WARNING'},
                    "No visible overlap: the selected bones and the parent "
                    "driving them do not move in this frame range. Animate "
                    "the parent (or pick another range) and run again.",
                )
            if props.animate_translate and solver.translate_skipped:
                self.report(
                    {'WARNING'},
                    "Translation skipped on {} connected/locked control(s)".format(
                        len(solver.translate_skipped),
                    ),
                )
            if solver.model_mismatch:
                self.report(
                    {'WARNING'},
                    "Constraints or Inherit settings on {} are not modelled "
                    "exactly; the overlap there may be slightly off.".format(
                        _short_list(solver.model_mismatch),
                    ),
                )
            if use_layer and solver.layer_static:
                self.report(
                    {'WARNING'},
                    "{} have a pose that is not the rest pose and no keys "
                    "below the layer; outside the baked range they return to "
                    "the rest pose.".format(_short_list(solver.layer_static)),
                )
            if not use_layer and solver.filled_keys:
                self.report(
                    {'WARNING'},
                    "Blender skipped {} baked key(s); they were re-inserted.".format(
                        solver.filled_keys,
                    ),
                )
            if not use_layer and nla_warn:
                self.report(
                    {'WARNING'},
                    "These bones are animated through the NLA. Baking "
                    "directly writes an override Action on top of it, which "
                    "can look like keys jumping or missing outside the range. "
                    "Tick Anim Layer to bake onto a separate NLA track instead.",
                )
            _report_failures(self, solver)

        except Exception as exc:
            traceback.print_exc()
            if use_layer and muted:
                solver.layer_unmute(muted)
            if touched:
                # The animation was already partly rewritten. Return FINISHED
                # so Blender pushes ONE undo step for it.
                self.report(
                    {'ERROR'},
                    "Partial bake ({}). Press Ctrl+Z to undo.".format(exc),
                )
                result = {'FINISHED'}
            else:
                self.report(
                    {'ERROR'},
                    "Bake failed, animation unchanged: {}".format(exc),
                )
                result = {'CANCELLED'}
            try:
                if snapshot and chains:
                    solver.restore_pose(chains, snapshot)
            except Exception:
                pass

        finally:
            _restore_scene(scene, original_frame, original_subframe)

        return result


class PERFECTOVERLAP_OT_delete(bpy.types.Operator):
    bl_idname = "perfect_overlap.delete"
    bl_label = "Delete Keys"
    bl_description = (
        "Delete the rotation keys (and location keys when Translation is on) "
        "of the selected chain in the frame range. With Anim Layer on, the "
        "selected bones are removed from the Perfect Overlap NLA layer instead"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        if not _blender_ok():
            self.report(
                {'ERROR'},
                "Blender {}.{}.{} or newer required".format(*REQUIRED_BLENDER),
            )
            return {'CANCELLED'}
        if not _armature_ok(context):
            self.report({'ERROR'}, "Must be in Pose Mode on an armature")
            return {'CANCELLED'}

        obj = context.active_object
        ok, why = _action_editable(obj)
        if not ok:
            self.report({'ERROR'}, why)
            return {'CANCELLED'}

        props = context.scene.perfect_overlap_props
        if props.start_frame >= props.end_frame:
            self.report({'ERROR'}, "Start frame must be less than end frame")
            return {'CANCELLED'}

        scene = context.scene
        original_frame = scene.frame_current
        original_subframe = current_subframe(scene)
        solver = _apply_props(PerfectOverlapSolver(), props)

        chains = None
        snapshot = None
        touched = False
        result = {'FINISHED'}

        try:
            chains = solver.build_chains()
            if not chains:
                self.report({'WARNING'}, "No chain detected")
                return {'CANCELLED'}

            if props.use_anim_layer:
                removed = solver.layer_remove(chains)
                self.report(
                    {'INFO'},
                    "Removed {} curve(s) from the '{}' layer".format(
                        removed, LAYER_NAME,
                    ),
                )
                return result

            snapshot = solver.capture_pose(chains)
            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = solver.capture_pose(chains)

            touched = True
            removed = solver.delete_keys(chains)
            solver.restore_pose(chains, start_pose)
            try:
                solver.remove_cycle_modifiers(chains)
            except Exception:
                traceback.print_exc()

            self.report(
                {'INFO'},
                "Deleted {} key(s) on {} bone(s)".format(
                    removed, sum(1 for _ in solver.iter_bones(chains)),
                ),
            )
            _report_failures(self, solver)

        except Exception as exc:
            traceback.print_exc()
            if touched:
                self.report(
                    {'ERROR'},
                    "Partial delete ({}). Press Ctrl+Z to undo.".format(exc),
                )
                result = {'FINISHED'}
            else:
                self.report(
                    {'ERROR'},
                    "Delete failed, animation unchanged: {}".format(exc),
                )
                result = {'CANCELLED'}
            try:
                if snapshot and chains:
                    solver.restore_pose(chains, snapshot)
            except Exception:
                pass

        finally:
            _restore_scene(scene, original_frame, original_subframe)

        return result


class PERFECTOVERLAP_OT_reset(bpy.types.Operator):
    bl_idname = "perfect_overlap.reset_settings"
    bl_label = "Reset Settings"
    bl_description = "Reset all Perfect Overlap values to defaults"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.perfect_overlap_props
        for key in props.bl_rna.properties.keys():
            if key in {"rna_type", "name"}:
                continue
            try:
                props.property_unset(key)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self.report({'INFO'}, "Settings reset to defaults")
        return {'FINISHED'}


# ============================================================================
# Panel
# ============================================================================

class PERFECTOVERLAP_PT_panel(bpy.types.Panel):
    bl_label = "Perfect Overlap"
    bl_idname = "PERFECTOVERLAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Perfect Overlap"
    bl_context = "posemode"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.perfect_overlap_props

        layout.label(text="Select the chain bones (not the driver)")
        layout.separator()

        box = layout.box()
        box.label(text="Frame Range")
        row = box.row(align=True)
        row.prop(props, "start_frame")
        row.prop(props, "end_frame")

        box = layout.box()
        box.label(text="Solver")
        row = box.row(align=True)
        row.prop(props, "delay")
        row.prop(props, "recursion")
        row.prop(props, "strength")
        box.prop(props, "threshold")

        box = layout.box()
        box.label(text="Options")
        box.prop(props, "animate_translate")
        box.prop(props, "cycle")
        if props.cycle:
            box.prop(props, "cycle_preroll")
        box.prop(props, "use_anim_layer")

        box = layout.box()
        box.prop(props, "debug")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.6
        row.operator("perfect_overlap.calculate", icon="KEYTYPE_KEYFRAME_VEC")
        row = layout.row()
        row.operator("perfect_overlap.delete", icon="KEYFRAME")
        row = layout.row()
        row.operator("perfect_overlap.reset_settings", icon="LOOP_BACK")


# ============================================================================
# Registration
# ============================================================================

classes = (
    PERFECTOVERLAP_PG_props,
    PERFECTOVERLAP_OT_calculate,
    PERFECTOVERLAP_OT_delete,
    PERFECTOVERLAP_OT_reset,
    PERFECTOVERLAP_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.perfect_overlap_props = bpy.props.PointerProperty(
        type=PERFECTOVERLAP_PG_props,
    )


def unregister():
    if hasattr(bpy.types.Scene, "perfect_overlap_props"):
        del bpy.types.Scene.perfect_overlap_props
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
