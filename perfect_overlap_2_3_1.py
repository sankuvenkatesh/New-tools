# ##### BEGIN GPL LICENSE BLOCK #####
#
# Perfect Overlap Addon
# Version 2.5.1 - tail-spring overlap solver (reviewed and fixed).
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
# 2.5.1 - fixes over 2.5.0
# ------------------------
# * Key reduction is now Douglas-Peucker. The old greedy pass only compared
#   the key next to the end of a segment, so the error could pile up.
# * Cycle: only the curves this add-on writes are touched (the seam code used
#   to overwrite unrelated channels such as scale), and the loop is solved as
#   a true periodic state, so first frame == last frame without a pop.
# * Armature OBJECT motion now also drives the lag (ignored in Cycle mode).
# * Translation spring is consistent with the aim solve and tracks the value
#   that was really applied, so locked axes cannot wind up.
# * Model check: warns about bones whose pose cannot be reproduced exactly
#   (constraints on the bone, non-default Inherit Rotation / Scale).
# * "Nothing to overlap" warning when neither the chain nor its driver moves.
# * AttributeError is no longer swallowed while sampling.

bl_info = {
    "name": "Perfect Overlap",
    "author": "Sanku Venkatesh",
    "version": (2, 5, 1),
    "blender": (4, 4, 0),
    "location": "3D Viewport > Sidebar > Perfect Overlap (Pose Mode)",
    "description": (
        "Overlap and follow-through animation for bone chains. "
        "Simulates a spring-lagged bone tip and re-aims the bone at it."
    ),
    "category": "Animation",
}

import bpy
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

    def reduce_keys(self, chains):
        """Key reduction on the channels this add-on wrote. Returns the number
        of keys removed."""
        groups = {}
        for fc, pbn, channel in self._baked_curves(chains):
            groups.setdefault((pbn.name, channel), []).append((fc, pbn))

        removed = 0
        for (name, channel), curves in groups.items():
            fcurves = [fc for fc, _ in curves]
            pbn = curves[0][1]
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
        return removed

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

    def close_cycle_seam(self, chains):
        """Make the end key equal the start key on every baked curve."""
        for fc, _pbn, _ch in self._baked_curves(chains):
            first, last = self._seam_keys(fc)
            if first is None or last is None or first is last:
                continue
            last.co[1] = first.co[1]
            try:
                fc.update()
            except Exception:
                pass

    def smooth_cycle_seam(self, chains):
        """Give the seam keys a shared tangent so the loop point is smooth."""
        for fc, _pbn, _ch in self._baked_curves(chains):
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
                self.report(
                    {'WARNING'},
                    "No chain detected. Select the child controls; each "
                    "chain needs a parent that drives it.",
                )
                return {'CANCELLED'}

            snapshot = solver.capture_pose(chains)

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
            touched = True
            solver.delete_keys(chains)
            solver.restore_pose(chains, start_pose)
            solver.write_results(chains, results, start_pose)

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
                "Baked overlap on {} bone(s), frames {} to {} "
                "(largest offset {:.1f} deg)".format(
                    bone_count, props.start_frame, props.end_frame,
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
            _report_failures(self, solver)

        except Exception as exc:
            traceback.print_exc()
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
        "of the selected chain in the frame range"
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
