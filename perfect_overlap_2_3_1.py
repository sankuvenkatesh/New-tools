# ##### BEGIN GPL LICENSE BLOCK #####
#
# Perfect Overlap Addon
# Version 2.3.1 - source-cache + local-space overlap solver (stability fix).
# Copyright 2021-2026 CaptainHansode, sakaiden.com
# Copyright 2026 Sanku Venkatesh
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version
# 2 of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####
#
# 2.3.1 changes
# -------------
# * FIX: safe_quaternion() used ``Quaternion.length``. mathutils.Quaternion has
#   no ``length`` (only Vector does); the size of a quaternion is
#   ``.magnitude``. That raised AttributeError, which escaped as
#   "RuntimeError: Perfect Overlap calculation failed". The same bug was also
#   being swallowed inside bone_rotation_quaternion(), which silently turned
#   every sampled rotation into the identity quaternion.
# * SAFER BAKE: the solver now works in three phases.
#     1. sample_source()  - read-only
#     2. solve_all()      - pure maths on the cached samples
#     3. del_animkey() + write_results() - the only phase that edits keys
#   If anything fails in phases 1-2 your animation is left untouched.
# * Operators no longer re-raise RuntimeError; failures are reported in the UI
#   and the full traceback is printed to the system console.
# * Post-processing (key reduction, cycle seam) can no longer fail the whole
#   bake - problems there are reported as warnings.
# * Cycle residual uses a proper quaternion angle (handles q == -q).
# * Removed the forced wm.redraw_timer call (console spam, not needed).

bl_info = {
    "name": "Perfect Overlap Addon",
    "author": "Sanku Venkatesh",
    "version": (2, 3, 1),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar > Perfect Overlap Addon (Pose Mode)",
    "description": (
        "Stable local-space overlap and follow-through for selected bone "
        "controls, with optional local translation and cycle support"
    ),
    "doc_url": "https://sakaiden.com",
    "category": "Animation",
}

import json
import math
import traceback

import bpy
import mathutils

__author__ = "Sanku Venkatesh"
__copyright__ = (
    "Copyright 2021-2026 CaptainHansode, sakaiden.com; "
    "2026 Sanku Venkatesh"
)
__credits__ = ["Sanku Venkatesh", "CaptainHansode (original author)"]
__license__ = "GPL"
__maintainer__ = "Sanku Venkatesh"
__status__ = "Production"

REQUIRED_BLENDER = (5, 1, 0)

NON_CONTROL_PREFIXES = (
    "def-", "def_", "def.",
    "org-", "org_", "org.",
    "mch-", "mch_", "mch.",
    "wgt-", "wgt_", "wgt.",
    "vis-", "vis_", "vis.",
)

EPS = 1.0e-6
ZERO = mathutils.Vector((0.0, 0.0, 0.0))
MAX_ANGULAR_STEP = math.radians(35.0)
MAX_TRANSLATION_STEP_RATIO = 0.20
MAX_TRANSLATION_OFFSET_RATIO = 1.5
CYCLE_MODIFIER_NAME = "Perfect Overlap Cycle"
TRANSLATION_OWNERSHIP_KEY = "_perfect_overlap_translation_ownership_v3"
LEGACY_OWNERSHIP_KEY = "_perfect_overlap_translation_ownership_v2"
ROTATION_CHANNELS = (
    "rotation_quaternion",
    "rotation_euler",
    "rotation_axis_angle",
)


# ----------------------------------------------------------------------
# Small math helpers
#
# NOTE: mathutils.Quaternion has NO ``.length`` attribute. Use ``.magnitude``.
# (mathutils.Vector has both ``.length`` and ``.magnitude``.)
# ----------------------------------------------------------------------

def is_finite_vector(vec):
    try:
        return all(math.isfinite(float(v)) for v in vec)
    except (TypeError, ValueError, AttributeError):
        return False


def is_finite_quaternion(quat):
    try:
        return all(math.isfinite(float(v)) for v in quat)
    except (TypeError, ValueError, AttributeError):
        return False


def identity_quaternion():
    return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))


def safe_quaternion(value, fallback=None):
    """Return a normalized copy of ``value``.

    Falls back to a copy of ``fallback`` (identity by default) when the input
    is zero-length, non-finite or not a quaternion at all. Never raises.
    """
    if fallback is None:
        fallback = identity_quaternion()
    try:
        q = value.copy()
        magnitude = float(q.magnitude)
        if (
            not math.isfinite(magnitude)
            or magnitude < EPS
            or not is_finite_quaternion(q)
        ):
            return fallback.copy()
        q.normalize()
        return q
    except Exception:
        return fallback.copy()


def safe_vector(value, fallback=None):
    """Return a copy of ``value`` or of ``fallback`` (zero by default). Never raises."""
    if fallback is None:
        fallback = ZERO
    try:
        v = value.copy()
        if not is_finite_vector(v):
            return fallback.copy()
        return v
    except Exception:
        return fallback.copy()


def quaternion_angle(q0, q1):
    """Smallest rotation angle in radians between two rotations.

    q and -q are the same rotation, so the absolute dot product is used.
    """
    try:
        a = safe_quaternion(q0)
        b = safe_quaternion(q1)
        dot = min(1.0, abs(float(a.dot(b))))
        return 2.0 * math.acos(dot)
    except Exception:
        return math.pi


def frame_list(sf, ef):
    return list(range(int(sf), int(ef) + 1))


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------

class PerfectOverlapSolver:
    """Local-space overlap solver.

    Pipeline (driven by PERFECTOVERLAP_OT_calculate):

      1. sample_source()  Read the user's evaluated local animation.
      2. solve_all()      Pure maths on those samples. It never touches the
                          scene, the pose or any F-Curve.
      3. del_animkey() / write_results()
                          Only now is the animation changed.

    Rotation overlap edits only rotation properties. Optional translation
    overlap edits local ``location`` only. No pose-bone matrix is ever
    assigned, so world-space writes cannot disturb IK / constraints.
    """

    def __init__(self):
        self.delay = 3.0
        self.recursion = 5.0
        self.strength = 1.0
        self.threshold = 0.001
        self.sf = 0
        self.ef = 100
        self.debug = False
        self.animate_translate = False

        self.passes_run = 0
        self.translate_skipped = []
        self.key_failures = set()
        self.fcurve_lookup_failures = set()
        self.pose_failures = set()

    @staticmethod
    def _record_failure(collection, value):
        if value:
            collection.add(str(value))

    def log(self, message):
        if self.debug:
            print("[Perfect Overlap] {}".format(message))

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def solver_params(self):
        delay = min(max(float(self.delay), 1.0), 10.0)
        k = 1.0 / delay
        # Damping is deliberately conservative. Higher recursion means less
        # damping and therefore more visible follow-through.
        rec = min(max(float(self.recursion), 0.0), 10.0)
        damping = 0.76 - (rec / 10.0) * 0.34
        damping = min(max(damping, 0.38), 0.76)

        strength = min(max(float(self.strength), 1.0), 10.0)
        amp = 1.0 + (strength - 1.0) / 18.0
        return k, damping, amp

    # ------------------------------------------------------------------
    # Rig inspection / chain building
    # ------------------------------------------------------------------

    def _bone_visible(self, pbn):
        try:
            bone = pbn.bone
            if bone.hide:
                return False
        except (AttributeError, ReferenceError):
            return False

        collections = getattr(pbn.bone, "collections", None)
        if not collections:
            return True

        for collection in collections:
            visible = getattr(collection, "is_visible_effectively", None)
            if visible is None:
                visible = getattr(collection, "is_visible_with_ancestors", None)
            if visible is None:
                visible = getattr(collection, "is_visible", True)
            if visible:
                return True
        return False

    def is_control_bone(self, pbn):
        try:
            if pbn.name.lower().startswith(NON_CONTROL_PREFIXES):
                return False
        except (AttributeError, ReferenceError):
            return False
        return self._bone_visible(pbn)

    def control_children(self, pbn, depth=0):
        if depth >= 8:
            return []
        result = []
        try:
            children = pbn.children
        except (AttributeError, ReferenceError):
            return result
        for child in children:
            if self.is_control_bone(child):
                result.append(child)
            else:
                result.extend(self.control_children(child, depth + 1))
        return result

    def next_in_chain(self, pbn, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        for child in candidates:
            try:
                if child.bone.use_connect:
                    return child
            except (AttributeError, ReferenceError):
                pass

        tail = pbn.tail
        try:
            direction = (pbn.tail - pbn.head).normalized()
        except Exception:
            direction = mathutils.Vector((0.0, 1.0, 0.0))
        scale = max(float(pbn.length), EPS)

        best = None
        best_score = None
        for child in candidates:
            try:
                distance = (child.head - tail).length / scale
                child_vec = child.tail - child.head
                if child_vec.length < EPS:
                    child_dir = -direction
                else:
                    child_dir = child_vec.normalized()
                alignment = float(child_dir.dot(direction))
                score = distance - alignment
                if best_score is None or score < best_score:
                    best_score = score
                    best = child
            except Exception:
                continue
        return best

    @staticmethod
    def get_bone_depth(pbn):
        depth = 0
        current = pbn
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def get_selection(self):
        obj = bpy.context.active_object
        if obj is None or obj.type != 'ARMATURE':
            return []
        result = []
        for pbn in bpy.context.selected_pose_bones or []:
            try:
                if pbn.id_data == obj:
                    result.append(pbn)
            except (AttributeError, ReferenceError):
                pass
        return result

    def get_tree_list(self):
        selected = self.get_selection()
        if not selected:
            return {}

        selected_names = {p.name for p in selected}
        ordered = sorted(selected, key=self.get_bone_depth)
        used = set()
        trees = {}
        count = 0

        for pbn in ordered:
            if pbn.name in used:
                continue
            if pbn.parent is None:
                continue

            chain = [pbn]
            used.add(pbn.name)
            current = pbn

            while True:
                candidates = [
                    child
                    for child in self.control_children(current)
                    if child.name in selected_names and child.name not in used
                ]
                next_bone = self.next_in_chain(current, candidates)
                if next_bone is None:
                    break
                chain.append(next_bone)
                used.add(next_bone.name)
                current = next_bone

            depth = self.get_bone_depth(pbn)
            trees.setdefault(depth, {})["tree{}".format(count)] = {
                "obj_list": chain,
            }
            count += 1

        return trees

    @staticmethod
    def iter_chains(obj_trees):
        for depth in sorted(obj_trees.keys()):
            for chain in obj_trees[depth].values():
                yield chain

    def iter_bones(self, obj_trees):
        seen = set()
        for chain in self.iter_chains(obj_trees):
            for pbn in chain["obj_list"]:
                if pbn.name in seen:
                    continue
                seen.add(pbn.name)
                yield pbn

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def rotation_data_path(pbn):
        if pbn.rotation_mode == 'QUATERNION':
            return 'rotation_quaternion'
        if pbn.rotation_mode == 'AXIS_ANGLE':
            return 'rotation_axis_angle'
        return 'rotation_euler'

    @staticmethod
    def bone_rotation_quaternion(pbn):
        """Return the quaternion represented by the bone's ACTIVE rotation channel.

        Blender keeps rotation_quaternion, rotation_euler and
        rotation_axis_angle as separate RNA properties. Only the channel
        selected by rotation_mode is evaluated, so never read
        rotation_quaternion unconditionally for Euler/Axis-Angle bones.
        """
        mode = pbn.rotation_mode

        try:
            if mode == 'QUATERNION':
                return safe_quaternion(pbn.rotation_quaternion)

            if mode == 'AXIS_ANGLE':
                aa = pbn.rotation_axis_angle
                axis = mathutils.Vector((
                    float(aa[1]),
                    float(aa[2]),
                    float(aa[3]),
                ))
                angle = float(aa[0])
                if (
                    axis.length < EPS
                    or not is_finite_vector(axis)
                    or not math.isfinite(angle)
                ):
                    # Blender treats a zero axis as "no rotation".
                    return identity_quaternion()
                axis.normalize()
                return safe_quaternion(mathutils.Quaternion(axis, angle))

            # All Euler rotation modes (XYZ, ZYX, etc.) use the active
            # rotation_euler channel.
            return safe_quaternion(pbn.rotation_euler.to_quaternion())

        except (ValueError, ArithmeticError, TypeError):
            return identity_quaternion()

    @staticmethod
    def _make_tracker(seed):
        """Continuity tracker seeded from a capture_pose() entry.

        The tracker remembers the last value written to each rotation
        representation so Euler / Axis-Angle / Quaternion output never flips
        between equivalent-but-different values from one frame to the next.
        """
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

    def _apply_quaternion_rotation(self, pbn, quat, tracker):
        q = safe_quaternion(quat)
        mode = pbn.rotation_mode

        try:
            if mode == 'QUATERNION':
                previous = tracker.get("last_quat")
                if previous is not None and q.dot(previous) < 0.0:
                    q.negate()
                pbn.rotation_quaternion = q
                tracker["last_quat"] = q.copy()
                return

            if mode == 'AXIS_ANGLE':
                angle = float(q.angle)
                axis = q.axis.copy()
                previous = tracker.get("last_axis_angle")
                if previous is not None:
                    prev_angle = float(previous[0])
                    prev_axis = mathutils.Vector(previous[1:4])
                    if prev_axis.length < EPS:
                        prev_axis = axis.copy()
                    else:
                        prev_axis.normalize()
                    if axis.dot(prev_axis) < 0.0:
                        axis.negate()
                        angle = -angle
                    angle += math.tau * round((prev_angle - angle) / math.tau)
                pbn.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
                tracker["last_axis_angle"] = (angle, axis.x, axis.y, axis.z)
                return

            euler = q.to_euler(mode)
            previous = tracker.get("last_euler")
            if previous is not None:
                try:
                    euler.make_compatible(previous)
                except (ValueError, ArithmeticError):
                    pass
            pbn.rotation_euler = euler
            tracker["last_euler"] = euler.copy()

        except Exception as exc:
            self._record_failure(
                self.pose_failures,
                "{} rotation assignment: {}".format(pbn.name, exc),
            )

    # ------------------------------------------------------------------
    # Phase 1 - source sampling (read-only)
    # ------------------------------------------------------------------

    def sample_source(self, obj_trees):
        """Sample the user's evaluated local animation before any key is edited."""
        scene = bpy.context.scene
        original_frame = scene.frame_current
        original_subframe = _current_subframe(scene)
        cache = {}

        bones = list(self.iter_bones(obj_trees))
        if not bones:
            return cache

        try:
            for frame in frame_list(self.sf, self.ef):
                scene.frame_set(frame)
                bpy.context.view_layer.update()
                frame_data = {}
                for pbn in bones:
                    frame_data[pbn.name] = {
                        "location": safe_vector(pbn.location),
                        "quaternion": self.bone_rotation_quaternion(pbn),
                    }
                cache[frame] = frame_data
        finally:
            scene.frame_set(original_frame, subframe=original_subframe)
            bpy.context.view_layer.update()

        return cache

    # ------------------------------------------------------------------
    # Phase 2 - local-space spring solver (pure maths)
    # ------------------------------------------------------------------

    def translate_axes(self, pbn):
        if not self.animate_translate:
            return []
        try:
            if pbn.bone.use_connect:
                return []
            return [i for i in range(3) if not pbn.lock_location[i]]
        except (AttributeError, ReferenceError):
            return []

    def _solve_rotation(self, state, target, k, damping, amp):
        target = safe_quaternion(target, state["quaternion"])
        current = state["quaternion"]

        try:
            delta = current.rotation_difference(target)
            delta.normalize()
            if delta.w < 0.0:
                delta.negate()
            angle = float(delta.angle)
            axis = delta.axis
            if axis.length < EPS or angle < EPS:
                error = ZERO.copy()
            else:
                axis.normalize()
                error = axis * angle
        except (ValueError, ArithmeticError, TypeError):
            error = ZERO.copy()

        velocity = state["angular_velocity"] * damping + error * k
        if not is_finite_vector(velocity):
            velocity = ZERO.copy()

        step = velocity * amp
        if step.length > MAX_ANGULAR_STEP:
            step = step.normalized() * MAX_ANGULAR_STEP

        try:
            step_len = float(step.length)
            if step_len > EPS:
                step_axis = step.normalized()
                dq = mathutils.Quaternion(step_axis, step_len)
                result = current @ dq
                result.normalize()
            else:
                result = current.copy()
        except (ValueError, ArithmeticError, TypeError):
            result = current.copy()

        state["angular_velocity"] = velocity.copy()
        state["quaternion"] = result.copy()
        return result

    def _solve_location(self, state, target, info, k, damping, amp):
        allowed = info["axes"]
        target = safe_vector(target, state["location"])
        current = state["location"]

        # Keep all locked / disallowed axes exactly at the source value.
        for axis in range(3):
            if axis not in allowed:
                current[axis] = target[axis]
                state["location_velocity"][axis] = 0.0

        error = target - current
        velocity = state["location_velocity"] * damping + error * k
        if not is_finite_vector(velocity):
            velocity = ZERO.copy()

        length = info["length"]
        max_step = length * MAX_TRANSLATION_STEP_RATIO
        step = velocity * amp
        if step.length > max_step:
            step = step.normalized() * max_step

        for axis in range(3):
            if axis not in allowed:
                step[axis] = 0.0

        result = current + step

        # Prevent an accumulating positional runaway on control bones.
        maximum_offset = length * MAX_TRANSLATION_OFFSET_RATIO
        offset = result - target
        if offset.length > maximum_offset:
            result = target + offset.normalized() * maximum_offset
            velocity = ZERO.copy()

        state["location_velocity"] = velocity.copy()
        state["location"] = result.copy()
        return result

    def _step_frame(self, frame_data, states, info, k, damping, amp):
        """Advance every bone by one frame. Pure maths, no Blender writes."""
        out = {}
        for name, state in states.items():
            data = frame_data.get(name)
            if data is None:
                continue
            quat = self._solve_rotation(
                state, data["quaternion"], k, damping, amp
            )
            loc = None
            if info[name]["axes"]:
                loc = self._solve_location(
                    state, data["location"], info[name], k, damping, amp
                )
            out[name] = {
                "quaternion": quat.copy(),
                "location": loc.copy() if loc is not None else None,
            }
        return out

    def solve_all(self, obj_trees, source_cache, cycle=False, preroll=0):
        """Simulate the overlap for the whole frame range.

        Returns ``{frame: {bone_name: {"quaternion": Quaternion,
        "location": Vector or None}}}`` or ``None`` when there is nothing to
        solve. Touches no Blender data, so it cannot damage the animation.
        """
        frames = frame_list(self.sf, self.ef)
        first = source_cache.get(frames[0])
        if not first:
            return None

        k, damping, amp = self.solver_params()
        self.translate_skipped = []
        states = {}
        info = {}

        for pbn in self.iter_bones(obj_trees):
            data = first.get(pbn.name)
            if data is None:
                continue
            axes = self.translate_axes(pbn)
            if self.animate_translate and not axes:
                self.translate_skipped.append(pbn.name)
            info[pbn.name] = {
                "axes": axes,
                "length": max(float(pbn.length), 1.0e-5),
            }
            states[pbn.name] = {
                "quaternion": data["quaternion"].copy(),
                "location": data["location"].copy(),
                "angular_velocity": ZERO.copy(),
                "location_velocity": ZERO.copy(),
            }

        if not states:
            return None

        # The Start Frame remains the exact source pose. This avoids creating
        # a visible jump at the beginning of the selected range.
        results = {
            frames[0]: {
                name: {
                    "quaternion": first[name]["quaternion"].copy(),
                    "location": (
                        first[name]["location"].copy()
                        if info[name]["axes"] else None
                    ),
                }
                for name in states
            }
        }

        if cycle and preroll > 0:
            # Settle the momentum through repeated source loops. Nothing is
            # recorded during these hidden passes.
            for _ in range(int(preroll)):
                for frame in frames[1:]:
                    self._step_frame(
                        source_cache.get(frame, {}),
                        states, info, k, damping, amp,
                    )

            # Restart from the exact source pose at the Start Frame but keep
            # the settled velocities. The visible output begins at source.
            for name, state in states.items():
                state["quaternion"] = first[name]["quaternion"].copy()
                state["location"] = first[name]["location"].copy()

        for frame in frames[1:]:
            results[frame] = self._step_frame(
                source_cache.get(frame, {}),
                states, info, k, damping, amp,
            )

        self.passes_run = int(preroll) if cycle else 1
        self.log(
            "solved {} bone(s) over {} frame(s), pre-roll passes: {}".format(
                len(states), len(frames), self.passes_run
            )
        )
        return results

    # ------------------------------------------------------------------
    # Phase 3 - writing
    # ------------------------------------------------------------------

    def write_results(self, obj_trees, results, start_pose):
        """Write the solved rotation (and optional location) as keyframes."""
        scene = bpy.context.scene
        obj = bpy.context.active_object
        start_pose = start_pose or {}

        bones = {}
        for pbn in self.iter_bones(obj_trees):
            bones[pbn.name] = pbn
        trackers = {
            name: self._make_tracker(start_pose.get(name)) for name in bones
        }

        self._ensure_action_slot(obj)

        for frame in frame_list(self.sf, self.ef):
            frame_result = results.get(frame)
            if not frame_result:
                continue
            scene.frame_set(frame)
            for name, pbn in bones.items():
                res = frame_result.get(name)
                if res is None:
                    continue
                self._apply_quaternion_rotation(
                    pbn, res["quaternion"], trackers[name]
                )
                if res["location"] is not None:
                    try:
                        pbn.location = res["location"].copy()
                    except Exception as exc:
                        self._record_failure(
                            self.pose_failures,
                            "{} location assignment: {}".format(name, exc),
                        )
                self.set_animkey(pbn)
            bpy.context.view_layer.update()

        if not self._ensure_action_slot(obj):
            self._record_failure(self.key_failures, "Action slot")

    # ------------------------------------------------------------------
    # Keying / action slot handling
    # ------------------------------------------------------------------

    def _ensure_action_slot(self, id_data):
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
            from bpy_extras import anim_utils
            helper = getattr(anim_utils, "action_get_first_suitable_slot", None)
            if helper is not None:
                slot = helper(adt.action, id_data.id_type)
                if slot is not None:
                    adt.action_slot = slot
                    return True
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass

        try:
            adt.action_slot = adt.action.slots.new(
                id_type=id_data.id_type,
                name=id_data.name,
            )
            return True
        except (AttributeError, RuntimeError, TypeError):
            return False

    def set_animkey(self, pbn):
        frame = bpy.context.scene.frame_current

        for axis in self.translate_axes(pbn):
            try:
                result = pbn.keyframe_insert(
                    data_path='location', index=axis, frame=frame
                )
                if result is False:
                    self._record_failure(
                        self.key_failures,
                        "{} loc[{}]".format(pbn.name, axis),
                    )
            except Exception as exc:
                self._record_failure(
                    self.key_failures,
                    "{} loc[{}]: {}".format(pbn.name, axis, exc),
                )

        try:
            result = pbn.keyframe_insert(
                data_path=self.rotation_data_path(pbn),
                frame=frame,
            )
            if result is False:
                self._record_failure(
                    self.key_failures, "{} rotation".format(pbn.name)
                )
        except Exception as exc:
            self._record_failure(
                self.key_failures, "{} rotation: {}".format(pbn.name, exc)
            )

    # ------------------------------------------------------------------
    # Pose capture / restore
    # ------------------------------------------------------------------

    def capture_pose(self, obj_trees):
        snapshot = {}
        for pbn in self.iter_bones(obj_trees):
            snapshot[pbn.name] = {
                "location": pbn.location.copy(),
                "rotation_quaternion": pbn.rotation_quaternion.copy(),
                "rotation_euler": pbn.rotation_euler.copy(),
                "rotation_axis_angle": tuple(
                    float(v) for v in pbn.rotation_axis_angle
                ),
                "scale": pbn.scale.copy(),
            }
        return snapshot

    def restore_pose(self, obj_trees, snapshot):
        if not snapshot:
            return
        for pbn in self.iter_bones(obj_trees):
            values = snapshot.get(pbn.name)
            if values is None:
                continue
            pbn.location = values["location"].copy()
            pbn.rotation_quaternion = values["rotation_quaternion"].copy()
            e = values["rotation_euler"]
            pbn.rotation_euler = mathutils.Euler((e[0], e[1], e[2]), e.order)
            pbn.rotation_axis_angle = values["rotation_axis_angle"]
            pbn.scale = values["scale"].copy()
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------
    # Blender 5.1 slotted Action / F-Curve API
    # ------------------------------------------------------------------

    def _fcurve_container(self, id_data, report_missing=False):
        if id_data is None:
            return None
        try:
            adt = id_data.animation_data
        except AttributeError:
            return None
        if adt is None or adt.action is None:
            return None
        if not self._ensure_action_slot(id_data):
            if report_missing:
                self._record_failure(
                    self.fcurve_lookup_failures,
                    "{} Action slot".format(id_data.name),
                )
            return None

        slot = getattr(adt, "action_slot", None)
        if slot is None:
            if report_missing:
                self._record_failure(
                    self.fcurve_lookup_failures,
                    "{} assigned Action slot".format(id_data.name),
                )
            return None

        try:
            from bpy_extras import anim_utils
            helper = getattr(
                anim_utils, "animdata_get_channelbag_for_assigned_slot", None
            )
            channelbag = helper(adt) if helper is not None else None
            if channelbag is None:
                fallback = getattr(
                    anim_utils, "action_get_channelbag_for_slot", None
                )
                if fallback is not None:
                    channelbag = fallback(adt.action, slot)
            # No channelbag simply means this slot has no F-Curves yet
            # (for example a brand-new Action). That is not a failure.
            if channelbag is None:
                return None
            return channelbag.fcurves
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            if report_missing:
                self._record_failure(
                    self.fcurve_lookup_failures,
                    "{} channelbag lookup: {}".format(id_data.name, exc),
                )
            return None

    def _get_fcurves(self, id_data, report_missing=False):
        container = self._fcurve_container(id_data, report_missing)
        return list(container) if container is not None else []

    def _chain_fcurve_map(self, obj_trees, report_missing=True):
        obj = bpy.context.active_object
        if obj is None:
            return []
        by_prefix = {
            pbn.path_from_id(): pbn for pbn in self.iter_bones(obj_trees)
        }
        result = []
        for fc in self._get_fcurves(obj, report_missing):
            for prefix, pbn in by_prefix.items():
                if fc.data_path.startswith(prefix + "."):
                    channel = fc.data_path[len(prefix) + 1:]
                    result.append((fc, pbn, channel))
                    break
        return result

    def _baked_channels(self, obj_trees):
        groups = {}
        for fc, pbn, channel in self._chain_fcurve_map(obj_trees, True):
            if channel == self.rotation_data_path(pbn):
                pass
            elif (
                channel == 'location'
                and fc.array_index in self.translate_axes(pbn)
            ):
                pass
            else:
                continue
            groups.setdefault((pbn.name, channel), []).append((fc, pbn))
        return groups

    def _baked_fcurves(self, obj_trees):
        result = []
        for group in self._baked_channels(obj_trees).values():
            result.extend(group)
        return result

    # ------------------------------------------------------------------
    # Translation ownership
    # ------------------------------------------------------------------

    def _assigned_action_slot(self, id_data):
        if id_data is None:
            return None, None, None
        try:
            adt = id_data.animation_data
        except AttributeError:
            return None, None, None
        if adt is None or adt.action is None:
            return None, adt, None
        if not self._ensure_action_slot(id_data):
            return adt.action, adt, None
        return adt.action, adt, getattr(adt, "action_slot", None)

    @staticmethod
    def _ownership_entry_matches(entry, _id_data, slot):
        try:
            return int(entry.get("slot_handle", -999999)) == int(slot.handle)
        except (AttributeError, TypeError, ValueError):
            return False

    def _load_translation_ownership(self, id_data):
        action, _adt, _slot = self._assigned_action_slot(id_data)
        if action is None:
            return []
        raw = action.get(TRANSLATION_OWNERSHIP_KEY)
        if not raw:
            # Also read v2 metadata created by 2.2.x.
            raw = action.get(LEGACY_OWNERSHIP_KEY)
        if not raw:
            return []
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            self._record_failure(
                self.fcurve_lookup_failures,
                "{} translation ownership metadata".format(id_data.name),
            )
            return []
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
        return []

    def _save_translation_ownership(self, id_data, entries):
        action, _adt, _slot = self._assigned_action_slot(id_data)
        if action is None:
            return True
        try:
            action[TRANSLATION_OWNERSHIP_KEY] = json.dumps(
                entries, separators=(",", ":")
            )
            return True
        except Exception as exc:
            self._record_failure(
                self.fcurve_lookup_failures,
                "{} translation ownership save: {}".format(id_data.name, exc),
            )
            return False

    @staticmethod
    def _normalized_owned_frames(entry):
        frames = set()
        for frame in entry.get("frames", []):
            try:
                value = float(frame)
                if math.isfinite(value):
                    frames.add(round(value, 6))
            except (TypeError, ValueError):
                continue
        return sorted(frames)

    @staticmethod
    def _ownership_entry_with_frames(entry, frames):
        normalized = sorted({round(float(frame), 6) for frame in frames})
        if not normalized:
            return None
        result = dict(entry)
        result["version"] = 3
        result["frames"] = normalized
        result["start"] = min(normalized)
        result["end"] = max(normalized)
        return result

    def remove_owned_translation_animation(self, id_data, obj_trees=None):
        action, _adt, slot = self._assigned_action_slot(id_data)
        if action is None or slot is None:
            return 0

        entries = self._load_translation_ownership(id_data)
        if not entries:
            return 0

        selected_paths = None
        if obj_trees is not None:
            selected_paths = {
                pbn.path_from_id() + ".location"
                for pbn in self.iter_bones(obj_trees)
            }

        targets = []
        kept_entries = []
        for entry in entries:
            if not self._ownership_entry_matches(entry, id_data, slot):
                kept_entries.append(entry)
                continue
            if (
                selected_paths is not None
                and entry.get("data_path") not in selected_paths
            ):
                kept_entries.append(entry)
                continue
            targets.append(entry)

        if not targets:
            return 0

        container = self._fcurve_container(id_data, True)
        if container is None:
            return 0

        curves = {
            (fc.data_path, int(fc.array_index)): fc for fc in list(container)
        }
        sf = float(self.sf)
        ef = float(self.ef)
        remove_frames = {}
        retained = {}
        target_curve_keys = set()

        for entry in targets:
            key = (
                entry.get("data_path", ""),
                int(entry.get("array_index", 0)),
            )
            target_curve_keys.add(key)
            frames = self._normalized_owned_frames(entry)
            outside = [f for f in frames if f < sf or f > ef]
            inside = [f for f in frames if sf <= f <= ef]
            if outside:
                value = self._ownership_entry_with_frames(entry, outside)
                if value is not None:
                    prev = retained.get(key)
                    if prev is None:
                        retained[key] = value
                    else:
                        merged = sorted(set(prev["frames"]) | set(value["frames"]))
                        prev["frames"] = merged
                        prev["start"] = min(merged)
                        prev["end"] = max(merged)
            if inside:
                remove_frames.setdefault(key, set()).update(inside)

        removed = 0
        for key, frames in remove_frames.items():
            fc = curves.get(key)
            if fc is None:
                continue
            indices = [
                i for i, k in enumerate(fc.keyframe_points)
                if round(float(k.co[0]), 6) in frames
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

        kept_entries.extend(retained.values())
        for key in target_curve_keys:
            fc = curves.get(key)
            if fc is None:
                continue
            try:
                if len(fc.keyframe_points) == 0:
                    container.remove(fc)
            except (RuntimeError, ReferenceError):
                pass

        self._save_translation_ownership(id_data, kept_entries)
        bpy.context.view_layer.update()
        return removed

    def record_translation_ownership(self, obj_trees):
        obj = bpy.context.active_object
        if obj is None:
            return False
        action, _adt, slot = self._assigned_action_slot(obj)
        if action is None or slot is None:
            return True

        existing = self._load_translation_ownership(obj)
        selected_paths = {
            pbn.path_from_id() + ".location"
            for pbn in self.iter_bones(obj_trees)
        }
        sf = float(self.sf)
        ef = float(self.ef)

        kept = []
        preserved_by_key = {}
        for entry in existing:
            if (
                not self._ownership_entry_matches(entry, obj, slot)
                or entry.get("data_path") not in selected_paths
            ):
                kept.append(entry)
                continue
            frames = self._normalized_owned_frames(entry)
            outside = [f for f in frames if f < sf or f > ef]
            if not outside:
                continue
            key = (
                entry.get("data_path", ""),
                int(entry.get("array_index", 0)),
            )
            value = self._ownership_entry_with_frames(entry, outside)
            if value is not None:
                preserved_by_key[key] = value
        kept.extend(preserved_by_key.values())

        container = self._fcurve_container(obj, True)
        if container is None:
            return False

        owner_id = obj.name
        slot_handle = int(slot.handle)
        slot_identifier = getattr(slot, "identifier", "")
        selected_by_path = {
            pbn.path_from_id() + ".location": pbn
            for pbn in self.iter_bones(obj_trees)
        }

        for fc in list(container):
            pbn = selected_by_path.get(fc.data_path)
            if pbn is None or fc.array_index not in self.translate_axes(pbn):
                continue
            frames = [
                round(float(key.co[0]), 6)
                for key in fc.keyframe_points
                if sf <= float(key.co[0]) <= ef
            ]
            if not frames:
                continue
            frames = sorted(set(frames))
            kept.append({
                "version": 3,
                "owner_id": owner_id,
                "slot_handle": slot_handle,
                "slot_identifier": slot_identifier,
                "data_path": fc.data_path,
                "array_index": int(fc.array_index),
                "frames": frames,
                "start": min(frames),
                "end": max(frames),
            })

        return self._save_translation_ownership(obj, kept)

    # ------------------------------------------------------------------
    # Delete / cleanup
    # ------------------------------------------------------------------

    def del_animkey(self, obj_trees, pose_snapshot=None):
        """Delete rotation keys and selected translation keys in range.

        When Translation is enabled, all existing location keys for unlocked
        translation axes in the selected range are removed before the new
        bake is written. This is intentional: the bake owns that channel
        over the requested range, so manually keyed location values inside
        the range will be replaced by the generated result.

        Returns the number of keys removed.
        """
        obj = bpy.context.active_object
        container = self._fcurve_container(obj)
        emptied = []
        removed = 0

        for fc, pbn, channel in self._chain_fcurve_map(obj_trees):
            if channel == 'location':
                if fc.array_index not in self.translate_axes(pbn):
                    continue
            elif channel not in ROTATION_CHANNELS:
                continue

            indices = [
                i for i, key in enumerate(fc.keyframe_points)
                if self.sf <= float(key.co[0]) <= self.ef
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

        bpy.context.view_layer.update()
        self.restore_pose(obj_trees, pose_snapshot)
        self._ensure_action_slot(obj)
        self.remove_addon_cycles(obj_trees)
        return removed

    # ------------------------------------------------------------------
    # Key reduction
    # ------------------------------------------------------------------

    @staticmethod
    def _dp_max_error(values, frames, left, right):
        if right - left <= 1:
            return 0.0, None
        t0 = frames[left]
        t1 = frames[right]
        span = t1 - t0
        if abs(span) < EPS:
            return 0.0, None
        max_error = -1.0
        max_index = None
        for idx in range(left + 1, right):
            ratio = (frames[idx] - t0) / span
            err = 0.0
            for component in values:
                predicted = component[left] + (component[right] - component[left]) * ratio
                err = max(err, abs(component[idx] - predicted))
            if err > max_error:
                max_error = err
                max_index = idx
        return max_error, max_index

    def _douglas_peucker_keep_indices(self, values, frames, tolerance):
        count = len(frames)
        if count <= 2:
            return {0, max(0, count - 1)}
        keep = {0, count - 1}
        stack = [(0, count - 1)]
        while stack:
            left, right = stack.pop()
            error, idx = self._dp_max_error(values, frames, left, right)
            if idx is None or error <= tolerance:
                continue
            keep.add(idx)
            stack.append((left, idx))
            stack.append((idx, right))
        return keep

    def _cleanup_tolerance_for_group(self, channel, pbn):
        if channel == 'location':
            return max(float(self.threshold) * max(float(pbn.length), EPS), 1.0e-8)
        return max(float(self.threshold), 1.0e-6)

    def cleanup_keys(self, obj_trees):
        removed = 0
        for (bone_name, channel), curves in self._baked_channels(obj_trees).items():
            if not curves:
                continue
            fcurves = [fc for fc, _ in curves]
            reference_bone = curves[0][1]
            frames = None
            for fc in fcurves:
                channel_frames = [
                    float(k.co[0])
                    for k in fc.keyframe_points
                    if self.sf <= float(k.co[0]) <= self.ef
                ]
                if frames is None:
                    frames = channel_frames
                else:
                    common = set(channel_frames)
                    frames = [f for f in frames if f in common]
            if not frames or len(frames) <= 2:
                self._smooth(fcurves)
                continue

            values = []
            valid = True
            for fc in fcurves:
                lookup = {float(k.co[0]): float(k.co[1]) for k in fc.keyframe_points}
                try:
                    values.append([lookup[f] for f in frames])
                except KeyError:
                    valid = False
                    break
            if not valid:
                self._record_failure(
                    self.fcurve_lookup_failures,
                    "{} {} incomplete key set".format(bone_name, channel),
                )
                self._smooth(fcurves)
                continue

            tolerance = self._cleanup_tolerance_for_group(channel, reference_bone)
            keep = self._douglas_peucker_keep_indices(values, frames, tolerance)
            remove_frames = {frames[i] for i in range(len(frames)) if i not in keep}
            for fc in fcurves:
                indices = [
                    i for i, k in enumerate(fc.keyframe_points)
                    if float(k.co[0]) in remove_frames
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
            self._smooth(fcurves)
        return removed

    def _smooth(self, fcurves):
        for fc in fcurves:
            for key in fc.keyframe_points:
                frame = float(key.co[0])
                if self.sf <= frame <= self.ef:
                    key.interpolation = 'BEZIER'
                    key.handle_left_type = 'AUTO_CLAMPED'
                    key.handle_right_type = 'AUTO_CLAMPED'
            try:
                fc.update()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Cycles
    # ------------------------------------------------------------------

    @staticmethod
    def _seam_keys(fcurve, sf, ef):
        first = None
        last = None
        for key in fcurve.keyframe_points:
            frame = float(key.co[0])
            if math.isclose(frame, float(sf), abs_tol=EPS):
                first = key
            if math.isclose(frame, float(ef), abs_tol=EPS):
                last = key
        return first, last

    def remove_addon_cycles(self, obj_trees):
        """Remove this add-on's Cycles modifiers from every rotation/location F-Curve of the chain."""
        for fc, _pbn, channel in self._chain_fcurve_map(obj_trees):
            if channel != 'location' and channel not in ROTATION_CHANNELS:
                continue
            for modifier in list(fc.modifiers):
                if (
                    modifier.type == 'CYCLES'
                    and getattr(modifier, "name", None) == CYCLE_MODIFIER_NAME
                ):
                    try:
                        fc.modifiers.remove(modifier)
                    except (RuntimeError, ReferenceError):
                        pass

    def add_cyclic_modifiers(self, obj_trees):
        for fc, _ in self._baked_fcurves(obj_trees):
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

    def enforce_cycle_seam(self, obj_trees):
        for fc, _ in self._baked_fcurves(obj_trees):
            first, last = self._seam_keys(fc, self.sf, self.ef)
            if first is None or last is None or first is last:
                continue
            last.co[1] = first.co[1]
            fc.update()

    def set_cycle_seam_tangents(self, obj_trees):
        for fc, _ in self._baked_fcurves(obj_trees):
            first, last = self._seam_keys(fc, self.sf, self.ef)
            if first is None or last is None or first is last:
                continue
            keys = sorted(fc.keyframe_points, key=lambda k: float(k.co[0]))
            inner = [k for k in keys if self.sf < float(k.co[0]) < self.ef]
            if len(inner) < 2:
                continue
            after = inner[0]
            before = inner[-1]
            dt_after = float(after.co[0]) - float(self.sf)
            dt_before = float(self.ef) - float(before.co[0])
            if dt_after < EPS or dt_before < EPS:
                continue
            slope = (float(after.co[1]) - float(before.co[1])) / (dt_after + dt_before)
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
            fc.update()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def cycle_residual(self, obj_trees):
        """Return (bone_name, angle_radians) of the worst start/end rotation mismatch."""
        scene = bpy.context.scene
        original_frame = scene.frame_current
        original_subframe = _current_subframe(scene)
        try:
            scene.frame_set(self.sf)
            bpy.context.view_layer.update()
            start = {
                p.name: self.bone_rotation_quaternion(p)
                for p in self.iter_bones(obj_trees)
            }
            scene.frame_set(self.ef)
            bpy.context.view_layer.update()
            end = {
                p.name: self.bone_rotation_quaternion(p)
                for p in self.iter_bones(obj_trees)
            }
        finally:
            scene.frame_set(original_frame, subframe=original_subframe)
            bpy.context.view_layer.update()

        worst_name = ""
        worst_angle = 0.0
        for name, q0 in start.items():
            q1 = end.get(name)
            if q1 is None:
                continue
            angle = quaternion_angle(q0, q1)
            if angle > worst_angle:
                worst_angle = angle
                worst_name = name
        return worst_name, worst_angle


# ----------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------

class PERFECTOVERLAP_PG_props(bpy.types.PropertyGroup):
    start_frame: bpy.props.IntProperty(name="Start Frame", default=0, min=-1048574, max=1048574)
    end_frame: bpy.props.IntProperty(name="End Frame", default=100, min=-1048574, max=1048574)

    delay: bpy.props.FloatProperty(
        name="Delay", default=3.0, min=1.0, max=10.0,
        description="How slowly the overlapped rotation catches the source motion.",
    )
    recursion: bpy.props.FloatProperty(
        name="Recursion", default=5.0, min=0.0, max=10.0,
        description="Controls damping/momentum in the overlap solver.",
    )
    strength: bpy.props.FloatProperty(
        name="Strength", default=1.0, min=1.0, max=10.0,
        description="Adds controlled follow-through. 1 is the safest/default value.",
    )
    threshold: bpy.props.FloatProperty(
        name="Threshold", default=0.001, min=0.00001, max=0.1,
        step=0.01, precision=4,
        description="Key reduction tolerance. Location values use bone length as scale.",
    )
    debug: bpy.props.BoolProperty(
        name="Debug", default=False,
        description="Print solver details to the system console.",
    )
    animate_translate: bpy.props.BoolProperty(
        name="Translation", default=False,
        description=(
            "OFF: rotation only, with the original local translation preserved. "
            "ON: also overlap unlocked local location channels."
        ),
    )
    cycle: bpy.props.BoolProperty(
        name="Cycle", default=False,
        description="Solve as a loop and close the generated seam.",
    )
    cycle_preroll: bpy.props.IntProperty(
        name="Pre-roll Passes", default=2, min=1, max=20,
        description="Number of hidden source loops used to settle momentum before baking.",
    )


# ----------------------------------------------------------------------
# Operator helpers
# ----------------------------------------------------------------------

def _configure(module, props):
    module.sf = int(props.start_frame)
    module.ef = int(props.end_frame)
    module.debug = bool(props.debug)
    module.delay = float(props.delay)
    module.recursion = float(props.recursion)
    module.strength = float(props.strength)
    module.threshold = float(props.threshold)
    module.animate_translate = bool(props.animate_translate)
    return module


def _current_subframe(scene):
    try:
        return float(scene.frame_subframe)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _restore_scene_state(scene, frame, subframe, module=None, obj_trees=None, snapshot=None):
    """Put the timeline back where the user left it (and the pose, after a failed bake)."""
    try:
        scene.frame_set(frame, subframe=subframe)
        bpy.context.view_layer.update()
        if module is not None and obj_trees and snapshot:
            module.restore_pose(obj_trees, snapshot)
            scene.frame_set(frame, subframe=subframe)
            bpy.context.view_layer.update()
    except Exception:
        traceback.print_exc()


def _safe_step(operator, label, func, *args):
    """Run a non-essential post-processing step; report a warning instead of failing."""
    try:
        return func(*args)
    except Exception as exc:
        traceback.print_exc()
        operator.report({'WARNING'}, "{} skipped: {}".format(label, exc))
        return None


def _frame_range_ok(operator, props):
    if props.start_frame < props.end_frame:
        return True
    message = "Make the Start Frame smaller than the End Frame."
    try:
        bpy.context.window_manager.popup_menu(
            lambda self, context: self.layout.label(text=message),
            title="Info", icon="INFO",
        )
    except Exception:
        pass
    operator.report({'INFO'}, message)
    return False


def _blender_version_ok(operator):
    if bpy.app.version >= REQUIRED_BLENDER:
        return True
    message = "Perfect Overlap requires Blender {}.{}.{} or newer.".format(*REQUIRED_BLENDER)
    operator.report({'ERROR'}, message)
    return False


def _active_armature_ok(operator, context):
    obj = context.active_object
    if obj is None or obj.type != 'ARMATURE':
        operator.report({'ERROR'}, "Active object must be an armature.")
        return False
    if context.mode != 'POSE':
        operator.report({'ERROR'}, "Perfect Overlap must be run in Pose Mode.")
        return False
    return True


def _action_editable_ok(operator, obj):
    try:
        if not obj.is_editable:
            operator.report({'ERROR'}, "The active armature is read-only. Make it local or create an editable library override.")
            return False
    except AttributeError:
        if obj.library is not None and obj.override_library is None:
            operator.report({'ERROR'}, "The active armature is linked and read-only.")
            return False

    adt = obj.animation_data
    action = adt.action if adt else None
    if action is not None:
        try:
            if not action.is_editable:
                operator.report({'ERROR'}, "The active Action is read-only. Assign a local editable Action before baking.")
                return False
        except AttributeError:
            if action.library is not None and action.override_library is None:
                operator.report({'ERROR'}, "The active Action is linked and read-only.")
                return False
    return True


def _format_failures(label, failures, limit=12):
    if not failures:
        return ""
    values = sorted(str(v) for v in failures)
    if len(values) > limit:
        return "{}: {} (+{} more)".format(label, ", ".join(values[:limit]), len(values) - limit)
    return "{}: {}".format(label, ", ".join(values))


def _report_failures(operator, module):
    for label, values in (
        ("Keying failures", module.key_failures),
        ("Animation-channel lookup failures", module.fcurve_lookup_failures),
        ("Pose/assignment failures", module.pose_failures),
    ):
        message = _format_failures(label, values)
        if message:
            operator.report({'WARNING'}, message)


# ----------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------

class PERFECTOVERLAP_OT_calculate(bpy.types.Operator):
    bl_idname = "perfect_overlap.calculate"
    bl_label = "Calculate"
    bl_description = "Calculate stable local-space overlapping follow-through animation."
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and context.mode == 'POSE'

    def execute(self, context):
        if not _blender_version_ok(self) or not _active_armature_ok(self, context):
            return {'CANCELLED'}

        obj = context.active_object
        if not _action_editable_ok(self, obj):
            return {'CANCELLED'}

        props = context.scene.perfect_overlap_props
        if not _frame_range_ok(self, props):
            return {'CANCELLED'}

        scene = context.scene
        original_frame = scene.frame_current
        original_subframe = _current_subframe(scene)
        module = _configure(PerfectOverlapSolver(), props)
        obj_trees = None
        operation_snapshot = None
        animation_touched = False
        failed = False
        result = {'FINISHED'}

        try:
            obj_trees = module.get_tree_list()
            if not obj_trees:
                self.report({'WARNING'}, "No usable selected control chain. Select child controls with a parent driver.")
                return {'CANCELLED'}

            operation_snapshot = module.capture_pose(obj_trees)

            # Phase 1 + 2: read and solve. Nothing is modified yet, so an
            # error here leaves the user's animation exactly as it was.
            source_cache = module.sample_source(obj_trees)
            if not source_cache:
                self.report({'ERROR'}, "Could not sample the source animation.")
                return {'CANCELLED'}

            results = module.solve_all(
                obj_trees,
                source_cache,
                cycle=bool(props.cycle),
                preroll=int(props.cycle_preroll) if props.cycle else 0,
            )
            if not results:
                self.report({'ERROR'}, "Overlap solver produced no output.")
                return {'CANCELLED'}

            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = module.capture_pose(obj_trees)

            # Phase 3: replace the animation in the requested range.
            animation_touched = True
            module.remove_owned_translation_animation(obj, obj_trees)
            module.del_animkey(obj_trees, start_pose)
            module.write_results(obj_trees, results, start_pose)

            # Post-processing. These steps polish the bake, so a failure in
            # one of them is only a warning.
            _safe_step(self, "Key reduction", module.cleanup_keys, obj_trees)
            _safe_step(self, "Cycle modifier cleanup", module.remove_addon_cycles, obj_trees)

            bone_count = sum(1 for _ in module.iter_bones(obj_trees))
            summary = "Overlap baked on {} bone(s), frames {} to {}.".format(
                bone_count, props.start_frame, props.end_frame
            )
            cycle_warning = ""

            if props.cycle:
                residual = _safe_step(self, "Cycle residual check", module.cycle_residual, obj_trees)
                _safe_step(self, "Cycle seam", module.enforce_cycle_seam, obj_trees)
                _safe_step(self, "Cycle seam tangents", module.set_cycle_seam_tangents, obj_trees)
                _safe_step(self, "Cycle modifier", module.add_cyclic_modifiers, obj_trees)
                if residual is not None:
                    name, angle = residual
                    if angle > math.radians(2.0):
                        cycle_warning = (
                            "Cycle rotation residual {:.2f} degrees on '{}'. "
                            "Increase Pre-roll Passes or use a clean source loop."
                        ).format(math.degrees(angle), name or "unknown")
                    else:
                        summary += " Cycle residual {:.2f} degrees after {} pre-roll pass(es).".format(
                            math.degrees(angle), int(props.cycle_preroll)
                        )

            if props.animate_translate:
                _safe_step(self, "Translation ownership", module.record_translation_ownership, obj_trees)

            self.report({'INFO'}, summary)
            if cycle_warning:
                self.report({'WARNING'}, cycle_warning)
            if props.animate_translate and module.translate_skipped:
                self.report(
                    {'WARNING'},
                    "Translation skipped on {} control(s) because they are connected or location-locked.".format(
                        len(module.translate_skipped)
                    ),
                )
            _report_failures(self, module)

        except Exception as exc:
            failed = True
            traceback.print_exc()
            if animation_touched:
                # Part of the animation was already replaced. Finish the
                # operator so Blender records ONE undo step for it.
                self.report(
                    {'ERROR'},
                    "Perfect Overlap stopped part-way ({}). Press Ctrl+Z to undo the partial bake.".format(exc),
                )
                result = {'FINISHED'}
            else:
                self.report(
                    {'ERROR'},
                    "Perfect Overlap failed, your animation was not changed: {}".format(exc),
                )
                result = {'CANCELLED'}

        finally:
            _restore_scene_state(
                scene, original_frame, original_subframe,
                module if (failed and animation_touched) else None,
                obj_trees, operation_snapshot,
            )

        return result


class PERFECTOVERLAP_OT_del_anim(bpy.types.Operator):
    bl_idname = "perfect_overlap.del_anim"
    bl_label = "Delete Keyframe"
    bl_description = "Delete Perfect Overlap rotation/location animation in the selected range."
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and context.mode == 'POSE'

    def execute(self, context):
        if not _blender_version_ok(self) or not _active_armature_ok(self, context):
            return {'CANCELLED'}

        obj = context.active_object
        if not _action_editable_ok(self, obj):
            return {'CANCELLED'}
        props = context.scene.perfect_overlap_props
        if not _frame_range_ok(self, props):
            return {'CANCELLED'}

        scene = context.scene
        original_frame = scene.frame_current
        original_subframe = _current_subframe(scene)
        module = _configure(PerfectOverlapSolver(), props)
        obj_trees = None
        operation_snapshot = None
        animation_touched = False
        failed = False
        result = {'FINISHED'}

        try:
            obj_trees = module.get_tree_list()
            if not obj_trees:
                self.report({'WARNING'}, "No usable selected control chain.")
                return {'CANCELLED'}

            operation_snapshot = module.capture_pose(obj_trees)
            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = module.capture_pose(obj_trees)

            animation_touched = True
            removed = module.remove_owned_translation_animation(obj, obj_trees)
            removed += module.del_animkey(obj_trees, start_pose)

            self.report({'INFO'}, "Deleted {} key(s) in frames {} to {}.".format(
                removed, props.start_frame, props.end_frame
            ))
            for label, values in (
                ("Keying failures", module.key_failures),
                ("Animation-channel lookup failures", module.fcurve_lookup_failures),
            ):
                message = _format_failures(label, values)
                if message:
                    self.report({'WARNING'}, message)

        except Exception as exc:
            failed = True
            traceback.print_exc()
            if animation_touched:
                self.report(
                    {'ERROR'},
                    "Perfect Overlap delete stopped part-way ({}). Press Ctrl+Z to undo it.".format(exc),
                )
                result = {'FINISHED'}
            else:
                self.report(
                    {'ERROR'},
                    "Perfect Overlap delete failed, your animation was not changed: {}".format(exc),
                )
                result = {'CANCELLED'}

        finally:
            _restore_scene_state(
                scene, original_frame, original_subframe,
                module if (failed and animation_touched) else None,
                obj_trees, operation_snapshot,
            )

        return result


class PERFECTOVERLAP_OT_reset_settings(bpy.types.Operator):
    bl_idname = "perfect_overlap.reset_settings"
    bl_label = "Reset Settings"
    bl_description = "Reset all Perfect Overlap settings to defaults."
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
        self.report({'INFO'}, "Perfect Overlap settings reset to defaults")
        return {'FINISHED'}


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

class PERFECTOVERLAP_PT_panel(bpy.types.Panel):
    bl_label = "Perfect Overlap"
    bl_idname = "PERFECTOVERLAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = "posemode"
    bl_category = "Perfect Overlap Addon"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.perfect_overlap_props

        layout.label(text="Frame")
        row = layout.row(align=True)
        row.prop(props, "start_frame")
        row.prop(props, "end_frame")

        layout.label(text="Properties")
        row = layout.row(align=True)
        row.prop(props, "delay")
        row.prop(props, "recursion")
        row.prop(props, "strength")
        row = layout.row(align=True)
        row.prop(props, "threshold")
        layout.label(text="Translation uses local bone space when enabled")

        layout.label(text="Options")
        box = layout.box()
        box.prop(props, "animate_translate")
        box.label(text="OFF = rotation only; no generated translation", icon='INFO')

        box = layout.box()
        box.prop(props, "cycle")
        if props.cycle:
            box.prop(props, "cycle_preroll")

        layout.prop(props, "debug")

        layout.label(text="Main")
        row = layout.row()
        row.scale_y = 1.8
        row.operator("perfect_overlap.calculate", icon="KEYTYPE_KEYFRAME_VEC")

        row = layout.row()
        row.operator("perfect_overlap.del_anim", icon="KEYFRAME")

        row = layout.row()
        row.operator("perfect_overlap.reset_settings", icon="LOOP_BACK")


classes = (
    PERFECTOVERLAP_PG_props,
    PERFECTOVERLAP_OT_calculate,
    PERFECTOVERLAP_OT_del_anim,
    PERFECTOVERLAP_OT_reset_settings,
    PERFECTOVERLAP_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.perfect_overlap_props = bpy.props.PointerProperty(
        type=PERFECTOVERLAP_PG_props
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
