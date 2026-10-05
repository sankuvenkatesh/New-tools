# ##### BEGIN GPL LICENSE BLOCK #####
#
# Perfect Overlap Addon
# Version 2.3.0 - source-cache + local-space overlap solver.
# Copyright 2021-2026 CaptainHansode, sakaiden.com
# Copyright 2026 Sanku Venkatesh
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version
# 2 of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Perfect Overlap Addon",
    "author": "Sanku Venkatesh",
    "version": (2, 3, 0),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar > Perfect Overlap Addon (Pose Mode)",
    "description": (
        "Stable local-space overlap and follow-through for selected bone "
        "controls, with optional local translation and cycle support"
    ),
    "doc_url": "https://sakaiden.com",
    "category": "Animation",
}

import bpy
import math
import mathutils
import json

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


def is_finite_matrix(mat):
    try:
        return all(math.isfinite(float(v)) for row in mat for v in row)
    except (TypeError, ValueError, AttributeError):
        return False


def safe_quaternion(value, fallback=None):
    if fallback is None:
        fallback = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    try:
        q = value.copy()
        if q.length < EPS or not is_finite_quaternion(q):
            return fallback.copy()
        q.normalize()
        return q
    except (TypeError, ValueError, ArithmeticError):
        return fallback.copy()


def safe_vector(value, fallback=None):
    if fallback is None:
        fallback = ZERO
    try:
        v = value.copy()
        if not is_finite_vector(v):
            return fallback.copy()
        return v
    except (TypeError, ValueError, AttributeError):
        return fallback.copy()


def frame_list(sf, ef):
    return list(range(int(sf), int(ef) + 1))


class PerfectOverlapSolver:
    """Local-space overlap solver.

    Important design change from 2.2.x:
    - Source animation is sampled BEFORE any keys are deleted.
    - Each selected bone is solved from its own sampled local animation.
    - Rotation overlap edits only rotation properties.
    - Optional translation overlap edits local location only.
    - No pbn.matrix assignment is used by the overlap solver.

    That prevents world-space matrix writes from unintentionally changing
    location, IK/constraint feedback, or parent-relative transforms.
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
        self.matrix_assign_failures = set()

    @staticmethod
    def _record_failure(collection, value):
        if value:
            collection.add(str(value))

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

    @staticmethod
    def _default_chain():
        return {"obj_list": []}

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
            trees.setdefault(depth, {})[f"tree{count}"] = {
                **self._default_chain(),
                "obj_list": chain,
            }
            count += 1

        return trees

    def iter_chains(self, obj_trees):
        for depth in sorted(obj_trees.keys()):
            for name in sorted(obj_trees[depth].keys()):
                yield obj_trees[depth][name]

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
                if axis.length < EPS:
                    axis = mathutils.Vector((1.0, 0.0, 0.0))
                else:
                    axis.normalize()

                return safe_quaternion(
                    mathutils.Quaternion(axis, float(aa[0]))
                )

            # All Euler rotation modes (XYZ, ZYX, etc.) use the active
            # rotation_euler channel.
            return safe_quaternion(
                pbn.rotation_euler.to_quaternion()
            )

        except (ValueError, ArithmeticError, TypeError, AttributeError):
            return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))

    def _apply_quaternion_rotation(self, pbn, quat, state):
        q = safe_quaternion(quat)
        mode = pbn.rotation_mode

        try:
            if mode == 'QUATERNION':
                previous = state.get("last_quat")
                if previous is not None and q.dot(previous) < 0.0:
                    q.negate()
                pbn.rotation_quaternion = q
                state["last_quat"] = q.copy()
                return

            if mode == 'AXIS_ANGLE':
                angle = float(q.angle)
                axis = q.axis.copy()
                previous = state.get("last_axis_angle")
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
                state["last_axis_angle"] = (angle, axis.x, axis.y, axis.z)
                return

            euler = q.to_euler(mode)
            previous = state.get("last_euler")
            if previous is not None:
                try:
                    euler.make_compatible(previous)
                except (ValueError, ArithmeticError):
                    pass
            pbn.rotation_euler = euler
            state["last_euler"] = euler.copy()

        except Exception as exc:
            self._record_failure(
                self.matrix_assign_failures,
                f"{pbn.name} rotation assignment: {exc}",
            )

    # ------------------------------------------------------------------
    # Source sampling - MUST happen before key deletion
    # ------------------------------------------------------------------

    def sample_source(self, obj_trees):
        """Sample the user's evaluated local animation before destructive bake steps."""
        scene = bpy.context.scene
        original_frame = scene.frame_current
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
                    try:
                        frame_data[pbn.name] = {
                            "location": safe_vector(pbn.location),
                            "quaternion": self.bone_rotation_quaternion(pbn),
                            "scale": safe_vector(pbn.scale, mathutils.Vector((1.0, 1.0, 1.0))),
                        }
                    except (ReferenceError, AttributeError):
                        continue
                cache[frame] = frame_data
        finally:
            scene.frame_set(original_frame)
            bpy.context.view_layer.update()

        return cache

    # ------------------------------------------------------------------
    # Simulation state
    # ------------------------------------------------------------------

    def init_state(self, obj_trees, source_cache):
        states = {}
        self.translate_skipped = []

        first_frame = int(self.sf)
        frame_data = source_cache.get(first_frame, {})

        for pbn in self.iter_bones(obj_trees):
            data = frame_data.get(pbn.name)
            if data is None:
                data = {
                    "location": safe_vector(pbn.location),
                    "quaternion": self.bone_rotation_quaternion(pbn),
                    "scale": safe_vector(
                        pbn.scale,
                        mathutils.Vector((1.0, 1.0, 1.0)),
                    ),
                }

            if self.animate_translate and not self.translate_axes(pbn):
                self.translate_skipped.append(pbn.name)

            states[pbn.name] = {
                "location": data["location"].copy(),
                "location_velocity": ZERO.copy(),
                "quaternion": data["quaternion"].copy(),
                "angular_velocity": ZERO.copy(),
                "last_quat": data["quaternion"].copy(),
                "last_euler": pbn.rotation_euler.copy(),
                "last_axis_angle": tuple(float(v) for v in pbn.rotation_axis_angle),
            }

        return states

    def translate_axes(self, pbn):
        if not self.animate_translate:
            return []
        try:
            if pbn.bone.use_connect:
                return []
            return [i for i in range(3) if not pbn.lock_location[i]]
        except (AttributeError, ReferenceError):
            return []

    # ------------------------------------------------------------------
    # Local-space spring solver
    # ------------------------------------------------------------------

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
        except Exception:
            error = ZERO.copy()

        velocity = state["angular_velocity"] * damping + error * k
        if not is_finite_vector(velocity):
            velocity = ZERO.copy()

        step = velocity * amp
        step_len = step.length
        if step_len > MAX_ANGULAR_STEP:
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
        except Exception:
            result = current.copy()

        state["angular_velocity"] = velocity.copy()
        state["quaternion"] = result.copy()
        return result

    def _solve_location(self, state, target, pbn, k, damping, amp):
        target = safe_vector(target, state["location"])
        current = state["location"]
        allowed = self.translate_axes(pbn)

        # Keep all locked / disallowed axes exactly at the source value.
        for axis in range(3):
            if axis not in allowed:
                current[axis] = target[axis]
                state["location_velocity"][axis] = 0.0

        error = target - current
        velocity = state["location_velocity"] * damping + error * k
        if not is_finite_vector(velocity):
            velocity = ZERO.copy()

        length = max(float(getattr(pbn, "length", 1.0)), 1.0e-5)
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
            velocity *= 0.0

        state["location_velocity"] = velocity.copy()
        state["location"] = result.copy()
        return result

    def solve_frame(self, frame_data, states, obj_trees, k, damping, amp, write):
        for pbn in self.iter_bones(obj_trees):
            data = frame_data.get(pbn.name)
            state = states.get(pbn.name)
            if data is None or state is None:
                continue

            try:
                q = self._solve_rotation(
                    state,
                    data["quaternion"],
                    k,
                    damping,
                    amp,
                )
                self._apply_quaternion_rotation(pbn, q, state)

                # Preserve animator scale; overlap never synthesizes scale.
                try:
                    pbn.scale = data["scale"].copy()
                except Exception:
                    pass

                if self.animate_translate:
                    loc = self._solve_location(
                        state,
                        data["location"],
                        pbn,
                        k,
                        damping,
                        amp,
                    )
                    pbn.location = loc

                if write:
                    self.set_animkey(pbn)

            except Exception as exc:
                self._record_failure(
                    self.matrix_assign_failures,
                    f"{pbn.name}: {exc}",
                )

        bpy.context.view_layer.update()

    def solve_bake(self, obj_trees, source_cache, cycle=False, preroll=0):
        if not source_cache:
            return False

        k, damping, amp = self.solver_params()
        states = self.init_state(obj_trees, source_cache)
        frames = frame_list(self.sf, self.ef)

        # The Start Frame remains the exact source pose. This avoids creating
        # a visible jump at the beginning of the selected range.
        scene = bpy.context.scene
        scene.frame_set(self.sf)
        bpy.context.view_layer.update()

        first = source_cache.get(self.sf, {})
        for pbn in self.iter_bones(obj_trees):
            data = first.get(pbn.name)
            if data is None:
                continue
            state = states[pbn.name]
            self._apply_quaternion_rotation(pbn, data["quaternion"], state)
            if self.animate_translate:
                pbn.location = data["location"].copy()
            try:
                pbn.scale = data["scale"].copy()
            except Exception:
                pass
            self.set_animkey(pbn)
        bpy.context.view_layer.update()

        if cycle and preroll > 0:
            # Settle the state through repeated source loops without writing.
            for _ in range(int(preroll)):
                for frame in frames[1:]:
                    scene.frame_set(frame)
                    self.solve_frame(
                        source_cache.get(frame, {}),
                        states,
                        obj_trees,
                        k,
                        damping,
                        amp,
                        write=False,
                    )
                scene.frame_set(self.sf)
                bpy.context.view_layer.update()

            # Reset to the source Start Frame for the actual visible bake, but
            # keep the settled velocities. The output begins exactly at source.
            start_data = source_cache.get(self.sf, {})
            for pbn in self.iter_bones(obj_trees):
                data = start_data.get(pbn.name)
                if data is None:
                    continue
                states[pbn.name]["quaternion"] = data["quaternion"].copy()
                states[pbn.name]["location"] = data["location"].copy()
                states[pbn.name]["last_quat"] = data["quaternion"].copy()
                if self.animate_translate:
                    pbn.location = data["location"].copy()
                self._apply_quaternion_rotation(
                    pbn,
                    data["quaternion"],
                    states[pbn.name],
                )
            bpy.context.view_layer.update()

        for frame in frames[1:]:
            scene.frame_set(frame)
            self.solve_frame(
                source_cache.get(frame, {}),
                states,
                obj_trees,
                k,
                damping,
                amp,
                write=True,
            )

        self.passes_run = int(preroll) if cycle else 1
        return True

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
        self._ensure_action_slot(pbn.id_data)

        for axis in self.translate_axes(pbn):
            try:
                result = pbn.keyframe_insert(
                    data_path='location', index=axis, frame=frame
                )
                if result is False:
                    self._record_failure(self.key_failures, f"{pbn.name} loc[{axis}]")
            except Exception as exc:
                self._record_failure(self.key_failures, f"{pbn.name} loc[{axis}]: {exc}")

        try:
            result = pbn.keyframe_insert(
                data_path=self.rotation_data_path(pbn),
                frame=frame,
            )
            if result is False:
                self._record_failure(self.key_failures, f"{pbn.name} rotation")
        except Exception as exc:
            self._record_failure(self.key_failures, f"{pbn.name} rotation: {exc}")

        if not self._ensure_action_slot(pbn.id_data):
            self._record_failure(self.key_failures, f"{pbn.name} Action slot")

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
                "rotation_axis_angle": tuple(float(v) for v in pbn.rotation_axis_angle),
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
            aa = values["rotation_axis_angle"]
            pbn.rotation_axis_angle = aa
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
                self._record_failure(self.fcurve_lookup_failures, f"{id_data.name} Action slot")
            return None

        slot = getattr(adt, "action_slot", None)
        if slot is None:
            if report_missing:
                self._record_failure(self.fcurve_lookup_failures, f"{id_data.name} assigned Action slot")
            return None

        try:
            from bpy_extras import anim_utils
            helper = getattr(anim_utils, "animdata_get_channelbag_for_assigned_slot", None)
            channelbag = helper(adt) if helper is not None else None
            if channelbag is None:
                fallback = getattr(anim_utils, "action_get_channelbag_for_slot", None)
                if fallback is not None:
                    channelbag = fallback(adt.action, slot)
            if channelbag is None:
                if report_missing:
                    self._record_failure(self.fcurve_lookup_failures, f"{id_data.name} channelbag")
                return None
            return channelbag.fcurves
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            if report_missing:
                self._record_failure(self.fcurve_lookup_failures, f"{id_data.name} channelbag lookup: {exc}")
            return None

    def _get_fcurves(self, id_data, report_missing=False):
        container = self._fcurve_container(id_data, report_missing)
        return list(container) if container is not None else []

    def _chain_fcurve_map(self, obj_trees, report_missing=True):
        obj = bpy.context.active_object
        if obj is None:
            return []
        by_prefix = {pbn.path_from_id(): pbn for pbn in self.iter_bones(obj_trees)}
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
            elif channel == 'location' and fc.array_index in self.translate_axes(pbn):
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
            raw = action.get("_perfect_overlap_translation_ownership_v2")
        if not raw:
            return []
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            self._record_failure(self.fcurve_lookup_failures, f"{id_data.name} translation ownership metadata")
            return []
        return [entry for entry in data if isinstance(entry, dict)] if isinstance(data, list) else []

    def _save_translation_ownership(self, id_data, entries):
        action, _adt, _slot = self._assigned_action_slot(id_data)
        if action is None:
            return True
        try:
            action[TRANSLATION_OWNERSHIP_KEY] = json.dumps(entries, separators=(",", ":"))
            return True
        except Exception as exc:
            self._record_failure(self.fcurve_lookup_failures, f"{id_data.name} translation ownership save: {exc}")
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
            selected_paths = {pbn.path_from_id() + ".location" for pbn in self.iter_bones(obj_trees)}

        targets = []
        kept_entries = []
        for entry in entries:
            if not self._ownership_entry_matches(entry, id_data, slot):
                kept_entries.append(entry)
                continue
            if selected_paths is not None and entry.get("data_path") not in selected_paths:
                kept_entries.append(entry)
                continue
            targets.append(entry)

        if not targets:
            return 0

        container = self._fcurve_container(id_data, True)
        if container is None:
            return 0

        curves = {(fc.data_path, int(fc.array_index)): fc for fc in list(container)}
        sf = float(self.sf)
        ef = float(self.ef)
        remove_frames = {}
        retained = {}
        target_curve_keys = set()

        for entry in targets:
            key = (entry.get("data_path", ""), int(entry.get("array_index", 0)))
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
        selected_paths = {pbn.path_from_id() + ".location" for pbn in self.iter_bones(obj_trees)}
        sf = float(self.sf)
        ef = float(self.ef)

        kept = []
        preserved_by_key = {}
        for entry in existing:
            if not self._ownership_entry_matches(entry, obj, slot) or entry.get("data_path") not in selected_paths:
                kept.append(entry)
                continue
            frames = self._normalized_owned_frames(entry)
            outside = [f for f in frames if f < sf or f > ef]
            if not outside:
                continue
            key = (entry.get("data_path", ""), int(entry.get("array_index", 0)))
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
        selected_by_path = {pbn.path_from_id() + ".location": pbn for pbn in self.iter_bones(obj_trees)}

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
        """Delete baked rotation keys and selected translation keys in range.

        When Translation is enabled, all existing location keys for unlocked
        translation axes in the selected range are removed before the new
        bake is written. This is intentional: the bake owns that channel
        over the requested range, so manually keyed location values inside
        the range will be replaced by the generated result.
        """
        container = self._fcurve_container(bpy.context.active_object)
        emptied = []

        for fc, pbn, channel in self._chain_fcurve_map(obj_trees):
            if channel == 'location':
                if fc.array_index not in self.translate_axes(pbn):
                    continue
            elif channel in ('rotation_euler', 'rotation_quaternion', 'rotation_axis_angle'):
                pass
            else:
                continue

            indices = [
                i for i, key in enumerate(fc.keyframe_points)
                if self.sf <= float(key.co[0]) <= self.ef
            ]
            for i in reversed(indices):
                try:
                    fc.keyframe_points.remove(fc.keyframe_points[i], fast=True)
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
        self._ensure_action_slot(bpy.context.active_object)
        self.remove_addon_cycles(obj_trees)

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
                self._record_failure(self.fcurve_lookup_failures, f"{bone_name} {channel} incomplete key set")
                self._smooth(fcurves)
                continue

            tolerance = self._cleanup_tolerance_for_group(channel, reference_bone)
            keep = self._douglas_peucker_keep_indices(values, frames, tolerance)
            remove_frames = {frames[i] for i in range(len(frames)) if i not in keep}
            for fc in fcurves:
                indices = [i for i, k in enumerate(fc.keyframe_points) if float(k.co[0]) in remove_frames]
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
        for fc, _ in self._baked_fcurves(obj_trees):
            for modifier in list(fc.modifiers):
                if modifier.type == 'CYCLES' and modifier.name == CYCLE_MODIFIER_NAME:
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
                mod.name = CYCLE_MODIFIER_NAME
            except (RuntimeError, ReferenceError):
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
        scene = bpy.context.scene
        original_frame = scene.frame_current
        try:
            scene.frame_set(self.sf)
            bpy.context.view_layer.update()
            start = {p.name: self.bone_rotation_quaternion(p) for p in self.iter_bones(obj_trees)}
            scene.frame_set(self.ef)
            bpy.context.view_layer.update()
            end = {p.name: self.bone_rotation_quaternion(p) for p in self.iter_bones(obj_trees)}
        finally:
            scene.frame_set(original_frame)
            bpy.context.view_layer.update()

        worst = 0.0
        worst_name = ""
        worst_angle = 0.0
        for name, q0 in start.items():
            q1 = end.get(name)
            if q1 is None:
                continue
            try:
                angle = q0.rotation_difference(q1).angle
            except Exception:
                angle = math.pi
            if angle > worst:
                worst = float(angle)
                worst_name = name
                worst_angle = float(angle)
        return worst_name, worst, worst_angle


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
    debug: bpy.props.BoolProperty(name="Debug", default=False)
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


def _redraw():
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except RuntimeError:
        pass


def _frame_range_ok(operator, props):
    if props.start_frame < props.end_frame:
        return True
    message = "Make the Start Frame smaller than the End Frame."
    bpy.context.window_manager.popup_menu(
        lambda self, context: self.layout.label(text=message),
        title="Info", icon="INFO",
    )
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
        return f"{label}: {', '.join(values[:limit])} (+{len(values) - limit} more)"
    return f"{label}: {', '.join(values)}"


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
        module = _configure(PerfectOverlapSolver(), props)
        obj_trees = None
        operation_snapshot = None
        failed = False

        try:
            obj_trees = module.get_tree_list()
            if not obj_trees:
                self.report({'WARNING'}, "No usable selected control chain. Select child controls with a parent driver.")
                return {'CANCELLED'}

            operation_snapshot = module.capture_pose(obj_trees)

            # CRITICAL: sample the source animation BEFORE deleting keys.
            source_cache = module.sample_source(obj_trees)
            if not source_cache:
                self.report({'ERROR'}, "Could not sample the source animation.")
                return {'CANCELLED'}

            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = module.capture_pose(obj_trees)

            # Remove only previous Perfect Overlap-owned translation keys.
            module.remove_owned_translation_animation(obj, obj_trees)
            bpy.context.view_layer.update()

            # Delete rotation keys in the requested range. The source is safe in
            # source_cache, so the bake no longer depends on deleted animation.
            module.del_animkey(obj_trees, start_pose)

            solved = module.solve_bake(
                obj_trees,
                source_cache,
                cycle=bool(props.cycle),
                preroll=int(props.cycle_preroll) if props.cycle else 0,
            )
            if not solved:
                raise RuntimeError("Overlap solver produced no output.")

            module.cleanup_keys(obj_trees)
            module.remove_addon_cycles(obj_trees)

            if props.cycle:
                name, residual, angle = module.cycle_residual(obj_trees)
                module.enforce_cycle_seam(obj_trees)
                module.set_cycle_seam_tangents(obj_trees)
                module.add_cyclic_modifiers(obj_trees)
                if props.animate_translate:
                    module.record_translation_ownership(obj_trees)
                if residual > math.radians(2.0):
                    self.report(
                        {'WARNING'},
                        "Cycle rotation residual {:.2f} degrees on '{}'. Increase Pre-roll Passes or use a clean source loop.".format(
                            math.degrees(angle), name or "unknown"
                        ),
                    )
                else:
                    self.report(
                        {'INFO'},
                        "Cycle solved with {} pre-roll pass(es). Residual {:.2f} degrees.".format(
                            int(props.cycle_preroll), math.degrees(residual)
                        ),
                    )
            elif props.animate_translate:
                module.record_translation_ownership(obj_trees)

            if props.animate_translate and module.translate_skipped:
                self.report(
                    {'WARNING'},
                    "Translation skipped on {} control(s) because they are connected or location-locked.".format(
                        len(module.translate_skipped)
                    ),
                )

            for label, values in (
                ("Keying failures", module.key_failures),
                ("Animation-channel lookup failures", module.fcurve_lookup_failures),
                ("Pose/assignment failures", module.matrix_assign_failures),
            ):
                message = _format_failures(label, values)
                if message:
                    self.report({'WARNING'}, message)

        except Exception as exc:
            failed = True
            raise RuntimeError("Perfect Overlap calculation failed: {}".format(exc)) from exc

        finally:
            try:
                scene.frame_set(original_frame)
                bpy.context.view_layer.update()
                if failed and operation_snapshot and obj_trees:
                    module.restore_pose(obj_trees, operation_snapshot)
                    scene.frame_set(original_frame)
                    bpy.context.view_layer.update()
                _redraw()
            except Exception:
                pass

        return {'FINISHED'}


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
        module = _configure(PerfectOverlapSolver(), props)
        obj_trees = None
        operation_snapshot = None
        failed = False

        try:
            obj_trees = module.get_tree_list()
            if not obj_trees:
                self.report({'WARNING'}, "No usable selected control chain.")
                return {'CANCELLED'}

            operation_snapshot = module.capture_pose(obj_trees)
            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = module.capture_pose(obj_trees)

            module.remove_owned_translation_animation(obj, obj_trees)
            bpy.context.view_layer.update()
            module.del_animkey(obj_trees, start_pose)

            key_msg = _format_failures("Keying failures", module.key_failures)
            if key_msg:
                self.report({'WARNING'}, key_msg)
            lookup_msg = _format_failures("Animation-channel lookup failures", module.fcurve_lookup_failures)
            if lookup_msg:
                self.report({'WARNING'}, lookup_msg)

        except Exception as exc:
            failed = True
            raise RuntimeError("Perfect Overlap delete failed: {}".format(exc)) from exc

        finally:
            try:
                scene.frame_set(original_frame)
                bpy.context.view_layer.update()
                if failed and operation_snapshot and obj_trees:
                    module.restore_pose(obj_trees, operation_snapshot)
                    scene.frame_set(original_frame)
                    bpy.context.view_layer.update()
                _redraw()
            except Exception:
                pass

        return {'FINISHED'}


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
