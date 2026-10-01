# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Perfect Overlap Addon
#  Copyright 2021-2026 CaptainHansode, sakaiden.com
#  Copyright 2026 Sanku Venkatesh
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Perfect Overlap Addon",
    "author": "Sanku Venkatesh",
    "version": (2, 1, 0),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar > Perfect Overlap Addon (Pose Mode)",
    "description": (
        "Overlap and follow-through animation for bone chains with "
        "hierarchy detection, translation overlap and seamless cycles"
    ),
    "doc_url": "https://sakaiden.com",
    "category": "Animation",
}

import bpy
import math
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


# =============================================================================
# Constants
# =============================================================================

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
BONE_AXIS = mathutils.Vector((0.0, 1.0, 0.0))

MAX_LAG_ANGLE = math.radians(150.0)
MAX_HEAD_DRIFT = 2.0
COLLAPSED_AXIS_RATIO = 1.0e-6
MAX_INVERSE_ENTRY = 1.0e10

CYCLE_MODIFIER_NAME = "Perfect Overlap Cycle"


# =============================================================================
# Numerical helpers
# =============================================================================

def is_finite_vector(vec):
    try:
        return all(math.isfinite(float(v)) for v in vec)
    except (TypeError, ValueError, AttributeError):
        return False


def is_finite_matrix(mat):
    try:
        return all(math.isfinite(float(v)) for row in mat for v in row)
    except (TypeError, ValueError, AttributeError):
        return False


def has_collapsed_axis(mat):
    """Return True for matrices that are unsafe to invert."""
    try:
        scale = [abs(float(v)) for v in mat.to_scale()]
    except (ValueError, ArithmeticError, TypeError):
        return True

    if not scale:
        return True

    biggest = max(scale)
    if not math.isfinite(biggest) or biggest < EPS * EPS:
        return True

    if min(scale) <= biggest * COLLAPSED_AXIS_RATIO:
        return True

    try:
        det = float(mat.to_3x3().determinant())
    except (ValueError, ArithmeticError, TypeError):
        return True

    if not math.isfinite(det):
        return True

    # Avoid overflowing biggest ** 3 while keeping the test conservative.
    if biggest < 1.0e100:
        limit = (biggest ** 3) * COLLAPSED_AXIS_RATIO
        if abs(det) <= limit:
            return True

    return False


def _inverse_is_sane(inv):
    if not is_finite_matrix(inv):
        return False
    for row in inv:
        for value in row:
            if abs(float(value)) > MAX_INVERSE_ENTRY:
                return False
    return True


def rigid_part(mat):
    """Keep only location/rotation, removing scale/shear."""
    if not is_finite_matrix(mat):
        return mathutils.Matrix.Identity(4)
    try:
        loc, rot, _scale = mat.decompose()
        rigid = mathutils.Matrix.LocRotScale(loc, rot, None)
        if is_finite_matrix(rigid):
            return rigid
    except (ValueError, ArithmeticError, TypeError):
        pass
    return mathutils.Matrix.Identity(4)


def safe_inverse(mat):
    """Invert a matrix or return a safe rigid fallback."""
    if is_finite_matrix(mat) and not has_collapsed_axis(mat):
        try:
            inv = mat.inverted()
            if _inverse_is_sane(inv):
                return inv
        except (ValueError, ArithmeticError, TypeError):
            pass

    rigid = rigid_part(mat)
    try:
        return rigid.inverted()
    except (ValueError, ArithmeticError, TypeError):
        return mathutils.Matrix.Identity(4)


def safe_normalized(vec, fallback):
    try:
        if not is_finite_vector(vec):
            return fallback.copy()
        length = vec.length
        if not math.isfinite(length) or length < EPS:
            return fallback.copy()
        return vec.normalized()
    except (ValueError, ArithmeticError, TypeError):
        return fallback.copy()


# =============================================================================
# Solver
# =============================================================================

class PerfectOverlapSolver:

    def __init__(self, *args, **kwargs):
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

        # Soft-failure reporting. Sets prevent thousands of duplicate entries
        # when the same problem occurs on many frames.
        self.key_failures = set()
        self.matrix_assign_failures = set()

    # -------------------------------------------------------------------------
    # Failure reporting helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _record_failure(collection, value):
        if value:
            collection.add(str(value))

    # -------------------------------------------------------------------------
    # Solver coefficients
    # -------------------------------------------------------------------------

    def solver_params(self):
        k = 1.0 / max(float(self.delay), 1.0)
        k = min(max(k, 0.05), 0.9)

        rec = min(max(float(self.recursion), 0.0), 10.0)
        zeta = 1.15 - (rec / 10.0) * 0.8
        c = 2.0 * zeta * math.sqrt(k)
        c = min(max(c, 0.0), min(1.9, (4.0 - k) * 0.5 * 0.95))

        strength = min(max(float(self.strength), 1.0), 10.0)
        amp = 1.0 + (strength - 1.0) / 9.0
        return k, c, amp

    def _limit_dir(self, base, vec, max_angle):
        base = safe_normalized(base, BONE_AXIS)
        vec = safe_normalized(vec, base)
        try:
            dot = min(max(float(base.dot(vec)), -1.0), 1.0)
            angle = math.acos(dot)
            if angle <= max_angle or angle < EPS:
                return vec
            return safe_normalized(
                base.slerp(vec, max_angle / angle),
                base,
            )
        except (ValueError, ArithmeticError, TypeError):
            return base.copy()

    # -------------------------------------------------------------------------
    # Rig inspection
    # -------------------------------------------------------------------------

    def _bone_visible(self, pbn):
        bone = pbn.bone
        if bone.hide:
            return False

        collections = getattr(bone, "collections", None)
        if not collections:
            return True

        for collection in collections:
            # IMPORTANT: keep is_visible_effectively first. Blender renamed a
            # C++ helper to is_visible_with_ancestors, but explicitly retained
            # the RNA property is_visible_effectively for Python. The RNA
            # property represents effective viewport visibility.
            visible = getattr(
                collection,
                "is_visible_effectively",
                None,
            )
            if visible is None:
                visible = getattr(
                    collection,
                    "is_visible_with_ancestors",
                    None,
                )
            if visible is None:
                visible = getattr(
                    collection,
                    "is_visible",
                    True,
                )
            if visible:
                return True

        return False

    def is_control_bone(self, pbn):
        if pbn.name.lower().startswith(NON_CONTROL_PREFIXES):
            return False
        return self._bone_visible(pbn)

    def control_children(self, pbn, depth=0):
        if depth >= 8:
            return []

        result = []
        for child in pbn.children:
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
            if child.bone.use_connect:
                return child

        tail = pbn.tail
        direction = safe_normalized(pbn.tail - pbn.head, BONE_AXIS)
        scale = max(float(pbn.length), EPS)

        best = None
        best_score = None
        for child in candidates:
            distance = (child.head - tail).length / scale
            child_dir = safe_normalized(child.tail - child.head, -direction)
            alignment = float(child_dir.dot(direction))
            score = distance - alignment
            if best_score is None or score < best_score:
                best_score = score
                best = child
        return best

    def translate_axes(self, pbn):
        if not self.animate_translate or pbn.bone.use_connect:
            return []
        return [axis for axis in range(3) if not pbn.lock_location[axis]]

    def can_translate(self, pbn):
        return bool(self.translate_axes(pbn))

    # -------------------------------------------------------------------------
    # Selection / chain building
    # -------------------------------------------------------------------------

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

    def get_bone_depth(self, pbn):
        current = pbn
        depth = 0
        while current is not None:
            current = current.parent
            depth += 1
        return depth

    @staticmethod
    def get_default_data_table():
        return {
            "obj_list": [],
            "offset": [],
            "length": [],
            "state": [],
        }

    def get_tree_list(self):
        selected = self.get_selection()
        if not selected:
            return {}

        selected_names = {p.name for p in selected}
        ordered = sorted(selected, key=self.get_bone_depth)

        obj_trees = {}
        used = set()
        tree_count = 0

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
                    child for child in self.control_children(current)
                    if child.name in selected_names and child.name not in used
                ]
                next_bone = self.next_in_chain(current, candidates)
                if next_bone is None:
                    break
                chain.append(next_bone)
                used.add(next_bone.name)
                current = next_bone

            depth = self.get_bone_depth(pbn)
            obj_trees.setdefault(depth, {})
            table = self.get_default_data_table()
            table["obj_list"] = chain
            obj_trees[depth][f"tree{tree_count}"] = table
            tree_count += 1

        return obj_trees

    def iter_chains(self, obj_trees):
        for depth in sorted(obj_trees.keys()):
            for name in sorted(obj_trees[depth].keys()):
                yield obj_trees[depth][name]

    def iter_bones(self, obj_trees):
        for chain in self.iter_chains(obj_trees):
            yield from chain["obj_list"]

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def object_matrix(self, pbn):
        return pbn.matrix

    @staticmethod
    def _tip_of(matrix, length):
        return matrix @ mathutils.Vector((0.0, float(length), 0.0))

    def set_pre_data(self, obj_trees):
        self.translate_skipped = []

        for chain in self.iter_chains(obj_trees):
            obj_list = chain["obj_list"]
            chain["offset"] = []
            chain["length"] = []
            chain["state"] = []

            for index, pbn in enumerate(obj_list):
                ref = obj_list[index - 1] if index else pbn.parent
                ref_matrix = self.object_matrix(ref)
                bone_matrix = self.object_matrix(pbn)

                offset = safe_inverse(ref_matrix) @ bone_matrix
                if not is_finite_matrix(offset):
                    offset = mathutils.Matrix.Identity(4)
                chain["offset"].append(offset)

                length = max(float(pbn.length), EPS)
                chain["length"].append(length)

                tip = self._tip_of(bone_matrix, length)
                if not is_finite_vector(tip):
                    tip = bone_matrix.translation + BONE_AXIS * length

                chain["state"].append({
                    "head": bone_matrix.translation.copy(),
                    "head_vel": ZERO.copy(),
                    "tip": tip.copy(),
                    "tip_vel": ZERO.copy(),
                    "prev_quat": pbn.rotation_quaternion.copy(),
                    "prev_euler": pbn.rotation_euler.copy(),
                    "prev_axis_angle": tuple(float(v) for v in pbn.rotation_axis_angle),
                })

                if self.animate_translate and not self.can_translate(pbn):
                    self.translate_skipped.append(pbn.name)

        return obj_trees

    # -------------------------------------------------------------------------
    # Rotation continuity
    # -------------------------------------------------------------------------

    @staticmethod
    def rotation_data_path(pbn):
        if pbn.rotation_mode == 'QUATERNION':
            return 'rotation_quaternion'
        if pbn.rotation_mode == 'AXIS_ANGLE':
            return 'rotation_axis_angle'
        return 'rotation_euler'

    def _ensure_action_slot(self, id_data):
        """Ensure an existing Action has a compatible slot for the owner.

        This is called before keying when an Action already exists, and again
        after keying because Blender can create the Action/slot automatically
        on the first insertion.
        """
        if id_data is None:
            return True

        try:
            adt = id_data.animation_data
        except AttributeError:
            return False

        if adt is None or adt.action is None:
            return True  # keyframe_insert may create Action + Slot later.

        action = adt.action
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
            slot = anim_utils.action_get_first_suitable_slot(
                action, id_data.id_type
            )
            if slot is not None:
                adt.action_slot = slot
                return True
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass

        try:
            adt.action_slot = action.slots.new(
                id_type=id_data.id_type,
                name=id_data.name,
            )
            return True
        except (AttributeError, RuntimeError, TypeError):
            return False

    def make_rotation_continuous(self, pbn, state):
        mode = pbn.rotation_mode

        if mode == 'QUATERNION':
            quat = pbn.rotation_quaternion.copy()
            previous = state.get("prev_quat")
            if previous is not None:
                try:
                    if quat.dot(previous) < 0.0:
                        quat = mathutils.Quaternion((-quat.w, -quat.x, -quat.y, -quat.z))
                        pbn.rotation_quaternion = quat
                except (ValueError, ArithmeticError):
                    pass
            state["prev_quat"] = quat.copy()
            return

        if mode == 'AXIS_ANGLE':
            try:
                current = pbn.rotation_axis_angle
                current_angle = float(current[0])
                current_axis = safe_normalized(
                    mathutils.Vector((float(current[1]), float(current[2]), float(current[3]))),
                    mathutils.Vector((1.0, 0.0, 0.0)),
                )
            except (ValueError, ArithmeticError, TypeError):
                return

            previous = state.get("prev_axis_angle")
            if previous is None:
                state["prev_axis_angle"] = (
                    current_angle, current_axis.x, current_axis.y, current_axis.z
                )
                return

            try:
                previous_angle = float(previous[0])
                previous_axis = safe_normalized(
                    mathutils.Vector((float(previous[1]), float(previous[2]), float(previous[3]))),
                    current_axis,
                )
            except (ValueError, ArithmeticError, TypeError):
                previous_angle = current_angle
                previous_axis = current_axis.copy()

            best = None
            best_error = None
            two_pi = math.tau

            for sign in (1.0, -1.0):
                candidate_angle = current_angle * sign
                candidate_axis = current_axis * sign
                estimated_turns = round((previous_angle - candidate_angle) / two_pi)
                for turns in (estimated_turns - 1, estimated_turns, estimated_turns + 1):
                    test_angle = candidate_angle + two_pi * turns
                    angle_error = test_angle - previous_angle
                    axis_error = (candidate_axis - previous_axis).length_squared
                    error = angle_error * angle_error + axis_error * 0.25
                    if best_error is None or error < best_error:
                        best_error = error
                        best = (test_angle, candidate_axis.copy())

            if best is not None:
                angle, axis = best
                pbn.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
                state["prev_axis_angle"] = (angle, axis.x, axis.y, axis.z)
            return

        euler = pbn.rotation_euler.copy()
        previous = state.get("prev_euler")
        if previous is not None:
            try:
                euler.make_compatible(previous)
                pbn.rotation_euler = euler
            except (ValueError, ArithmeticError):
                pass
        state["prev_euler"] = euler.copy()

    # -------------------------------------------------------------------------
    # Keying
    # -------------------------------------------------------------------------

    def set_animkey(self, pbn):
        frame = bpy.context.scene.frame_current

        # Repair an already-existing Action before insertion.
        self._ensure_action_slot(pbn.id_data)

        for axis in self.translate_axes(pbn):
            try:
                result = pbn.keyframe_insert(
                    data_path='location', index=axis, frame=frame
                )
                if result is False:
                    self._record_failure(
                        self.key_failures,
                        f"{pbn.name} loc[{axis}]",
                    )
            except Exception as exc:
                self._record_failure(
                    self.key_failures,
                    f"{pbn.name} loc[{axis}]: {exc}",
                )

        rotation_path = self.rotation_data_path(pbn)
        try:
            result = pbn.keyframe_insert(
                data_path=rotation_path,
                frame=frame,
            )
            if result is False:
                self._record_failure(
                    self.key_failures,
                    f"{pbn.name} {rotation_path}",
                )
        except Exception as exc:
            self._record_failure(
                self.key_failures,
                f"{pbn.name} {rotation_path}: {exc}",
            )

        # Belt-and-braces repair after the first keyframe has potentially
        # created Action/slot data.
        if not self._ensure_action_slot(pbn.id_data):
            self._record_failure(
                self.key_failures,
                f"{pbn.name} Action slot",
            )

    # -------------------------------------------------------------------------
    # Pose capture/restore
    # -------------------------------------------------------------------------

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
            euler = values["rotation_euler"]
            pbn.rotation_euler = mathutils.Euler(
                (euler[0], euler[1], euler[2]), euler.order
            )
            aa = values["rotation_axis_angle"]
            pbn.rotation_axis_angle = (aa[0], aa[1], aa[2], aa[3])
            pbn.scale = values["scale"].copy()
        bpy.context.view_layer.update()

    # -------------------------------------------------------------------------
    # Blender 5.1 Action / F-Curve API
    # -------------------------------------------------------------------------

    def _fcurve_container(self, id_data):
        if id_data is None:
            return None

        try:
            adt = id_data.animation_data
        except AttributeError:
            return None

        if adt is None or adt.action is None:
            return None

        self._ensure_action_slot(id_data)
        slot = getattr(adt, "action_slot", None)
        if slot is None:
            return None

        try:
            from bpy_extras import anim_utils
            channelbag = anim_utils.action_get_channelbag_for_slot(
                adt.action, slot
            )
            return channelbag.fcurves if channelbag is not None else None
        except (ImportError, AttributeError, RuntimeError, TypeError):
            return None

    def _get_fcurves(self, id_data):
        container = self._fcurve_container(id_data)
        return list(container) if container is not None else []

    def _chain_fcurve_map(self, obj_trees):
        obj = bpy.context.active_object
        if obj is None:
            return []

        by_prefix = {pbn.path_from_id(): pbn for pbn in self.iter_bones(obj_trees)}
        result = []

        for fc in self._get_fcurves(obj):
            for prefix, pbn in by_prefix.items():
                if fc.data_path.startswith(prefix + "."):
                    channel = fc.data_path[len(prefix) + 1:]
                    result.append((fc, pbn, channel))
                    break
        return result

    def _baked_channels(self, obj_trees):
        groups = {}
        for fc, pbn, channel in self._chain_fcurve_map(obj_trees):
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

    # -------------------------------------------------------------------------
    # Cycle modifiers
    # -------------------------------------------------------------------------

    def remove_addon_cycles(self, obj_trees):
        for fc, _pbn in self._baked_fcurves(obj_trees):
            for modifier in list(fc.modifiers):
                if modifier.type == 'CYCLES' and modifier.name == CYCLE_MODIFIER_NAME:
                    try:
                        fc.modifiers.remove(modifier)
                    except (RuntimeError, ReferenceError):
                        pass

    def add_cyclic_modifiers(self, obj_trees):
        for fc, _pbn in self._baked_fcurves(obj_trees):
            if any(m.type == 'CYCLES' for m in fc.modifiers):
                continue
            try:
                modifier = fc.modifiers.new('CYCLES')
                modifier.name = CYCLE_MODIFIER_NAME
            except (RuntimeError, ReferenceError):
                pass

    # -------------------------------------------------------------------------
    # Key deletion / cleanup
    # -------------------------------------------------------------------------

    def del_animkey(self, obj_trees, pose_snapshot=None):
        container = self._fcurve_container(bpy.context.active_object)
        emptied = []

        for fc, pbn, channel in self._chain_fcurve_map(obj_trees):
            if channel == 'location':
                if fc.array_index not in self.translate_axes(pbn):
                    continue
            elif channel in ('rotation_euler', 'rotation_quaternion', 'rotation_axis_angle'):
                pass
            else:
                # SCALE and all other channels are untouched.
                continue

            keys = fc.keyframe_points
            remove_indices = [
                i for i, key in enumerate(keys)
                if self.sf <= float(key.co[0]) <= self.ef
            ]

            for i in reversed(remove_indices):
                try:
                    keys.remove(keys[i], fast=True)
                except (RuntimeError, ReferenceError):
                    pass

            fc.update()
            if len(keys) == 0:
                emptied.append(fc)

        if container is not None:
            for fc in emptied:
                try:
                    container.remove(fc)
                except (RuntimeError, ReferenceError):
                    pass

        bpy.context.view_layer.update()
        self.restore_pose(obj_trees, pose_snapshot)

        for pbn in self.iter_bones(obj_trees):
            for axis in self.translate_axes(pbn):
                try:
                    pbn.keyframe_insert(
                        data_path='location', index=axis, frame=self.sf
                    )
                except Exception as exc:
                    self._record_failure(
                        self.key_failures,
                        f"{pbn.name} loc[{axis}] at start: {exc}",
                    )
            try:
                pbn.keyframe_insert(
                    data_path=self.rotation_data_path(pbn),
                    frame=self.sf,
                )
            except Exception as exc:
                self._record_failure(
                    self.key_failures,
                    f"{pbn.name} rotation at start: {exc}",
                )

        self._ensure_action_slot(bpy.context.active_object)
        self.remove_addon_cycles(obj_trees)
        return None

    def cleanup_keys(self, obj_trees):
        tolerance = max(float(self.threshold), 1.0e-6)
        removed = 0

        for curves in self._baked_channels(obj_trees).values():
            fcurves = [fc for fc, _pbn in curves]
            frames = None

            for fc in fcurves:
                channel_frames = [
                    float(key.co[0])
                    for key in fc.keyframe_points
                    if self.sf <= float(key.co[0]) <= self.ef
                ]
                frames = channel_frames if frames is None else [
                    frame for frame in frames if frame in channel_frames
                ]

            if not frames or len(frames) <= 2:
                self._smooth(fcurves)
                continue

            values = []
            for fc in fcurves:
                lookup = {float(k.co[0]): float(k.co[1]) for k in fc.keyframe_points}
                values.append([lookup[f] for f in frames])

            drop = set()
            previous_index = 0

            for index in range(1, len(frames) - 1):
                t0 = frames[previous_index]
                t1 = frames[index]
                t2 = frames[index + 1]
                if abs(t2 - t0) < EPS:
                    continue

                ratio = (t1 - t0) / (t2 - t0)
                redundant = True

                for component in values:
                    predicted = component[previous_index] + (
                        component[index + 1] - component[previous_index]
                    ) * ratio
                    if abs(component[index] - predicted) > tolerance:
                        redundant = False
                        break

                if redundant:
                    drop.add(t1)
                else:
                    previous_index = index

            if drop:
                for fc in fcurves:
                    keys = fc.keyframe_points
                    indices = [
                        i for i, key in enumerate(keys)
                        if float(key.co[0]) in drop
                    ]
                    for i in reversed(indices):
                        try:
                            keys.remove(keys[i], fast=True)
                        except (RuntimeError, ReferenceError):
                            pass
                    removed += len(indices)
                    fc.update()

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
            fc.update()

    # -------------------------------------------------------------------------
    # Cycle residual / seam
    # -------------------------------------------------------------------------

    def _capture_cycle_matrices(self, obj_trees):
        result = {}
        for pbn in self.iter_bones(obj_trees):
            matrix = self.object_matrix(pbn)
            if not is_finite_matrix(matrix):
                continue
            result[pbn.name] = (
                matrix.translation.copy(),
                matrix.to_quaternion().normalized(),
                max(float(pbn.length), EPS),
            )
        return result

    def cycle_residual(self, obj_trees):
        scene = bpy.context.scene
        original_frame = scene.frame_current
        try:
            scene.frame_set(self.sf)
            bpy.context.view_layer.update()
            start = self._capture_cycle_matrices(obj_trees)

            scene.frame_set(self.ef)
            bpy.context.view_layer.update()
            end = self._capture_cycle_matrices(obj_trees)
        finally:
            scene.frame_set(original_frame)
            bpy.context.view_layer.update()

        worst = 0.0
        worst_name = ""
        worst_position = 0.0
        worst_angle = 0.0

        for name, (start_pos, start_quat, length) in start.items():
            if name not in end:
                continue

            end_pos, end_quat, _ = end[name]
            position_error = (end_pos - start_pos).length / length

            try:
                angular_error = start_quat.rotation_difference(end_quat).angle
            except (ValueError, ArithmeticError):
                angular_error = math.pi

            residual = max(float(position_error), float(angular_error))
            if residual > worst:
                worst = residual
                worst_name = name
                worst_position = float(position_error)
                worst_angle = float(angular_error)

        return worst_name, worst, worst_position, worst_angle

    @staticmethod
    def _seam_keys(fcurve, sf, ef):
        first = None
        last = None
        for key in fcurve.keyframe_points:
            frame = float(key.co[0])
            if math.isclose(frame, float(sf), abs_tol=EPS):
                first = key
            elif math.isclose(frame, float(ef), abs_tol=EPS):
                last = key
        return first, last

    def enforce_cycle_seam(self, obj_trees):
        for fc, _pbn in self._baked_fcurves(obj_trees):
            first, last = self._seam_keys(fc, self.sf, self.ef)
            if first is None or last is None or first is last:
                continue
            last.co[1] = first.co[1]
            fc.update()

    def set_cycle_seam_tangents(self, obj_trees):
        for fc, _pbn in self._baked_fcurves(obj_trees):
            first, last = self._seam_keys(fc, self.sf, self.ef)
            if first is None or last is None or first is last:
                continue

            keys = sorted(fc.keyframe_points, key=lambda k: float(k.co[0]))
            after = None
            before = None
            for key in keys:
                frame = float(key.co[0])
                if self.sf < frame < self.ef:
                    if after is None:
                        after = key
                    before = key

            if after is None or before is None:
                continue
            if after is before:
                continue

            dt_after = float(after.co[0]) - float(self.sf)
            dt_before = float(self.ef) - float(before.co[0])
            if dt_after < EPS or dt_before < EPS:
                continue

            slope = (
                float(after.co[1]) - float(before.co[1])
            ) / (dt_after + dt_before)

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

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def solve_bone(self, chain, index, driver_matrix, k, c, amp):
        pbn = chain["obj_list"][index]
        state = chain["state"][index]
        threshold = max(float(self.threshold), 1.0e-6)
        target_matrix = driver_matrix @ chain["offset"][index]

        if not is_finite_matrix(target_matrix):
            current = self.object_matrix(pbn)
            return current.copy() if is_finite_matrix(current) else mathutils.Matrix.Identity(4)

        try:
            target_quat = target_matrix.to_quaternion().normalized()
        except (ValueError, ArithmeticError):
            target_quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))

        target_head = target_matrix.translation.copy()
        target_tip = self._tip_of(target_matrix, chain["length"][index])
        if not is_finite_vector(target_head):
            target_head = state["head"].copy()
        if not is_finite_vector(target_tip):
            target_tip = state["tip"].copy()

        segment = target_tip - target_head
        segment_length = segment.length
        bone_length = max(float(chain["length"][index]), EPS)

        if not math.isfinite(segment_length) or segment_length < EPS:
            segment = safe_normalized(target_quat @ BONE_AXIS, BONE_AXIS)
            segment_length = bone_length
            target_tip = target_head + segment * segment_length

        target_dir = safe_normalized(target_tip - target_head, BONE_AXIS)

        if self.can_translate(pbn):
            head_velocity = (
                state["head_vel"] * (1.0 - c)
                + (target_head - state["head"]) * k
            )
            head_velocity = head_velocity if is_finite_vector(head_velocity) else ZERO.copy()
            if head_velocity.length < threshold:
                head_velocity = ZERO.copy()
            state["head_vel"] = head_velocity.copy()
            state["head"] = state["head"] + head_velocity

            drift = state["head"] - target_head
            if drift.length > segment_length * MAX_HEAD_DRIFT:
                state["head"] = (
                    target_head
                    + safe_normalized(drift, ZERO) * segment_length * MAX_HEAD_DRIFT
                )
                state["head_vel"] = ZERO.copy()

            out_head = target_head + (state["head"] - target_head) * amp
        else:
            state["head"] = target_head.copy()
            state["head_vel"] = ZERO.copy()
            out_head = target_head.copy()

        tip_velocity = (
            state["tip_vel"] * (1.0 - c)
            + (target_tip - state["tip"]) * k
        )
        tip_velocity = tip_velocity if is_finite_vector(tip_velocity) else ZERO.copy()
        if tip_velocity.length < threshold:
            tip_velocity = ZERO.copy()
        state["tip_vel"] = tip_velocity.copy()
        state["tip"] = state["tip"] + tip_velocity

        span = state["tip"] - state["head"]
        span_dir = safe_normalized(span, target_dir)
        state["tip"] = state["head"] + span_dir * segment_length

        sim_dir = safe_normalized(state["tip"] - state["head"], target_dir)
        if amp > 1.0:
            amplified = target_dir + (sim_dir - target_dir) * amp
            out_dir = safe_normalized(amplified, sim_dir)
        else:
            out_dir = sim_dir
        out_dir = self._limit_dir(target_dir, out_dir, MAX_LAG_ANGLE)

        try:
            out_quat = target_dir.rotation_difference(out_dir) @ target_quat
            out_quat.normalize()
        except (ValueError, ArithmeticError):
            out_quat = target_quat.copy()

        result = (
            mathutils.Matrix.Translation(out_head)
            @ out_quat.to_matrix().to_4x4()
        )

        if is_finite_matrix(result):
            return result

        current = self.object_matrix(pbn)
        return current.copy() if is_finite_matrix(current) else mathutils.Matrix.Identity(4)

    def solve_chain(self, chain, k, c, amp, write):
        first = chain["obj_list"][0]
        driver_matrix = self.object_matrix(first.parent)
        if not is_finite_matrix(driver_matrix):
            return

        for index, pbn in enumerate(chain["obj_list"]):
            original_location = pbn.location.copy()
            original_scale = pbn.scale.copy()
            translate_allowed = self.can_translate(pbn)

            new_matrix = self.solve_bone(
                chain, index, driver_matrix, k, c, amp
            )

            assigned = False
            if is_finite_matrix(new_matrix):
                try:
                    pbn.matrix = new_matrix
                    assigned = True
                except (ValueError, RuntimeError, TypeError) as exc:
                    self._record_failure(
                        self.matrix_assign_failures,
                        f"{pbn.name}: {exc}",
                    )

            if not assigned:
                # Do not pretend that downstream bones are valid if their
                # driver never received the requested result. Abort only this
                # chain; other selected chains can still be baked.
                break

            # Preserve animator scale exactly.
            try:
                pbn.scale = original_scale
            except (ValueError, RuntimeError, TypeError) as exc:
                self._record_failure(
                    self.matrix_assign_failures,
                    f"{pbn.name} scale restore: {exc}",
                )

            if not translate_allowed:
                try:
                    pbn.location = original_location
                except (ValueError, RuntimeError, TypeError) as exc:
                    self._record_failure(
                        self.matrix_assign_failures,
                        f"{pbn.name} location restore: {exc}",
                    )
            else:
                for axis in range(3):
                    if pbn.lock_location[axis]:
                        try:
                            pbn.location[axis] = original_location[axis]
                        except (ValueError, RuntimeError, TypeError) as exc:
                            self._record_failure(
                                self.matrix_assign_failures,
                                f"{pbn.name} lock[{axis}]: {exc}",
                            )

            # One dependency update per bone. The previous implementation did
            # two; the first was only needed to read the rotation representation,
            # which is already updated by the matrix setter.
            self.make_rotation_continuous(pbn, chain["state"][index])

            if write:
                self.set_animkey(pbn)

            # Keep this update: the next chain element must read the evaluated
            # result of the current bone (including constraints/drivers).
            bpy.context.view_layer.update()

            evaluated = self.object_matrix(pbn)
            if is_finite_matrix(evaluated):
                driver_matrix = evaluated
            else:
                driver_matrix = new_matrix

    def excute(self, obj_trees, write=True, include_start=False):
        """Historical spelling retained for compatibility."""
        k, c, amp = self.solver_params()
        scene = bpy.context.scene
        start = self.sf if include_start else self.sf + 1

        for frame in range(start, self.ef + 1):
            scene.frame_set(frame)
            for chain in self.iter_chains(obj_trees):
                self.solve_chain(chain, k, c, amp, write)

        self.passes_run += 1
        return True

    # Correctly-spelled public alias for new integrations.
    def execute(self, obj_trees, write=True, include_start=False):
        return self.excute(obj_trees, write=write, include_start=include_start)

    def solve_cycle(self, obj_trees, preroll):
        self.passes_run = 0
        passes = max(1, int(preroll))
        for _ in range(passes):
            self.excute(obj_trees, write=False, include_start=True)
        self.excute(obj_trees, write=True, include_start=True)
        return passes


# =============================================================================
# Properties
# =============================================================================

class PERFECTOVERLAP_PG_props(bpy.types.PropertyGroup):
    start_frame: bpy.props.IntProperty(name="Start Frame", default=0, min=0)
    end_frame: bpy.props.IntProperty(name="End Frame", default=100, min=1)

    delay: bpy.props.FloatProperty(
        name="Delay",
        default=3.0,
        min=1.0,
        max=10.0,
        description="How slowly the overlap catches the driving motion.",
    )

    recursion: bpy.props.FloatProperty(
        name="Recursion",
        default=5.0,
        min=0.0,
        max=10.0,
        description="How much momentum carries through the motion.",
    )

    strength: bpy.props.FloatProperty(
        name="Strength",
        default=1.0,
        min=1.0,
        max=10.0,
        description=(
            "Exaggerates the simulated directional offset. "
            "1 is natural; the maximum is deliberately limited."
        ),
    )

    threshold: bpy.props.FloatProperty(
        name="Threshold",
        default=0.001,
        min=0.00001,
        max=0.1,
        step=0.01,
        precision=4,
        description="Small motion ignored and used for key cleanup.",
    )

    debug: bpy.props.BoolProperty(name="Debug", default=False)

    animate_translate: bpy.props.BoolProperty(
        name="Translation",
        default=False,
        description=(
            "Overlap unlocked location channels as well as rotation. "
            "Connected bones and locked axes are left untouched."
        ),
    )

    cycle: bpy.props.BoolProperty(
        name="Cycle",
        default=False,
        description=(
            "Solve the range as a continuous loop and close the seam."
        ),
    )

    cycle_preroll: bpy.props.IntProperty(
        name="Pre-roll Passes",
        default=2,
        min=1,
        max=20,
        description=(
            "Number of cyclic passes used to settle the simulated state."
        ),
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
        title="Info",
        icon="INFO",
    )
    operator.report({'INFO'}, message)
    return False


def _blender_version_ok(operator):
    if bpy.app.version >= REQUIRED_BLENDER:
        return True
    message = (
        "Perfect Overlap requires Blender {}.{}.{} or newer."
        .format(*REQUIRED_BLENDER)
    )
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
    """Refuse edits on read-only linked IDs; permit genuinely editable IDs."""
    try:
        if not obj.is_editable:
            operator.report(
                {'ERROR'},
                "The active armature is read-only. Make it local or create an editable library override.",
            )
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
                operator.report(
                    {'ERROR'},
                    "The active Action is read-only. Assign a local editable Action before baking.",
                )
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


# =============================================================================
# Operators
# =============================================================================

class PERFECTOVERLAP_OT_calculate(bpy.types.Operator):
    bl_idname = "perfect_overlap.calculate"
    bl_label = "Calculate"
    bl_description = "Calculate overlapping follow-through animation."
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and context.mode == 'POSE'

    def execute(self, context):
        if not _blender_version_ok(self):
            return {'CANCELLED'}
        if not _active_armature_ok(self, context):
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
        failed = False
        operation_snapshot = None
        obj_trees = None

        try:
            # Determine chains at the user's current pose before changing time.
            obj_trees = module.get_tree_list()
            if not obj_trees:
                self.report(
                    {'WARNING'},
                    "No usable control chain. Select child controls; each chain needs a parent driver.",
                )
                return {'CANCELLED'}

            operation_snapshot = module.capture_pose(obj_trees)

            scene.frame_set(props.start_frame)
            _redraw()

            start_pose = module.capture_pose(obj_trees)
            module.del_animkey(obj_trees, start_pose)

            scene.frame_set(props.start_frame)
            module.restore_pose(obj_trees, start_pose)
            module.set_pre_data(obj_trees)

            if props.cycle:
                passes = module.solve_cycle(obj_trees, props.cycle_preroll)
                name, residual, position_residual, angle_residual = module.cycle_residual(obj_trees)

                module.cleanup_keys(obj_trees)
                module.enforce_cycle_seam(obj_trees)
                module.set_cycle_seam_tangents(obj_trees)
                module.add_cyclic_modifiers(obj_trees)

                if residual > 0.01:
                    self.report(
                        {'WARNING'},
                        (
                            "Cycle residual {:.4f} on '{}': position {:.4f} bone-lengths, "
                            "rotation {:.3f} degrees. Raise Pre-roll Passes or verify the source loop."
                        ).format(
                            residual,
                            name or "unknown",
                            position_residual,
                            math.degrees(angle_residual),
                        ),
                    )
                else:
                    self.report(
                        {'INFO'},
                        "Cycle solved over {} pre-roll passes + bake. Residual {:.5f}.".format(
                            passes, residual
                        ),
                    )
            else:
                module.excute(obj_trees, write=True, include_start=False)
                module.cleanup_keys(obj_trees)
                module.remove_addon_cycles(obj_trees)

            if props.animate_translate and module.translate_skipped:
                self.report(
                    {'WARNING'},
                    "Translation skipped on {} control(s) because they are connected or location-locked.".format(
                        len(module.translate_skipped)
                    ),
                )

            key_msg = _format_failures("Keying failures", module.key_failures)
            if key_msg:
                self.report({'WARNING'}, key_msg)

            matrix_msg = _format_failures(
                "Matrix/pose failures", module.matrix_assign_failures
            )
            if matrix_msg:
                self.report({'WARNING'}, matrix_msg)

        except Exception as exc:
            failed = True
            raise RuntimeError(
                "Perfect Overlap calculation failed: {}".format(exc)
            ) from exc

        finally:
            try:
                scene.frame_set(original_frame)
                bpy.context.view_layer.update()
                if failed and operation_snapshot:
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
        if not _blender_version_ok(self):
            return {'CANCELLED'}
        if not _active_armature_ok(self, context):
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
        failed = False
        operation_snapshot = None

        try:
            obj_trees = module.get_tree_list()
            if not obj_trees:
                self.report({'WARNING'}, "No usable selected control chain.")
                return {'CANCELLED'}

            operation_snapshot = module.capture_pose(obj_trees)
            scene.frame_set(props.start_frame)
            _redraw()

            start_pose = module.capture_pose(obj_trees)
            module.del_animkey(obj_trees, start_pose)
            scene.frame_set(props.start_frame)
            module.restore_pose(obj_trees, start_pose)

            key_msg = _format_failures("Keying failures", module.key_failures)
            if key_msg:
                self.report({'WARNING'}, key_msg)

        except Exception as exc:
            failed = True
            raise RuntimeError(
                "Perfect Overlap delete failed: {}".format(exc)
            ) from exc

        finally:
            try:
                scene.frame_set(original_frame)
                bpy.context.view_layer.update()
                if failed and operation_snapshot:
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
    bl_description = "Reset all Perfect Overlap settings to their defaults."
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


# =============================================================================
# Panel
# =============================================================================

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

        layout.label(text="Options")
        box = layout.box()
        box.column(align=True).prop(props, "animate_translate")

        box = layout.box()
        box.prop(props, "cycle")
        if props.cycle:
            box.column(align=True).prop(props, "cycle_preroll")

        layout.label(text="Main")
        row = layout.row()
        row.scale_y = 1.8
        row.operator("perfect_overlap.calculate", icon="KEYTYPE_KEYFRAME_VEC")
        layout.row().operator("perfect_overlap.del_anim", icon="KEYFRAME")
        layout.row().operator("perfect_overlap.reset_settings", icon="LOOP_BACK")


# =============================================================================
# Registration
# =============================================================================

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
