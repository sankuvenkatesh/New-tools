bl_info = {
    "name": "Safe Scale Normalizer",
    "author": "Venkatesh Sanku",
    "version": (4, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Scale Fix",
    "description": (
        "Finds and removes negative / mirrored scale on every object type "
        "without moving geometry, children, rigs or animation"
    ),
    "category": "Object",
}

# ============================================================================
# HOW IT WORKS (short version)
# ----------------------------------------------------------------------------
# A negative scale is a mirror. A mirror can be removed from an object's
# transform only if the same mirror is moved somewhere else, so that every
# vertex, child and bone stays exactly where it is in world space:
#
#   * Meshes, curves, surfaces, metaballs, lattices, point clouds, hair and
#     grease pencil: the mirror is baked into the object's data, normals are
#     kept pointing outwards, rotation is untouched.
#   * Two negative axes are just a 180 degree rotation, so lights, cameras,
#     text, armatures and special empties get a rotation change instead and
#     their data is never touched.
#   * A single remaining mirror on an armature is moved into the rest pose
#     (bones, pose, animation, constraints and drivers are all converted).
#   * Children keep their world transform through their parent-inverse
#     matrix. Mirrors hidden in a parent-inverse matrix are found and removed.
#
# Nothing is assumed to work: every object is checked after the change
# (world matrices of the whole view layer, evaluated vertex positions and face
# normals, bone positions) and rolled back exactly if anything moved.
# ============================================================================

import math
import re
import traceback
from contextlib import contextmanager

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, PointerProperty, StringProperty
from mathutils import Euler, Matrix, Quaternion, Vector


# ============================================================================
# CONFIGURATION
# ============================================================================

EPS = 1.0e-7                  # values below -EPS count as negative
MATRIX_TOL = 1.0e-4           # relative tolerance for matrix comparisons
POSITION_TOL = 1.0e-4         # default relative tolerance for vertex positions
MIN_FACE_AREA = 1.0e-12       # faces smaller than this are ignored for normals
MAX_PANEL_ROWS = 8

# Set from the Tolerance option at the start of every run.
ACTIVE = {"tolerance": POSITION_TOL, "dependents": None}
IK_TOLERANCE = 1.0e-3

ADDON_PREFIX = "[Safe Scale Normalizer]"
REPORT_TEXT_NAME = "Scale Fix Report"
AXES = "XYZ"

IDENTITY = (1, 1, 1)
MIRROR_X = (-1, 1, 1)
FX3 = Matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
FX4 = FX3.to_4x4()

# Objects whose data can absorb a mirror.
DATA_TYPES = {
    'MESH', 'CURVE', 'SURFACE', 'META', 'LATTICE',
    'POINTCLOUD', 'CURVES', 'GREASEPENCIL',
}

# Objects where a single X mirror is invisible (symmetric emitters).
SYMMETRIC_TYPES = {'LIGHT', 'SPEAKER', 'LIGHT_PROBE'}

# Force fields whose effect has a handedness (a mirror changes them).
HANDED_FIELDS = {'VORTEX', 'MAGNET', 'TURBULENCE', 'TEXTURE'}

# Parent types whose parent matrix is the parent's world matrix.
VERTEX_PARENT_TYPES = {'VERTEX', 'VERTEX_3'}

# F-curve modifiers that stay correct when key values are negated.
SAFE_FCURVE_MODIFIERS = {'CYCLES', 'STEPPED'}

TYPE_LABEL = {
    'MESH': "Mesh", 'CURVE': "Curve", 'SURFACE': "Surface",
    'META': "Metaball", 'FONT': "Text", 'LATTICE': "Lattice",
    'EMPTY': "Empty", 'ARMATURE': "Armature", 'LIGHT': "Light",
    'CAMERA': "Camera", 'SPEAKER': "Speaker", 'LIGHT_PROBE': "Light Probe",
    'GREASEPENCIL': "Grease Pencil", 'CURVES': "Hair Curves",
    'POINTCLOUD': "Point Cloud", 'VOLUME': "Volume",
}

# Pose-bone channels that change sign when a bone's local X axis is mirrored.
# property name -> array indices to negate
POSE_MIRROR_CHANNELS = {
    "location": (0,),
    "rotation_quaternion": (2, 3),
    "rotation_euler": (1, 2),
    "rotation_axis_angle": (2, 3),
    "bbone_curveinx": (0,),
    "bbone_curveoutx": (0,),
    "bbone_rollin": (0,),
    "bbone_rollout": (0,),
}

BONE_PATH_RE = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(\w+)$')
BONE_PROP_PATH_RE = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(\w+)(?:\[(\d+)\])?$')
SIDE_SUFFIX_RE = re.compile(r'(?:[._\- ](?:L|R|l|r|Left|Right|left|right))$')


# ============================================================================
# LOGGING
# ============================================================================

def log(message):
    print(f"{ADDON_PREFIX} {message}")


# ============================================================================
# SMALL MATH HELPERS
# ============================================================================

def sign(value):
    """-1 for negative, +1 otherwise (zero counts as positive)."""
    return -1 if float(value) < -EPS else 1


def signs_of(values):
    return tuple(sign(v) for v in values)


def mul_signs(a, b):
    return tuple(x * y for x, y in zip(a, b))


def is_mirror(signs):
    return signs.count(-1) % 2 == 1


def sign_matrix4(signs):
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][i] = float(signs[i])
    return m


def det3(matrix):
    return matrix.to_3x3().determinant()


def handedness(matrix):
    """+1 normal, -1 mirrored, 0 singular."""
    d = det3(matrix)
    if d < -EPS:
        return -1
    if d > EPS:
        return 1
    return 0


def axes_text(signs):
    """Axes that are negative in a sign tuple, e.g. (-1, 1, -1) -> 'X+Z'."""
    idx = [i for i in range(3) if signs[i] < 0]
    return "+".join(AXES[i] for i in idx) if idx else "none"


def index_text(indices):
    return "+".join(AXES[i] for i in indices) if indices else "none"


def vec_text(values):
    return "(" + ", ".join(f"{float(v):.4f}" for v in values) + ")"


def matrix_close(a, b, tolerance=MATRIX_TOL):
    scale = 1.0
    for row in range(4):
        for col in range(4):
            scale = max(scale, abs(a[row][col]), abs(b[row][col]))
    limit = tolerance * scale
    for row in range(4):
        for col in range(4):
            if abs(a[row][col] - b[row][col]) > limit:
                return False
    return True


def rotation_matrix3(owner, include_delta=True):
    """Rotation part of an Object or PoseBone basis (Blender's own order)."""
    mode = owner.rotation_mode

    if mode == 'QUATERNION':
        rot = owner.rotation_quaternion.normalized().to_matrix()
    elif mode == 'AXIS_ANGLE':
        angle, ax, ay, az = owner.rotation_axis_angle
        axis = Vector((ax, ay, az))
        rot = Matrix.Rotation(angle, 3, axis) if axis.length > EPS else Matrix.Identity(3)
    else:
        rot = Euler(owner.rotation_euler[:], mode).to_matrix()

    if include_delta and hasattr(owner, "delta_rotation_euler"):
        if mode in {'QUATERNION', 'AXIS_ANGLE'}:
            delta = owner.delta_rotation_quaternion.normalized().to_matrix()
        else:
            delta = Euler(owner.delta_rotation_euler[:], mode).to_matrix()
        rot = delta @ rot

    return rot


def quaternion_turn_map(axis):
    """Channel map for q' = q @ (180 deg about axis).

    Returns {new_index: (old_index, sign)}; the map is linear, so it is exact
    for animated quaternion channels too.
    """
    turn = Quaternion((0.0,) + tuple(1.0 if i == axis else 0.0 for i in range(3)))
    mapping = {}
    for old in range(4):
        basis = [0.0, 0.0, 0.0, 0.0]
        basis[old] = 1.0
        result = Quaternion(basis) @ turn
        for new in range(4):
            if abs(result[new]) > 0.5:
                mapping[new] = (old, 1 if result[new] > 0 else -1)
    return mapping


def euler_turn_rule(order, axis):
    """How Euler channels change for R' = R @ (180 deg about axis).

    Blender applies Euler 'ABC' as A first: matrix = C @ B @ A. Pushing the
    turn through every inner rotation negates it; the turn axis itself gets
    +pi. Returns (indices_to_negate, index_to_offset).
    """
    pos = order.index(AXES[axis])
    negate = tuple(AXES.index(letter) for letter in order[:pos])
    return negate, axis


# ============================================================================
# NAMES AND DESCRIPTIONS
# ============================================================================

def label(obj):
    return f"{TYPE_LABEL.get(obj.type, obj.type.title())} '{obj.name}'"


def depth(obj):
    result = 0
    parent = obj.parent
    while parent is not None and result < 1024:
        result += 1
        parent = parent.parent
    return result


def hierarchy_path(obj):
    chain = []
    current = obj
    guard = 0
    while current is not None and guard < 1024:
        chain.append(current)
        current = current.parent
        guard += 1
    chain.reverse()

    parts = []
    for item in chain:
        if item.parent is not None and item.parent_type == 'BONE' and item.parent_bone:
            parts.append(f"bone '{item.parent_bone}'")
        elif item.parent is not None and item.parent_type in VERTEX_PARENT_TYPES:
            parts.append("vertex parent")
        parts.append(label(item))
    return " > ".join(parts)


def in_view_layer(context, obj):
    return context.view_layer.objects.get(obj.name) == obj


# ============================================================================
# EXCEPTIONS
# ============================================================================

class Blocked(Exception):
    """The fix needs a decision from the user or cannot be done safely."""

    def __init__(self, reason, hint=""):
        super().__init__(reason)
        self.reason = reason
        self.hint = hint


class VerifyFailed(Exception):
    """The fix ran but something moved; everything was rolled back."""


# ============================================================================
# JOURNAL (EXACT ROLLBACK)
# ============================================================================

def _copy_value(value):
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    if hasattr(value, "copy"):
        return value.copy()
    try:
        return tuple(value)
    except TypeError:
        return value


class Journal:
    """Records every change so it can be undone in reverse order."""

    def __init__(self):
        self._undo = []

    def set(self, owner, attr, value):
        old = _copy_value(getattr(owner, attr))
        self._undo.append(lambda o=owner, a=attr, v=old: setattr(o, a, v))
        setattr(owner, attr, value)

    def push(self, undo_function):
        self._undo.append(undo_function)

    def rollback(self):
        errors = []
        for undo in reversed(self._undo):
            try:
                undo()
            except Exception as exc:  # keep undoing the rest
                errors.append(str(exc))
        self._undo.clear()
        return errors

    def commit(self):
        self._undo.clear()


# ============================================================================
# REPORT
# ============================================================================

STATUS_ORDER = ('FIXED', 'FIXED_WARN', 'BLOCKED', 'FAILED', 'CLEAN')
STATUS_WORDS = {
    'FIX': {
        'FIXED': "FIXED",
        'FIXED_WARN': "FIXED (read the notes)",
        'BLOCKED': "NEEDS YOUR DECISION",
        'FAILED': "NOT CHANGED (safety check)",
        'CLEAN': "ALREADY OK",
    },
    'ANALYZE': {
        'FIXED_WARN': "WILL BE FIXED",
        'BLOCKED': "NEEDS YOUR DECISION",
        'CLEAN': "OK",
    },
    'VERIFY': {
        'FAILED': "STILL MIRRORED",
        'CLEAN': "OK",
    },
}
STATUS_ICON = {
    'FIXED': 'CHECKMARK',
    'FIXED_WARN': 'INFO',
    'BLOCKED': 'QUESTION',
    'FAILED': 'ERROR',
    'CLEAN': 'BLANK1',
}

# Last run, shown in the sidebar panel.
LAST_RUN = {"title": "", "summary": [], "rows": []}


class Entry:
    def __init__(self, obj):
        self.name = obj.name
        self.type = obj.type
        self.label = label(obj)
        self.path = hierarchy_path(obj)
        self.found = []
        self.done = []
        self.notes = []
        self.status = 'CLEAN'
        self.reason = ""
        self.hint = ""

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)


class Report:
    def __init__(self, kind, title, context):
        self.kind = kind
        self.title = title
        self.scene = context.scene.name
        self.frame = context.scene.frame_current
        self.entries = []

    def add(self, obj):
        entry = Entry(obj)
        self.entries.append(entry)
        return entry

    def counts(self):
        result = {key: 0 for key in STATUS_ORDER}
        for entry in self.entries:
            result[entry.status] += 1
        return result

    def summary(self):
        words = STATUS_WORDS[self.kind]
        counts = self.counts()
        return [(words[s].title(), counts[s]) for s in STATUS_ORDER if s in words]

    def lines(self):
        out = []
        bar = "=" * 72
        words = STATUS_WORDS[self.kind]
        out.append(bar)
        out.append(f"SAFE SCALE NORMALIZER {'.'.join(map(str, bl_info['version']))} - {self.title}")
        out.append(f"Scene '{self.scene}', frame {self.frame}")
        out.append(bar)
        out.append(
            f"Scanned {len(self.entries)} | "
            + " | ".join(f"{name.lower()}: {count}" for name, count in self.summary())
        )
        out.append("")

        for status in STATUS_ORDER:
            group = [e for e in self.entries if e.status == status]
            if not group:
                continue
            out.append(f"--- {words.get(status, status)} ({len(group)}) " + "-" * 30)
            for entry in group:
                out.append(f"  {entry.label}")
                if status == 'CLEAN' and not entry.found and not entry.notes:
                    continue
                out.append(f"      Path:  {entry.path}")
                for item in entry.found:
                    out.append(f"      Found: {item}")
                for item in entry.done:
                    out.append(f"      Done:  {item}" if self.kind == 'FIX' else f"      Plan:  {item}")
                if entry.reason:
                    out.append(f"      Why:   {entry.reason}")
                if entry.hint:
                    out.append(f"      Next:  {entry.hint}")
                for item in entry.notes:
                    out.append(f"      Note:  {item}")
            out.append("")
        return out

    def publish(self):
        lines = self.lines()
        for line in lines:
            log(line)

        text = bpy.data.texts.get(REPORT_TEXT_NAME)
        if text is None:
            text = bpy.data.texts.new(REPORT_TEXT_NAME)
        text.clear()
        text.write("\n".join(lines) + "\n")

        rows = []
        for entry in self.entries:
            if entry.status == 'CLEAN' or (entry.status == 'FIXED' and self.kind == 'FIX'):
                continue
            detail = entry.reason or (entry.notes[0] if entry.notes else "") \
                or (entry.found[0] if entry.found else "")
            rows.append((entry.name, entry.status, detail))
        LAST_RUN["title"] = self.title
        LAST_RUN["summary"] = self.summary()
        LAST_RUN["rows"] = rows


# ============================================================================
# HIERARCHY
# ============================================================================

def parent_handedness(obj):
    """Handedness of the matrix the parent contributes (before parent inverse)."""
    parent = obj.parent
    if parent is None or obj.parent_type in VERTEX_PARENT_TYPES:
        return 1

    result = handedness(parent.matrix_world) or 1

    if obj.parent_type == 'BONE' and parent.pose is not None:
        pose_bone = parent.pose.bones.get(obj.parent_bone)
        if pose_bone is not None:
            result *= handedness(pose_bone.matrix) or 1

    return result


def parent_inverse_needs_fix(obj):
    """True when the parent-inverse matrix adds a mirror the parent doesn't cancel."""
    if obj.parent is None:
        return False
    pi = handedness(obj.matrix_parent_inverse)
    if pi == 0:
        return False
    return pi != parent_handedness(obj)


def mirror_origin(obj):
    """Describe the ancestor (or bone) a world-space mirror comes from."""
    current = obj
    guard = 0
    while current.parent is not None and guard < 1024:
        guard += 1
        parent = current.parent
        if current.parent_type == 'BONE' and parent.pose is not None:
            pose_bone = parent.pose.bones.get(current.parent_bone)
            if pose_bone is not None and handedness(pose_bone.matrix) < 0:
                return f"pose bone '{current.parent_bone}' of {label(parent)}"
        if handedness(parent.matrix_world) < 0:
            current = parent
            continue
        break
    if current is obj:
        return None
    return label(current)


def collect_hierarchy(context, settings):
    if settings.scope == 'SCENE':
        base = list(context.view_layer.objects)
    else:
        base = list(context.selected_objects)

    collected = set(base)

    if settings.include_ancestors:
        for obj in base:
            parent = obj.parent
            while parent is not None and parent not in collected:
                collected.add(parent)
                parent = parent.parent

    for obj in list(collected):
        for child in obj.children_recursive:
            collected.add(child)

    return sorted(collected, key=lambda o: (depth(o), o.name))


# ============================================================================
# DIAGNOSIS
# ============================================================================

def negative_pose_bones(obj):
    result = []
    if obj.type != 'ARMATURE' or obj.pose is None:
        return result
    for pose_bone in obj.pose.bones:
        if any(v < -EPS for v in pose_bone.scale):
            result.append(pose_bone)
    return result


def skinned_objects(arm):
    result = []
    for obj in bpy.data.objects:
        for mod in getattr(obj, "modifiers", ()):
            if mod.type == 'ARMATURE' and mod.object == arm:
                result.append(obj)
                break
    return result


def describe_sources(obj):
    """Human readable list of every place a mirror comes from."""
    found = []

    neg_local = [i for i in range(3) if obj.scale[i] < -EPS]
    neg_delta = [i for i in range(3) if obj.delta_scale[i] < -EPS]

    if neg_local:
        found.append(
            f"negative local scale on {index_text(neg_local)}: scale = {vec_text(obj.scale)}"
        )
    if neg_delta:
        found.append(
            f"negative delta scale on {index_text(neg_delta)}: "
            f"delta_scale = {vec_text(obj.delta_scale)}"
        )

    if obj.parent is not None:
        pi = handedness(obj.matrix_parent_inverse)
        par = parent_handedness(obj)
        if pi != 0 and pi != par:
            if pi < 0:
                found.append("mirror hidden in the parent-inverse matrix")
            else:
                origin = mirror_origin(obj) or label(obj.parent)
                found.append(f"inherits a mirror from {origin}")

    if obj.type == 'ARMATURE':
        for pose_bone in negative_pose_bones(obj):
            found.append(
                f"pose bone '{pose_bone.name}' has negative scale {vec_text(pose_bone.scale)}"
            )

    return found


def protections(obj):
    """Things that make a change riskier; listed during analysis."""
    items = []
    ad = obj.animation_data
    if ad is not None:
        if ad.action is not None or len(ad.nla_tracks):
            items.append("animated (keys will be converted)")
        if len(ad.drivers):
            items.append(f"{len(ad.drivers)} driver(s)")
    if len(obj.constraints):
        names = ", ".join(f"'{c.name}' ({c.type})" for c in obj.constraints)
        items.append(f"constraints: {names}")
    if obj.library is not None:
        items.append(f"linked from library '{obj.library.filepath}'")
    data = getattr(obj, "data", None)
    if data is not None:
        users = [o.name for o in bpy.data.objects if o.data == data and o != obj]
        if users:
            items.append(f"data '{data.name}' shared with: {', '.join(users)}")
    if obj.type == 'ARMATURE':
        skinned = skinned_objects(obj)
        if skinned:
            items.append("deforms: " + ", ".join(f"'{o.name}'" for o in skinned))
        bone_children = [c for c in obj.children if c.parent_type == 'BONE']
        if bone_children:
            items.append(
                "bone children: "
                + ", ".join(f"'{c.name}' (bone '{c.parent_bone}')" for c in bone_children)
            )
    return items


# ============================================================================
# ANIMATION ACCESS (Blender 4.4+ slotted actions, legacy fallback)
# ============================================================================

def _slot_of(holder):
    return getattr(holder, "action_slot", None)


def action_fcurves(action, slot):
    if action is None:
        return []
    layers = getattr(action, "layers", None)
    if layers is not None and len(layers):
        if slot is None:
            return []
        result = []
        for layer in layers:
            for strip in layer.strips:
                getter = getattr(strip, "channelbag", None)
                if getter is None:
                    continue
                try:
                    bag = getter(slot)
                except (TypeError, RuntimeError):
                    bag = None
                if bag is not None:
                    result.extend(bag.fcurves)
        return result
    legacy = getattr(action, "fcurves", None)
    return list(legacy) if legacy is not None else []


def anim_holders(id_block):
    ad = getattr(id_block, "animation_data", None)
    if ad is None:
        return []
    holders = [ad] if ad.action is not None else []
    for track in ad.nla_tracks:
        for strip in track.strips:
            if strip.action is not None:
                holders.append(strip)
    return holders


def iter_anim_fcurves(id_block):
    seen = set()
    for holder in anim_holders(id_block):
        for fcurve in action_fcurves(holder.action, _slot_of(holder)):
            key = fcurve.as_pointer()
            if key not in seen:
                seen.add(key)
                yield fcurve


def iter_driver_fcurves(id_block):
    ad = getattr(id_block, "animation_data", None)
    if ad is not None:
        yield from ad.drivers


def _same_slot(a, b):
    if a is None or b is None:
        return a is None and b is None
    return getattr(a, "identifier", None) == getattr(b, "identifier", None)


def _action_shared(action, slot, owner):
    for other in bpy.data.objects:
        if other == owner:
            continue
        for holder in anim_holders(other):
            if holder.action == action and _same_slot(_slot_of(holder), slot):
                return True
    return False


def ensure_private_actions(obj, journal, entry):
    """Copy actions (per slot) that another object also plays."""
    for holder in anim_holders(obj):
        action = holder.action
        slot = _slot_of(holder)
        if not _action_shared(action, slot, obj):
            continue

        copy = action.copy()
        try:
            holder.action = copy
            if slot is not None and hasattr(holder, "action_slot"):
                for new_slot in copy.slots:
                    if new_slot.identifier == slot.identifier:
                        holder.action_slot = new_slot
                        break
        except (AttributeError, RuntimeError, TypeError) as exc:
            bpy.data.actions.remove(copy)
            raise Blocked(
                f"action '{action.name}' is shared and could not be duplicated ({exc})",
                "Exit NLA tweak mode or make the action single-user, then run again.",
            )

        def undo(h=holder, a=action, s=slot, c=copy):
            h.action = a
            if s is not None and hasattr(h, "action_slot"):
                h.action_slot = s
            if c.users == 0:
                bpy.data.actions.remove(c)

        journal.push(undo)
        entry.done.append(
            f"action '{action.name}' is shared with other objects; "
            f"'{obj.name}' now plays its own copy '{copy.name}'"
        )


def _key_arrays(fcurve):
    count = len(fcurve.keyframe_points)
    co = np.empty(count * 2, dtype=np.float32)
    left = np.empty(count * 2, dtype=np.float32)
    right = np.empty(count * 2, dtype=np.float32)
    if count:
        fcurve.keyframe_points.foreach_get("co", co)
        fcurve.keyframe_points.foreach_get("handle_left", left)
        fcurve.keyframe_points.foreach_get("handle_right", right)
    return co, left, right


def _set_key_arrays(fcurve, co, left, right):
    if len(fcurve.keyframe_points):
        fcurve.keyframe_points.foreach_set("co", co)
        fcurve.keyframe_points.foreach_set("handle_left", left)
        fcurve.keyframe_points.foreach_set("handle_right", right)
    fcurve.update()


def fcurve_name(fcurve):
    return f"{fcurve.data_path}[{fcurve.array_index}]"


def transform_fcurve_values(fcurve, factor, offset, journal):
    """value -> value * factor + offset for keys, handles and generators."""
    for mod in fcurve.modifiers:
        if mod.type in SAFE_FCURVE_MODIFIERS:
            continue
        if mod.type == 'GENERATOR' and offset == 0.0 and mod.mode == 'POLYNOMIAL':
            continue
        raise Blocked(
            f"F-curve {fcurve_name(fcurve)} has a {mod.type} modifier that can't be converted",
            "Apply or remove that F-curve modifier, then run the fix again.",
        )

    co, left, right = _key_arrays(fcurve)
    saved = (co.copy(), left.copy(), right.copy())
    for arr in (co, left, right):
        arr[1::2] = arr[1::2] * factor + offset
    _set_key_arrays(fcurve, co, left, right)

    generator_saves = []
    for mod in fcurve.modifiers:
        if mod.type == 'GENERATOR':
            old = tuple(mod.coefficients)
            generator_saves.append((mod, old))
            mod.coefficients = tuple(c * factor for c in old)

    def undo(f=fcurve, s=saved, g=generator_saves):
        _set_key_arrays(f, *s)
        for mod, old in g:
            mod.coefficients = old

    journal.push(undo)


def keys_sign_consistent(fcurve, expected_sign):
    co, _left, _right = _key_arrays(fcurve)
    values = co[1::2]
    if expected_sign < 0:
        return not np.any(values > EPS)
    return not np.any(values < -EPS)


def driver_expression(driver):
    names = [v.name for v in driver.variables]
    if driver.type == 'SCRIPTED':
        return driver.expression
    if not names:
        return "0.0"
    if driver.type == 'SUM':
        return " + ".join(names)
    if driver.type == 'AVERAGE':
        return f"({' + '.join(names)}) / {len(names)}"
    if driver.type == 'MIN':
        return f"min({', '.join(names)})" if len(names) > 1 else names[0]
    if driver.type == 'MAX':
        return f"max({', '.join(names)})" if len(names) > 1 else names[0]
    return driver.expression


def set_driver_expression(fcurve, expression, journal):
    driver = fcurve.driver
    old = (driver.type, driver.expression)
    driver.type = 'SCRIPTED'
    driver.expression = expression

    def undo(d=driver, o=old):
        d.type, d.expression = o

    journal.push(undo)


def negate_driver_output(fcurve, journal):
    if len(fcurve.keyframe_points) or len(fcurve.modifiers):
        transform_fcurve_values(fcurve, -1.0, 0.0, journal)
        return
    expression = driver_expression(fcurve.driver)
    set_driver_expression(fcurve, f"-({expression})", journal)


def negate_driver_variables(fcurve, names, journal):
    expression = driver_expression(fcurve.driver)
    for name in names:
        pattern = rf'(?<![\w.]){re.escape(name)}(?!\w)'
        expression = re.sub(pattern, f"(-{name})", expression)
    set_driver_expression(fcurve, expression, journal)


class AnimEditor:
    """Converts the animation of one ID; makes shared actions private first."""

    def __init__(self, id_block, journal, entry):
        self.id = id_block
        self.journal = journal
        self.entry = entry
        self._private = False

    def fcurves(self, path, index=None):
        def matches():
            return [
                fc for fc in iter_anim_fcurves(self.id)
                if fc.data_path == path and (index is None or fc.array_index == index)
            ]

        found = matches()
        if found and not self._private:
            ensure_private_actions(self.id, self.journal, self.entry)
            self._private = True
            found = matches()
        return found

    def all_fcurves(self):
        if not self._private and anim_holders(self.id):
            ensure_private_actions(self.id, self.journal, self.entry)
            self._private = True
        return list(iter_anim_fcurves(self.id))

    def drivers(self, path, index=None):
        return [
            fc for fc in iter_driver_fcurves(self.id)
            if fc.data_path == path and (index is None or fc.array_index == index)
        ]


# ============================================================================
# CHANNEL CONVERSIONS (SCALE / ROTATION) FOR OBJECTS AND POSE BONES
# ============================================================================

def negate_scale_channels(owner, attr, path, anim, journal):
    """Make every negative component of owner.<attr> positive, keys included."""
    values = getattr(owner, attr)
    flips = [i for i in range(3) if values[i] < -EPS]
    if not flips:
        return []

    for index in flips:
        for fcurve in anim.fcurves(path, index):
            if not keys_sign_consistent(fcurve, -1):
                raise Blocked(
                    f"{path}[{index}] is animated and changes sign over time "
                    "(an animated flip)",
                    "A flip that is animated can't be normalized statically. "
                    "Re-key it as rotation, or remove the sign change, then run again.",
                )
            transform_fcurve_values(fcurve, -1.0, 0.0, journal)
        for fcurve in anim.drivers(path, index):
            negate_driver_output(fcurve, journal)

    new = values.copy()
    for index in flips:
        new[index] = -new[index]
    journal.set(owner, attr, new)
    return flips


def apply_rotation_turn(owner, axis, prefix, anim, journal):
    """Right-multiply owner's rotation by 180 degrees about local `axis`.

    Exact for static values and for animated Euler / quaternion channels.
    """
    mode = owner.rotation_mode

    if mode == 'QUATERNION':
        path = prefix + "rotation_quaternion"
        if any(anim.drivers(path, i) for i in range(4)):
            raise Blocked(f"{path} is driven", "Remove the rotation driver or fix it by hand.")
        mapping = quaternion_turn_map(axis)
        curves = {fc.array_index: fc for fc in anim.fcurves(path)}
        for fcurve in curves.values():
            for new_index, (old_index, sgn) in mapping.items():
                if old_index == fcurve.array_index and sgn < 0:
                    transform_fcurve_values(fcurve, -1.0, 0.0, journal)
        moved = []
        for old_index, fcurve in curves.items():
            new_index = next(n for n, (o, _s) in mapping.items() if o == old_index)
            moved.append((fcurve, old_index, new_index))
            fcurve.array_index = 10 + new_index
        for fcurve, _old, new_index in moved:
            fcurve.array_index = new_index

        def undo_indices(items=moved):
            for fc, old, _new in items:
                fc.array_index = 10 + old
            for fc, old, _new in items:
                fc.array_index = old

        journal.push(undo_indices)
        turn = Quaternion((0.0,) + tuple(1.0 if i == axis else 0.0 for i in range(3)))
        journal.set(owner, "rotation_quaternion", owner.rotation_quaternion @ turn)

    elif mode == 'AXIS_ANGLE':
        path = prefix + "rotation_axis_angle"
        if anim.fcurves(path) or anim.drivers(path):
            raise Blocked(
                f"{path} is animated in Axis Angle mode",
                "Switch the rotation mode to Quaternion or Euler, then run again.",
            )
        angle, ax, ay, az = owner.rotation_axis_angle
        axis_vec = Vector((ax, ay, az))
        quat = Quaternion(axis_vec, angle) if axis_vec.length > EPS else Quaternion()
        turn = Quaternion((0.0,) + tuple(1.0 if i == axis else 0.0 for i in range(3)))
        new_axis, new_angle = (quat @ turn).to_axis_angle()
        journal.set(owner, "rotation_axis_angle", (new_angle, *new_axis))

    else:
        path = prefix + "rotation_euler"
        negate, offset_index = euler_turn_rule(mode, axis)
        if any(anim.drivers(path, i) for i in set(negate) | {offset_index}):
            raise Blocked(f"{path} is driven", "Remove the rotation driver or fix it by hand.")
        euler = owner.rotation_euler.copy()
        offset = -math.pi if euler[offset_index] > 0 else math.pi
        for index in negate:
            for fcurve in anim.fcurves(path, index):
                transform_fcurve_values(fcurve, -1.0, 0.0, journal)
            euler[index] = -euler[index]
        for fcurve in anim.fcurves(path, offset_index):
            transform_fcurve_values(fcurve, 1.0, offset, journal)
        euler[offset_index] += offset
        journal.set(owner, "rotation_euler", euler)


# ============================================================================
# PLANNING
# ============================================================================

class Plan:
    def __init__(self):
        self.strategy = None        # 'DATA', 'SIGN' or 'ROTATION'
        self.turn_axis = None       # 180 deg turn axis for ROTATION
        self.M = IDENTITY           # world change: W_new = W_old @ M
        self.pi_fix = False
        self.residual = None        # how a leftover mirror is handled
        self.convert_text = False
        self.warnings = []


def field_type(obj):
    field = getattr(obj, "field", None)
    return getattr(field, "type", 'NONE') or 'NONE'


def special_empty_reason(obj):
    if obj.instance_type == 'COLLECTION' and obj.instance_collection is not None:
        return f"instances collection '{obj.instance_collection.name}'"
    if obj.empty_display_type == 'IMAGE':
        return "is an image empty"
    return None


def strategy_for(obj):
    if obj.type in DATA_TYPES:
        return 'DATA'
    if obj.type == 'EMPTY':
        if special_empty_reason(obj) or field_type(obj) != 'NONE':
            return 'ROTATION'
        return 'SIGN'
    return 'ROTATION'


def make_plan(context, obj, settings):
    """Decide what to do. Returns None when the object needs nothing."""
    scale_signs = signs_of(obj.scale)
    delta_signs = signs_of(obj.delta_scale)
    total = mul_signs(scale_signs, delta_signs)

    has_scale_issue = -1 in scale_signs or -1 in delta_signs
    pi_fix = parent_inverse_needs_fix(obj)

    if not has_scale_issue and not pi_fix:
        return None

    plan = Plan()
    plan.pi_fix = pi_fix
    plan.strategy = strategy_for(obj)

    if obj.type == 'FONT' and settings.convert_text and is_mirror(total) != pi_fix:
        plan.convert_text = True
        plan.strategy = 'DATA'

    if plan.strategy in {'DATA', 'SIGN'}:
        plan.M = total
        if pi_fix:
            plan.M = mul_signs(plan.M, MIRROR_X)
    else:
        if is_mirror(total):
            remaining = MIRROR_X
            turn = mul_signs(total, MIRROR_X)
        else:
            remaining = IDENTITY
            turn = total
        plan.turn_axis = turn.index(1) if turn != IDENTITY else None
        plan.M = remaining
        if pi_fix:
            plan.M = mul_signs(plan.M, MIRROR_X)

    if plan.M == IDENTITY:
        return plan

    if plan.strategy == 'DATA':
        plan.residual = 'BAKE'
    elif plan.strategy == 'SIGN':
        plan.residual = 'CHILDREN'
    elif obj.type in SYMMETRIC_TYPES:
        plan.residual = 'DROP'
    elif obj.type == 'ARMATURE':
        if not settings.fix_armatures:
            raise Blocked(
                "the armature itself is mirrored (odd number of negative axes)",
                "Enable 'Mirror armature rest pose' to move the mirror into the bones.",
            )
        plan.residual = 'ARMATURE'
    elif obj.type == 'FONT':
        raise Blocked(
            "text is mirrored, and text data can't store a mirror",
            "Enable 'Convert mirrored text to mesh', or keep it as is if the mirror is intended.",
        )
    elif obj.type == 'EMPTY' and field_type(obj) in HANDED_FIELDS:
        plan.residual = 'DROP'
        plan.warnings.append(
            f"force field '{field_type(obj)}' acts in the opposite direction "
            "around its axis after removing the mirror; check the simulation"
        )
    elif obj.type == 'EMPTY' and field_type(obj) != 'NONE' and not special_empty_reason(obj):
        plan.residual = 'DROP'
    else:
        what = {
            'CAMERA': "the camera image is mirrored",
            'VOLUME': "the volume grid is mirrored",
        }.get(obj.type)
        if what is None and obj.type == 'EMPTY':
            what = f"the empty {special_empty_reason(obj)}, which is shown mirrored"
        if what is None:
            what = f"{obj.type.lower()} objects can't store a mirror in their data"
        if not settings.allow_visible_changes:
            raise Blocked(
                f"{what}; removing the mirror would visibly un-mirror it",
                "If the mirror is a mistake, enable 'Allow visible changes' and run again.",
            )
        plan.residual = 'DROP'
        plan.warnings.append(f"{what}; it now shows un-mirrored (allowed by option)")

    return plan


# ============================================================================
# DATA CHANGES
# ============================================================================

def ensure_private_data(obj, journal, entry):
    """Give obj its own copy of shared or linked data before changing it."""
    data = obj.data
    others = [o.name for o in bpy.data.objects if o.data == data and o != obj]
    if not others and data.library is None:
        return data

    copy = data.copy()
    try:
        obj.data = copy
    except (AttributeError, RuntimeError, TypeError) as exc:
        bpy.data.batch_remove(ids=[copy])
        raise Blocked(
            f"data '{data.name}' is shared or linked and can't be replaced ({exc})",
            "Make the object local / single-user first.",
        )

    def undo(o=obj, d=data, c=copy):
        o.data = d
        if c.users == 0:
            bpy.data.batch_remove(ids=[c])

    journal.push(undo)
    if others:
        entry.done.append(
            f"data '{data.name}' is shared with {', '.join(repr(n) for n in others)}; "
            f"'{obj.name}' now uses its own copy '{copy.name}' so they stay untouched"
        )
    else:
        entry.done.append(f"linked data '{data.name}' replaced by local copy '{copy.name}'")
    return copy


def _corner_normals(mesh):
    count = len(mesh.loops)
    values = np.empty(count * 3, dtype=np.float32)
    source = getattr(mesh, "corner_normals", None)
    if source is not None and len(source) == count:
        source.foreach_get("vector", values)
    else:
        mesh.calc_normals_split()
        mesh.loops.foreach_get("normal", values)
    return values.reshape(-1, 3)


def _flip_corner_map(mesh):
    """For each corner after a winding flip, the corner it came from."""
    count = len(mesh.polygons)
    starts = np.empty(count, dtype=np.int64)
    totals = np.empty(count, dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)
    corners = np.arange(len(mesh.loops), dtype=np.int64)
    owner = np.repeat(np.arange(count), totals)
    local = corners - starts[owner]
    return np.where(local == 0, corners, starts[owner] + totals[owner] - local)


def _flip_winding(mesh):
    if hasattr(mesh, "flip_normals"):
        mesh.flip_normals()
        return
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:], flip_multires=True)
    bm.to_mesh(mesh)
    bm.free()


def bake_mesh(mesh, signs, journal):
    matrix = sign_matrix4(signs)
    flip = is_mirror(signs)
    custom = _corner_normals(mesh) if mesh.has_custom_normals else None
    corner_map = _flip_corner_map(mesh) if (flip and custom is not None) else None

    def apply(normals_after):
        mesh.transform(matrix, shape_keys=True)
        if flip:
            _flip_winding(mesh)
        if normals_after is not None:
            mesh.normals_split_custom_set(normals_after.tolist())
        mesh.update()

    new_normals = None
    if custom is not None:
        source = custom[corner_map] if corner_map is not None else custom
        new_normals = source * np.array(signs, dtype=np.float32)

    apply(new_normals)
    journal.push(lambda: apply(custom))
    return flip


def bake_curve(curve, signs, journal):
    matrix = sign_matrix4(signs)
    flip = is_mirror(signs)

    def tilts(negate):
        for spline in curve.splines:
            for point in list(spline.bezier_points) + list(spline.points):
                if negate:
                    point.tilt = -point.tilt

    def apply():
        curve.transform(matrix, shape_keys=True)
        if flip:
            tilts(True)
        curve.update_tag()

    apply()
    journal.push(apply)  # sign changes are their own inverse


def bake_lattice(lattice, signs, journal):
    nu, nv, nw = lattice.points_u, lattice.points_v, lattice.points_w
    count = nu * nv * nw
    grid = np.arange(count).reshape(nw, nv, nu)
    if signs[0] < 0:
        grid = grid[:, :, ::-1]
    if signs[1] < 0:
        grid = grid[:, ::-1, :]
    if signs[2] < 0:
        grid = grid[::-1, :, :]
    source = grid.reshape(-1)  # new index -> old index

    rest = np.empty(count * 3, dtype=np.float32)
    lattice.points.foreach_get("co", rest)
    rest = rest.reshape(-1, 3)
    if not np.allclose(rest[source] * np.array(signs), rest, atol=1e-4):
        raise Blocked(
            "lattice grid is not symmetric, so the mirror can't be moved into it",
            "Reset the lattice resolution/offsets or apply the scale by hand.",
        )

    for point in lattice.points:
        if len(point.groups):
            raise Blocked(
                "lattice has vertex groups whose order can't be mirrored",
                "Remove the lattice vertex groups, or fix this lattice by hand.",
            )

    sign_arr = np.array(signs, dtype=np.float32)

    def read(points, attr, size):
        arr = np.empty(count * size, dtype=np.float32)
        points.foreach_get(attr, arr)
        return arr

    saved_co = read(lattice.points, "co_deform", 3)
    saved_w = read(lattice.points, "weight_softbody", 1)
    keys = []
    if lattice.shape_keys is not None:
        for block in lattice.shape_keys.key_blocks:
            keys.append((block, read(block.data, "co", 3)))

    co = (saved_co.reshape(-1, 3)[source] * sign_arr).reshape(-1)
    lattice.points.foreach_set("co_deform", co)
    lattice.points.foreach_set("weight_softbody", saved_w[source])
    for block, arr in keys:
        block.data.foreach_set("co", (arr.reshape(-1, 3)[source] * sign_arr).reshape(-1))
    lattice.update_tag()

    def undo():
        lattice.points.foreach_set("co_deform", saved_co)
        lattice.points.foreach_set("weight_softbody", saved_w)
        for block, arr in keys:
            block.data.foreach_set("co", arr)
        lattice.update_tag()

    journal.push(undo)


def bake_meta(meta, signs, journal):
    m3 = sign_matrix4(signs).to_3x3()
    flip = is_mirror(signs)
    saved = [(el, el.co.copy(), el.rotation.copy()) for el in meta.elements]
    for element, co, rotation in saved:
        element.co = m3 @ co
        rot = m3 @ rotation.to_matrix()
        if flip:
            rot = rot @ FX3
        element.rotation = rot.to_quaternion()
    meta.update_tag()

    def undo():
        for element, co, rotation in saved:
            element.co = co
            element.rotation = rotation
        meta.update_tag()

    journal.push(undo)


def _flip_vector_attributes(attributes, signs, names=("position", "handle_left", "handle_right")):
    sign_arr = np.array(signs, dtype=np.float32)
    for name in names:
        attribute = attributes.get(name)
        if attribute is None or attribute.data_type != 'FLOAT_VECTOR':
            continue
        arr = np.empty(len(attribute.data) * 3, dtype=np.float32)
        attribute.data.foreach_get("vector", arr)
        attribute.data.foreach_set("vector", (arr.reshape(-1, 3) * sign_arr).reshape(-1))


def bake_points(data, signs, journal):
    def apply():
        _flip_vector_attributes(data.attributes, signs)
        data.update_tag()

    apply()
    journal.push(apply)


def bake_grease_pencil(gp, signs, journal):
    m3 = sign_matrix4(signs).to_3x3()
    saved_layers = []
    for layer in gp.layers:
        if getattr(layer, "parent", None) is not None:
            raise Blocked(
                f"grease pencil layer '{layer.name}' has its own parent",
                "Clear the layer parent or fix this object by hand.",
            )
        saved_layers.append(
            (layer, layer.translation.copy(), layer.rotation.copy(), layer.scale.copy())
        )

    drawings = []
    seen = set()
    for layer in gp.layers:
        for frame in layer.frames:
            drawing = getattr(frame, "drawing", None)
            if drawing is None or drawing.as_pointer() in seen:
                continue
            seen.add(drawing.as_pointer())
            drawings.append(drawing)

    def apply_drawings():
        for drawing in drawings:
            _flip_vector_attributes(drawing.attributes, signs)

    for layer, translation, rotation, scale in saved_layers:
        layer.translation = m3 @ translation
        rot = m3 @ rotation.to_matrix() @ m3
        layer.rotation = rot.to_euler(rotation.order, rotation)
    apply_drawings()
    gp.update_tag()

    def undo():
        apply_drawings()
        for layer, translation, rotation, scale in saved_layers:
            layer.translation = translation
            layer.rotation = rotation
            layer.scale = scale
        gp.update_tag()

    journal.push(undo)


def adjust_modifiers(obj, signs, journal, entry):
    """Modifiers that work along local axes get the same mirror."""
    matrix = sign_matrix4(signs)
    sign_vec = Vector(signs)
    for mod in obj.modifiers:
        if mod.type == 'ARRAY':
            journal.set(mod, "relative_offset_displace",
                        mod.relative_offset_displace * sign_vec)
            journal.set(mod, "constant_offset_displace",
                        mod.constant_offset_displace * sign_vec)
            entry.done.append(f"Array modifier '{mod.name}': offsets mirrored to match")
        elif mod.type == 'MIRROR':
            flips = list(mod.use_bisect_flip_axis)
            changed = False
            for i in range(3):
                if signs[i] < 0 and mod.use_axis[i] and mod.use_bisect_axis[i]:
                    flips[i] = not flips[i]
                    changed = True
            if changed:
                journal.set(mod, "use_bisect_flip_axis", flips)
                entry.done.append(f"Mirror modifier '{mod.name}': bisect side flipped to match")
        elif mod.type == 'HOOK':
            journal.set(mod, "matrix_inverse", mod.matrix_inverse @ matrix)
            journal.set(mod, "center", Vector(mod.center) * sign_vec)
        elif mod.type == 'SCREW':
            axis = "XYZ".index(mod.axis)
            factor = (-1 if is_mirror(signs) else 1) * signs[axis]
            if factor < 0:
                journal.set(mod, "angle", -mod.angle)
            if signs[axis] < 0:
                journal.set(mod, "screw_offset", -mod.screw_offset)


def hair_particle_names(obj):
    names = []
    for system in getattr(obj, "particle_systems", ()):
        if system.settings.type == 'HAIR':
            names.append(system.name)
    return names


def cached_physics(obj):
    kinds = {'CLOTH', 'SOFT_BODY', 'DYNAMIC_PAINT', 'FLUID', 'PARTICLE_SYSTEM'}
    return [mod.name for mod in obj.modifiers if mod.type in kinds]


def bake_data(context, obj, signs, journal, entry):
    hair = hair_particle_names(obj)
    if hair:
        raise Blocked(
            f"hair particle system(s) {', '.join(hair)} would detach from the mirrored mesh",
            "Convert the particle hair to Curves, or fix this object by hand.",
        )

    data = ensure_private_data(obj, journal, entry)
    what = f"{axes_text(signs)} mirror"

    if obj.type == 'MESH':
        flipped = bake_mesh(data, signs, journal)
        entry.done.append(
            f"baked {what} into mesh '{data.name}'"
            + (" and flipped the face winding so normals still point out" if flipped else "")
            + (" (custom normals preserved)" if data.has_custom_normals else "")
            + (" (shape keys converted)" if data.shape_keys else "")
        )
    elif obj.type in {'CURVE', 'SURFACE'}:
        bake_curve(data, signs, journal)
        entry.done.append(f"baked {what} into curve data '{data.name}'")
    elif obj.type == 'LATTICE':
        bake_lattice(data, signs, journal)
        entry.done.append(f"baked {what} into lattice '{data.name}' (points re-ordered)")
    elif obj.type == 'META':
        bake_meta(data, signs, journal)
        entry.done.append(f"baked {what} into metaball '{data.name}'")
    elif obj.type in {'POINTCLOUD', 'CURVES'}:
        bake_points(data, signs, journal)
        entry.done.append(f"baked {what} into {obj.type.lower()} '{data.name}'")
    elif obj.type == 'GREASEPENCIL':
        bake_grease_pencil(data, signs, journal)
        entry.done.append(f"baked {what} into grease pencil '{data.name}'")
    else:
        raise Blocked(f"no data handler for {obj.type}")

    adjust_modifiers(obj, signs, journal, entry)

    physics = cached_physics(obj)
    if physics:
        entry.note(f"re-bake physics/particle caches ({', '.join(physics)})")


def convert_text_to_mesh(context, obj, entry):
    with context.temp_override(
        object=obj, active_object=obj,
        selected_objects=[obj], selected_editable_objects=[obj],
    ):
        result = bpy.ops.object.convert(target='MESH')
    if 'FINISHED' not in result or obj.type != 'MESH':
        raise Blocked("text could not be converted to mesh")
    entry.done.append("converted the mirrored text to a mesh (option enabled)")


# ============================================================================
# ARMATURE REST-POSE MIRROR
# ============================================================================

@contextmanager
def edit_mode(context, obj):
    """Enter Edit Mode on obj only, then restore selection and visibility."""
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = [o for o in view_layer.objects if o.select_get()]
    hidden = obj.hide_get()
    disabled = obj.hide_viewport

    for other in previous_selected:
        other.select_set(False)
    obj.hide_viewport = False
    obj.hide_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj

    try:
        bpy.ops.object.mode_set(mode='EDIT')
    except RuntimeError as exc:
        _restore_selection(view_layer, obj, previous_active, previous_selected, hidden, disabled)
        raise Blocked(f"couldn't enter Edit Mode on {label(obj)} ({exc})",
                      "Make sure its collection is visible and enabled.")
    try:
        yield
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        _restore_selection(view_layer, obj, previous_active, previous_selected, hidden, disabled)


def _restore_selection(view_layer, obj, active, selected, hidden, disabled):
    obj.select_set(False)
    for other in selected:
        try:
            other.select_set(True)
        except RuntimeError:
            pass
    view_layer.objects.active = active
    obj.hide_set(hidden)
    obj.hide_viewport = disabled


EDIT_BONE_X_PROPS = ("bbone_curveinx", "bbone_curveoutx", "bbone_rollin", "bbone_rollout")


def mirror_rest_bones(context, obj, journal):
    arm = obj.data
    saved = {}
    with edit_mode(context, obj):
        for bone in arm.edit_bones:
            saved[bone.name] = (
                bone.head.copy(), bone.tail.copy(), bone.roll,
                tuple(getattr(bone, p) for p in EDIT_BONE_X_PROPS),
            )
        # Targets come from the saved rest pose: moving a parent's tail also
        # drags a connected child's head, so live values can't be trusted.
        targets = {
            bone.name: (FX3 @ bone.head, FX3 @ bone.tail, FX3 @ bone.z_axis)
            for bone in arm.edit_bones
        }
        for bone in arm.edit_bones:
            head, tail, _z = targets[bone.name]
            bone.head = head
            bone.tail = tail
        for bone in arm.edit_bones:
            head, tail, z_axis = targets[bone.name]
            bone.head = head
            bone.tail = tail
            bone.align_roll(z_axis)
            for prop in EDIT_BONE_X_PROPS:
                setattr(bone, prop, -getattr(bone, prop))

    def undo(o=obj, s=saved):
        with edit_mode(bpy.context, o):
            for _ in range(2):  # twice: connected bones drag each other
                for bone in o.data.edit_bones:
                    if bone.name not in s:
                        continue
                    head, tail, roll, extra = s[bone.name]
                    bone.head, bone.tail, bone.roll = head, tail, roll
                    for prop, value in zip(EDIT_BONE_X_PROPS, extra):
                        setattr(bone, prop, value)

    journal.push(undo)


def mirror_pose_channels(obj, journal):
    for pose_bone in obj.pose.bones:
        location = pose_bone.location.copy()
        location.x = -location.x
        journal.set(pose_bone, "location", location)

        quat = pose_bone.rotation_quaternion.copy()
        quat.y, quat.z = -quat.y, -quat.z
        journal.set(pose_bone, "rotation_quaternion", quat)

        euler = pose_bone.rotation_euler.copy()
        euler.y, euler.z = -euler.y, -euler.z
        journal.set(pose_bone, "rotation_euler", euler)

        axis_angle = list(pose_bone.rotation_axis_angle)
        axis_angle[2], axis_angle[3] = -axis_angle[2], -axis_angle[3]
        journal.set(pose_bone, "rotation_axis_angle", axis_angle)

        for prop in EDIT_BONE_X_PROPS:
            journal.set(pose_bone, prop, -getattr(pose_bone, prop))

        # Keep custom bone shapes drawing exactly as before.
        if pose_bone.custom_shape is not None:
            translation = pose_bone.custom_shape_translation.copy()
            translation.x = -translation.x
            journal.set(pose_bone, "custom_shape_translation", translation)
            shape_rot = pose_bone.custom_shape_rotation_euler.copy()
            shape_rot.y, shape_rot.z = -shape_rot.y, -shape_rot.z
            journal.set(pose_bone, "custom_shape_rotation_euler", shape_rot)
            shape_scale = pose_bone.custom_shape_scale_xyz.copy()
            shape_scale.x = -shape_scale.x
            journal.set(pose_bone, "custom_shape_scale_xyz", shape_scale)


LOCAL_SPACES = {'LOCAL', 'LOCAL_WITH_PARENT', 'POSE', 'LOCAL_OWNER_ORIENT'}


def mirror_bone_constraints(obj, journal, entry):
    for pose_bone in obj.pose.bones:
        for con in pose_bone.constraints:
            local = getattr(con, "owner_space", 'WORLD') in LOCAL_SPACES
            if con.type == 'LIMIT_ROTATION' and local:
                for axis in ("y", "z"):
                    lo, hi = getattr(con, f"min_{axis}"), getattr(con, f"max_{axis}")
                    journal.set(con, f"min_{axis}", -hi)
                    journal.set(con, f"max_{axis}", -lo)
            elif con.type == 'LIMIT_LOCATION' and local:
                lo, hi = con.min_x, con.max_x
                use_lo, use_hi = con.use_min_x, con.use_max_x
                journal.set(con, "min_x", -hi)
                journal.set(con, "max_x", -lo)
                journal.set(con, "use_min_x", use_hi)
                journal.set(con, "use_max_x", use_lo)
            elif con.type == 'IK' and con.pole_target is not None:
                # The pole angle is measured from the bone's X axis, which flips.
                journal.set(con, "pole_angle", math.remainder(math.pi - con.pole_angle, 2 * math.pi))


def mirror_pose_animation(obj, journal, entry):
    anim = AnimEditor(obj, journal, entry)
    for fcurve in anim.all_fcurves():
        match = BONE_PATH_RE.match(fcurve.data_path)
        if match and fcurve.array_index in POSE_MIRROR_CHANNELS.get(match.group(2), ()):
            transform_fcurve_values(fcurve, -1.0, 0.0, journal)
    for fcurve in iter_driver_fcurves(obj):
        match = BONE_PATH_RE.match(fcurve.data_path)
        if match and fcurve.array_index in POSE_MIRROR_CHANNELS.get(match.group(2), ()):
            negate_driver_output(fcurve, journal)


DRIVER_ID_COLLECTIONS = (
    "objects", "meshes", "shape_keys", "armatures", "materials", "node_groups",
    "curves", "lattices", "lights", "cameras", "worlds", "scenes", "textures",
    "grease_pencils", "hair_curves", "pointclouds", "volumes", "metaballs",
)


def _bone_reading_variable(var, arm, bones):
    """True when a driver variable reads a value the mirror negates."""
    for target in var.targets:
        if target.id != arm:
            continue
        if var.type == 'TRANSFORMS':
            if target.bone_target not in bones:
                continue
            if target.transform_space not in {'LOCAL_SPACE', 'TRANSFORM_SPACE'}:
                continue
            mode = getattr(target, "rotation_mode", 'AUTO')
            if mode.startswith('SWING') or mode.startswith('TWIST'):
                raise Blocked(
                    f"driver variable '{var.name}' reads swing/twist rotation of bone "
                    f"'{target.bone_target}'",
                    "Switch that variable to Euler/Quaternion mode, then run again.",
                )
            if target.transform_type in {'LOC_X', 'ROT_Y', 'ROT_Z'}:
                return True
        elif var.type == 'SINGLE_PROP':
            match = BONE_PROP_PATH_RE.match(target.data_path)
            if match and match.group(1) in bones:
                index = int(match.group(3)) if match.group(3) else 0
                if index in POSE_MIRROR_CHANNELS.get(match.group(2), ()):
                    return True
    return False


def mirror_bone_reading_drivers(arm, journal, entry):
    bones = {b.name for b in arm.data.bones}
    changed = 0
    for collection_name in DRIVER_ID_COLLECTIONS:
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue
        for id_block in collection:
            for fcurve in iter_driver_fcurves(id_block):
                names = [
                    var.name for var in fcurve.driver.variables
                    if _bone_reading_variable(var, arm, bones)
                ]
                if names:
                    negate_driver_variables(fcurve, names, journal)
                    changed += 1
    if changed:
        entry.done.append(f"{changed} driver(s) that read bone rotation/location adjusted")


def fix_armature_mirror(context, obj, journal, entry):
    ensure_private_data(obj, journal, entry)
    mirror_rest_bones(context, obj, journal)
    mirror_pose_channels(obj, journal)
    mirror_bone_constraints(obj, journal, entry)
    mirror_pose_animation(obj, journal, entry)
    mirror_bone_reading_drivers(obj, journal, entry)
    entry.done.append(
        f"moved the mirror into the rest pose of {len(obj.data.bones)} bone(s); "
        "pose, keys, B-bones, bone shapes and limits converted to match"
    )

    sided = [b.name for b in obj.data.bones if SIDE_SUFFIX_RE.search(b.name)]
    if sided:
        entry.note(
            f"{len(sided)} bones use .L/.R names (e.g. '{sided[0]}'); they now sit on the "
            "opposite local side, so X-Axis Mirror editing and Paste Flipped Pose work "
            "reversed for this rig"
        )


def fix_negative_pose_bones(obj, settings, journal, entry):
    """Even-count negative pose scales are rotations; odd ones are reported."""
    if not settings.fix_pose_bones or obj.pose is None:
        return
    anim = AnimEditor(obj, journal, entry)
    for pose_bone in obj.pose.bones:
        signs = signs_of(pose_bone.scale)
        if -1 not in signs:
            continue
        prefix = f'pose.bones["{bpy.utils.escape_identifier(pose_bone.name)}"].'
        if is_mirror(signs):
            entry.note(
                f"pose bone '{pose_bone.name}' is mirrored in the current pose "
                f"{vec_text(pose_bone.scale)}; a mirror inside a pose can't be removed "
                "without changing the animation"
            )
            continue
        negate_scale_channels(pose_bone, "scale", prefix + "scale", anim, journal)
        apply_rotation_turn(pose_bone, signs.index(1), prefix, anim, journal)
        entry.done.append(
            f"pose bone '{pose_bone.name}': negative scale on {axes_text(signs)} "
            f"replaced by a 180 degree turn about {AXES[signs.index(1)]}"
        )


# ============================================================================
# CHILDREN AND OTHER DEPENDENTS
# ============================================================================

def compensate_children(obj, change, journal):
    """Keep every child's world transform after W_parent -> W_parent @ change."""
    if matrix_close(change, Matrix.Identity(4), 1.0e-9):
        return 0
    count = 0
    for child in obj.children:
        if child.parent_type in VERTEX_PARENT_TYPES and obj.type in DATA_TYPES:
            continue  # vertex positions in world space did not move
        journal.set(child, "matrix_parent_inverse", change @ child.matrix_parent_inverse)
        count += 1

    for other in bpy.data.objects:
        owners = [other.constraints]
        if other.pose is not None:
            owners += [pb.constraints for pb in other.pose.bones]
        for constraints in owners:
            for con in constraints:
                if con.type == 'CHILD_OF' and con.target == obj:
                    journal.set(con, "inverse_matrix", change @ con.inverse_matrix)
                    count += 1
    return count


def _object_pointers(mod):
    """Objects a modifier points at (regular pointers and Geometry Nodes inputs)."""
    found = []
    if mod.type == 'NODES':
        for key in mod.keys():
            try:
                value = mod[key]
            except (KeyError, TypeError):
                continue
            if isinstance(value, bpy.types.Object):
                found.append(value)
    for prop in mod.bl_rna.properties:
        if prop.type != 'POINTER':
            continue
        fixed = getattr(prop, "fixed_type", None)
        if fixed is None or fixed.identifier != 'Object':
            continue
        value = getattr(mod, prop.identifier, None)
        if value is not None:
            found.append(value)
    return found


def build_dependents(context):
    """target pointer -> objects whose modifiers use that target (built once per run)."""
    index = {}
    for obj in context.view_layer.objects:
        seen = set()
        for mod in getattr(obj, "modifiers", ()):
            for target in _object_pointers(mod):
                key = target.as_pointer()
                if key not in seen:
                    seen.add(key)
                    index.setdefault(key, []).append(obj)
    return index


def dependent_objects(context, target):
    """Objects whose evaluated geometry depends on target."""
    index = ACTIVE.get("dependents")
    if index is None:
        index = build_dependents(context)
    return [o for o in index.get(target.as_pointer(), []) if o != target]


# ============================================================================
# VERIFICATION
# ============================================================================

class GeometrySample:
    __slots__ = ("points", "centers", "normals", "extent")


def _world(points, world):
    return points @ world[:3, :3].T + world[:3, 3]


def sample_geometry(depsgraph, obj):
    evaluated = obj.evaluated_get(depsgraph)
    try:
        mesh = evaluated.to_mesh()
    except (RuntimeError, AttributeError, TypeError):
        return None
    if mesh is None:
        return None
    try:
        count = len(mesh.vertices)
        local = np.empty(count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", local)
        local = local.reshape(-1, 3).astype(np.float64)

        poly_count = len(mesh.polygons)
        normals = np.empty(poly_count * 3, dtype=np.float32)
        centers = np.empty(poly_count * 3, dtype=np.float32)
        areas = np.empty(poly_count, dtype=np.float32)
        if poly_count:
            source = getattr(mesh, "polygon_normals", None)
            if source is not None and len(source) == poly_count:
                source.foreach_get("vector", normals)
            else:
                mesh.polygons.foreach_get("normal", normals)
            mesh.polygons.foreach_get("center", centers)
            mesh.polygons.foreach_get("area", areas)
        normals = normals.reshape(-1, 3).astype(np.float64)
        centers = centers.reshape(-1, 3).astype(np.float64)
        keep = areas >= MIN_FACE_AREA
    finally:
        evaluated.to_mesh_clear()

    world = np.array(evaluated.matrix_world, dtype=np.float64)
    sample = GeometrySample()
    sample.points = _world(local, world)
    sample.centers = _world(centers[keep], world)
    if sample.points.size:
        span = sample.points.max(axis=0) - sample.points.min(axis=0)
        sample.extent = float(np.linalg.norm(span))
    else:
        sample.extent = 0.0

    basis = world[:3, :3]
    sample.normals = None
    if poly_count and abs(np.linalg.det(basis)) > EPS:
        world_normals = normals[keep] @ np.linalg.inv(basis)
        lengths = np.linalg.norm(world_normals, axis=1)
        lengths[lengths < EPS] = 1.0
        sample.normals = world_normals / lengths[:, None]
    return sample


def _nearest(reference, queries):
    """Index of the nearest reference point and its distance, per query."""
    from mathutils.kdtree import KDTree
    tree = KDTree(len(reference))
    for index, point in enumerate(reference):
        tree.insert(point, index)
    tree.balance()
    found = [tree.find(point) for point in queries]
    return (
        np.array([f[1] for f in found], dtype=np.int64),
        np.array([f[2] for f in found], dtype=np.float64),
    )


def compare_geometry(obj, before, after):
    """None when identical in world space, else a short description."""
    if before is None:
        return None
    if after is None:
        return f"{label(obj)} no longer produces geometry"
    tol = ACTIVE["tolerance"] * max(1.0, before.extent)

    if obj.type == 'META':
        # Metaball surfaces are re-tessellated on a local grid; compare shape.
        tol = max(tol, obj.data.resolution * 2.0)
        if before.points.size and after.points.size:
            _i, d1 = _nearest(before.points, after.points)
            _i, d2 = _nearest(after.points, before.points)
            error = max(d1.max(), d2.max())
            if error > tol:
                return f"{label(obj)} metaball surface moved by {error:.4f}"
        return None

    if before.points.shape != after.points.shape:
        return (
            f"{label(obj)} evaluated vertex count changed "
            f"({len(before.points)} -> {len(after.points)})"
        )
    if not before.points.size:
        return None

    same_order = float(np.abs(before.points - after.points).max()) <= tol
    if not same_order:
        # Some modifiers (Mirror, Array merge...) legitimately re-order vertices.
        _i, d1 = _nearest(before.points, after.points)
        _i, d2 = _nearest(after.points, before.points)
        error = max(d1.max(), d2.max())
        if error > tol:
            return f"{label(obj)} vertices moved by up to {error:.5f} (allowed {tol:.5f})"

    if before.normals is None or after.normals is None:
        return None
    if before.normals.shape != after.normals.shape or not len(before.normals):
        return None

    if same_order:
        matched = before.normals
    else:
        index, dist = _nearest(before.centers, after.centers)
        if dist.size and dist.max() > tol:
            return f"{label(obj)} faces moved by up to {dist.max():.5f}"
        matched = before.normals[index]

    flipped = int(((matched * after.normals).sum(axis=1) < 0.0).sum())
    if flipped:
        return f"{label(obj)}: {flipped} face(s) would show inverted normals"
    return None


def bone_points(obj):
    if obj is None or obj.type != 'ARMATURE' or obj.pose is None:
        return {}
    world = obj.matrix_world
    return {
        pb.name: (world @ pb.head, world @ pb.tail)
        for pb in obj.pose.bones
    }


def _bulk_worlds(context):
    objects = list(context.view_layer.objects)
    values = np.empty(len(objects) * 16, dtype=np.float32)
    try:
        context.view_layer.objects.foreach_get("matrix_world", values)
    except (AttributeError, RuntimeError, TypeError):
        for index, obj in enumerate(objects):
            values[index * 16:(index + 1) * 16] = [v for col in obj.matrix_world.col for v in col]
    return objects, values.reshape(-1, 16).astype(np.float64)


class Snapshot:
    def __init__(self, context, target, geometry_objects, armature=None):
        context.view_layer.update()
        self.objects, self.worlds = _bulk_worlds(context)
        self.target_world = target.matrix_world.copy()
        depsgraph = context.evaluated_depsgraph_get()
        self.geometry = {o: sample_geometry(depsgraph, o) for o in geometry_objects}
        self.armature = armature
        self.bones = bone_points(armature)


def verify(context, snapshot, target, expected_world):
    context.view_layer.update()
    problems = []

    if expected_world is not None and not matrix_close(target.matrix_world, expected_world):
        cons = ", ".join(f"'{c.name}'" for c in target.constraints if c.enabled)
        problems.append(
            f"{label(target)} did not end up where expected"
            + (f" (constraints {cons} override its transform)" if cons else "")
        )

    objects, worlds = _bulk_worlds(context)
    if len(objects) == len(snapshot.objects):
        before = snapshot.worlds
        scale = np.maximum(1.0, np.maximum(np.abs(before).max(axis=1), np.abs(worlds).max(axis=1)))
        moved = np.abs(before - worlds).max(axis=1) > MATRIX_TOL * scale
        for index in np.nonzero(moved)[0]:
            obj = objects[index]
            if obj != target:
                problems.append(f"{label(obj)} moved")
    else:
        problems.append("objects were added or removed during the change")

    depsgraph = context.evaluated_depsgraph_get()
    for obj, before in snapshot.geometry.items():
        problem = compare_geometry(obj, before, sample_geometry(depsgraph, obj))
        if problem:
            problems.append(problem)

    if snapshot.bones:
        after = bone_points(snapshot.armature)
        tol = ACTIVE["tolerance"] * max(1.0, max(
            (h.length for h, _t in snapshot.bones.values()), default=1.0))
        for name, (head, tail) in snapshot.bones.items():
            if name not in after:
                problems.append(f"bone '{name}' disappeared")
                continue
            new_head, new_tail = after[name]
            if (new_head - head).length > tol or (new_tail - tail).length > tol:
                problems.append(f"bone '{name}' moved")

    if problems:
        extra = f" (+{len(problems) - 4} more)" if len(problems) > 4 else ""
        raise VerifyFailed("; ".join(problems[:4]) + extra)


# ============================================================================
# FIX ONE OBJECT
# ============================================================================

def precheck(context, obj):
    if obj.library is not None:
        raise Blocked(
            f"object is linked from library '{obj.library.filepath}'",
            "Use Object > Library Override > Make, or Make Local, then run again.",
        )
    if not in_view_layer(context, obj):
        raise Blocked(
            "object is not in the active view layer, so the result can't be verified",
            "Enable its collection in this view layer, then run again.",
        )


def describe_plan(obj, plan):
    lines = []
    if plan.convert_text:
        lines.append("convert text to mesh, then bake the mirror into the mesh")
    if plan.strategy == 'DATA' and plan.M != IDENTITY:
        lines.append(f"bake {axes_text(plan.M)} mirror into the object data "
                     "(normals kept outward, rotation unchanged)")
    if plan.strategy == 'SIGN':
        lines.append("make scale positive; children are compensated")
    if plan.turn_axis is not None:
        lines.append(f"replace two negative axes with a 180 degree turn about "
                     f"{AXES[plan.turn_axis]} (data untouched)")
    if plan.residual == 'DROP':
        lines.append("drop the remaining X mirror (invisible for this object type)")
    if plan.residual == 'ARMATURE':
        lines.append("move the remaining mirror into the rest pose (bones, pose, keys)")
    if plan.pi_fix:
        lines.append("remove the mirror from the parent-inverse matrix")
    return lines


def has_active_constraints(obj):
    return any(getattr(c, "enabled", True) and c.influence > 0.0 for c in obj.constraints)


def fix_object(context, obj, settings, entry):
    """Fix one object. Returns True if something changed."""
    precheck(context, obj)

    plan = make_plan(context, obj, settings)
    if plan is None:
        if obj.type == 'ARMATURE' and negative_pose_bones(obj):
            plan = Plan()
            plan.strategy = 'NONE'
        else:
            return False

    if plan.convert_text:
        # Conversion can't be undone by the journal; it keeps the look
        # identical, so it is safe to do once, up front.
        convert_text_to_mesh(context, obj, entry)

    geometry = []
    if settings.verify_geometry:
        if obj.type not in {'EMPTY', 'LIGHT', 'CAMERA', 'SPEAKER', 'LIGHT_PROBE', 'ARMATURE'}:
            geometry.append(obj)
        geometry += dependent_objects(context, obj)
    snapshot = Snapshot(context, obj, geometry, armature=obj if obj.type == 'ARMATURE' else None)

    constrained = has_active_constraints(obj)

    # IK chains are re-solved iteratively, so a rig change can only match to
    # solver precision. Allow 0.1 % there (still far below anything visible).
    user_tolerance = ACTIVE["tolerance"]
    if plan.residual == 'ARMATURE' and any(
        con.type in {'IK', 'SPLINE_IK'}
        for pose_bone in obj.pose.bones for con in pose_bone.constraints
    ):
        ACTIVE["tolerance"] = max(user_tolerance, IK_TOLERANCE)
        entry.note("rig has IK: checked to IK-solver precision (0.1% of size)")

    # Which mirror goes into the data. A constraint that rebuilds rotation
    # (Track To, Copy Rotation...) decides on its own how a negative scale is
    # shown, so for constrained objects every mirror choice is tried in turn
    # and the one that verifies wins.
    bakes = [plan.M]
    if constrained and plan.strategy == 'DATA' and plan.M != IDENTITY:
        all_signs = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        bakes += [b for b in all_signs if b != plan.M and b != IDENTITY]

    try:
        return _try_bakes(context, obj, plan, bakes, settings, entry, snapshot, constrained)
    finally:
        ACTIVE["tolerance"] = user_tolerance


def _try_bakes(context, obj, plan, bakes, settings, entry, snapshot, constrained):
    first_failure = None
    for bake in bakes:
        done_mark, notes_mark = len(entry.done), len(entry.notes)
        try:
            _attempt(context, obj, plan, bake, settings, entry, snapshot, constrained)
            if bake != plan.M:
                entry.note(
                    f"constraints rebuild this object's rotation, so the mirror was baked "
                    f"as {axes_text(bake)} to keep it looking identical"
                )
            return True
        except VerifyFailed as failure:
            del entry.done[done_mark:]
            del entry.notes[notes_mark:]
            first_failure = first_failure or failure
    raise first_failure


def _attempt(context, obj, plan, bake, settings, entry, snapshot, constrained):
    journal = Journal()
    old_world = snapshot.target_world
    expected = None if constrained else old_world @ sign_matrix4(plan.M)

    try:
        anim = AnimEditor(obj, journal, entry)

        if plan.strategy == 'DATA' and bake != IDENTITY:
            bake_data(context, obj, bake, journal, entry)
        elif plan.residual == 'ARMATURE':
            fix_armature_mirror(context, obj, journal, entry)

        flipped = negate_scale_channels(obj, "scale", "scale", anim, journal)
        if flipped:
            entry.done.append(f"scale -> {vec_text(obj.scale)}")
        flipped = negate_scale_channels(obj, "delta_scale", "delta_scale", anim, journal)
        if flipped:
            entry.done.append(f"delta scale -> {vec_text(obj.delta_scale)}")

        if plan.turn_axis is not None:
            apply_rotation_turn(obj, plan.turn_axis, "", anim, journal)
            entry.done.append(
                f"two negative axes replaced by a 180 degree turn about "
                f"{AXES[plan.turn_axis]} (rotation adjusted, data untouched)"
            )

        if plan.pi_fix:
            location = Matrix.Translation(obj.location + obj.delta_location)
            rotation = rotation_matrix3(obj).to_4x4()
            move = location @ rotation @ FX4 @ rotation.transposed() @ location.inverted()
            journal.set(obj, "matrix_parent_inverse", obj.matrix_parent_inverse @ move)
            entry.done.append("removed the mirror from the parent-inverse matrix")

        if plan.residual == 'DROP':
            entry.done.append(
                "remaining X mirror dropped (does not change how this object looks)"
            )

        # Children follow the object's real world change. Without constraints
        # that is exactly W @ M; with constraints it is measured.
        change = sign_matrix4(plan.M)
        if constrained and plan.residual != 'ARMATURE':
            context.view_layer.update()
            new_world = obj.matrix_world
            if handedness(new_world) != 0:
                change = new_world.inverted() @ old_world
        compensated = compensate_children(obj, change, journal)
        if compensated:
            entry.done.append(f"{compensated} child/Child-Of link(s) compensated, nothing moved")

        if obj.type == 'ARMATURE':
            fix_negative_pose_bones(obj, settings, journal, entry)

        for warning in plan.warnings:
            entry.note(warning)

        verify(context, snapshot, obj, expected)

    except (Blocked, VerifyFailed):
        _rollback(context, journal, obj)
        raise
    except Exception as exc:
        log(traceback.format_exc())
        _rollback(context, journal, obj)
        raise VerifyFailed(f"unexpected error: {exc}")

    journal.commit()


def _rollback(context, journal, obj):
    errors = journal.rollback()
    context.view_layer.update()
    for error in errors:
        log(f"ROLLBACK WARNING on {obj.name}: {error}")


def failure_hint(obj, message):
    hints = []
    if "moved by up to" in message and "allowed" in message:
        hints.append(
            "if this small difference is acceptable (often Bevel/Boolean evaluation "
            "order), raise Options > Tolerance and run again"
        )
    local_modifiers = [
        m.name for m in getattr(obj, "modifiers", ())
        if m.type in {'DISPLACE', 'SIMPLE_DEFORM', 'CAST', 'WAVE', 'SOLIDIFY', 'NODES',
                      'SCREW', 'CURVE', 'ARRAY'}
    ]
    if "constraint" in message:
        hints.append("a constraint overrides the transform; disable it, fix, re-enable")
    if local_modifiers:
        hints.append(
            "modifiers that work along local axes may differ after mirroring: "
            + ", ".join(f"'{n}'" for n in local_modifiers)
            + "; apply them or fix by hand"
        )
    return "; ".join(hints) or "Inspect the object; nothing was changed."


# ============================================================================
# RUNNERS
# ============================================================================

@contextmanager
def object_mode(context):
    mode = context.mode
    active = context.view_layer.objects.active
    if mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        yield
    finally:
        if mode != 'OBJECT' and active is not None:
            try:
                context.view_layer.objects.active = active
                target = {'EDIT_MESH': 'EDIT', 'EDIT_ARMATURE': 'EDIT', 'POSE': 'POSE',
                          'EDIT_CURVE': 'EDIT', 'EDIT_LATTICE': 'EDIT',
                          'SCULPT': 'SCULPT'}.get(mode)
                if target:
                    bpy.ops.object.mode_set(mode=target)
            except RuntimeError:
                pass


def run_fix(context, settings):
    ACTIVE["tolerance"] = settings.tolerance
    try:
        return _run_fix(context, settings)
    finally:
        ACTIVE["dependents"] = None
        ACTIVE["tolerance"] = POSITION_TOL


def _run_fix(context, settings):
    report = Report('FIX', "FIX", context)
    objects = collect_hierarchy(context, settings)
    initial = {obj: describe_sources(obj) for obj in objects}

    with object_mode(context):
        ACTIVE["dependents"] = build_dependents(context)
        for obj in objects:
            entry = report.add(obj)
            entry.found = list(initial.get(obj, []))
            try:
                changed = fix_object(context, obj, settings, entry)
                if changed:
                    entry.status = 'FIXED_WARN' if entry.notes else 'FIXED'
                elif entry.found:
                    entry.status = 'CLEAN'
                    entry.note("already consistent after its parents were fixed")
            except Blocked as exc:
                entry.status = 'BLOCKED'
                entry.reason = exc.reason
                entry.hint = exc.hint
                entry.done.clear()
            except VerifyFailed as exc:
                entry.status = 'FAILED'
                entry.reason = f"safety check failed and everything was rolled back: {exc}"
                entry.hint = failure_hint(obj, str(exc))
                entry.done.clear()
            context.view_layer.update()

        for obj, entry in zip(objects, report.entries):
            if entry.status in {'FIXED', 'FIXED_WARN', 'CLEAN'} \
                    and handedness(obj.matrix_world) < 0:
                origin = mirror_origin(obj)
                if origin:
                    entry.note(f"still mirrored in world space by {origin}")
                    if entry.status == 'FIXED':
                        entry.status = 'FIXED_WARN'

    report.publish()
    return report


def run_analysis(context, settings):
    report = Report('ANALYZE', "ANALYSIS (nothing changed)", context)
    for obj in collect_hierarchy(context, settings):
        entry = report.add(obj)
        entry.found = describe_sources(obj)
        if not entry.found:
            continue
        entry.status = 'FIXED_WARN'
        try:
            precheck(context, obj)
            plan = make_plan(context, obj, settings)
            if plan is not None:
                for line in describe_plan(obj, plan):
                    entry.done.append(f"will {line}")
                for warning in plan.warnings:
                    entry.note(warning)
            elif any("inherits" in f for f in entry.found):
                entry.done.append("will be corrected automatically when its parent is fixed")
            if obj.type == 'ARMATURE' and negative_pose_bones(obj):
                entry.done.append("will turn even-count negative pose-bone scales into rotations")
            ancestor = obj.parent
            while ancestor is not None:
                if describe_sources(ancestor):
                    entry.note(
                        f"{label(ancestor)} above it is fixed first, so the exact steps "
                        "for this object may change"
                    )
                    break
                ancestor = ancestor.parent
        except Blocked as exc:
            entry.status = 'BLOCKED'
            entry.reason = exc.reason
            entry.hint = exc.hint
        for item in protections(obj):
            entry.note(item)
    report.publish()
    return report


def run_verify(context, settings):
    report = Report('VERIFY', "VERIFICATION (nothing changed)", context)
    problems = 0
    for obj in collect_hierarchy(context, settings):
        entry = report.add(obj)
        issues = describe_sources(obj)
        if handedness(obj.matrix_world) < 0:
            origin = mirror_origin(obj)
            issues.append("world transform is mirrored" + (f" (by {origin})" if origin else ""))
        if handedness(obj.matrix_world) == 0:
            issues.append("world transform is singular (a zero scale axis)")
        if issues:
            entry.status = 'FAILED'
            entry.found = issues
            problems += 1
    report.publish()
    return problems


# ============================================================================
# SETTINGS
# ============================================================================

class ScaleFixSettings(bpy.types.PropertyGroup):
    scope: EnumProperty(
        name="Scope",
        items=(
            ('SELECTED', "Selection", "Selected objects with their children (and parents)"),
            ('SCENE', "Whole View Layer", "Every object in the active view layer"),
        ),
        default='SELECTED',
    )
    include_ancestors: BoolProperty(
        name="Include Parents",
        description="Also fix the parents of the selection, where mirrors usually start",
        default=True,
    )
    fix_armatures: BoolProperty(
        name="Mirror Armature Rest Pose",
        description=(
            "A mirrored armature gets its mirror moved into the bones. Pose, keys, "
            "B-bones, bone limits and drivers are converted to match"
        ),
        default=True,
    )
    fix_pose_bones: BoolProperty(
        name="Fix Pose-Bone Scale",
        description="Turn pose bones with two negative axes into a 180 degree rotation",
        default=True,
    )
    allow_visible_changes: BoolProperty(
        name="Allow Visible Changes",
        description=(
            "Cameras, volumes, image empties and collection instances can only lose a "
            "mirror by visibly un-mirroring. Enable when the mirror is a mistake"
        ),
        default=False,
    )
    convert_text: BoolProperty(
        name="Convert Mirrored Text",
        description="Text can't store a mirror. Convert mirrored text objects to meshes",
        default=False,
    )
    tolerance: bpy.props.FloatProperty(
        name="Tolerance",
        description=(
            "Largest change the safety check accepts, as a fraction of each object's "
            "size (0.0001 = 0.01%). Raise it only if a report shows a tiny difference "
            "you are happy with"
        ),
        default=POSITION_TOL, min=1.0e-6, max=0.01, precision=5, step=0.001,
    )
    verify_geometry: BoolProperty(
        name="Verify Geometry",
        description=(
            "Compare evaluated vertices and normals before/after every change and roll "
            "back on any difference (slower on very dense meshes)"
        ),
        default=True,
    )


def settings_of(context):
    return context.scene.scale_fix_settings


# ============================================================================
# OPERATORS
# ============================================================================

def _needs_objects(operator, context, settings):
    if settings.scope == 'SELECTED' and not context.selected_objects:
        operator.report({'WARNING'}, "Select at least one object, or set Scope to Whole View Layer")
        return False
    return True


class SCALE_OT_analyze(bpy.types.Operator):
    bl_idname = "scale_normalizer.analyze"
    bl_label = "Analyze"
    bl_description = "List every mirror source and what the fix would do. Changes nothing"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = settings_of(context)
        if not _needs_objects(self, context, settings):
            return {'CANCELLED'}
        report = run_analysis(context, settings)
        issues = sum(1 for e in report.entries if e.found)
        self.report({'INFO'}, f"{issues} object(s) with mirror issues. See '{REPORT_TEXT_NAME}'.")
        return {'FINISHED'}


class SCALE_OT_fix(bpy.types.Operator):
    bl_idname = "scale_normalizer.fix"
    bl_label = "Fix Negative Scale"
    bl_description = (
        "Remove negative/mirrored scale from every object in scope without moving "
        "anything. Each change is verified and rolled back if anything moves"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = settings_of(context)
        if not _needs_objects(self, context, settings):
            return {'CANCELLED'}
        report = run_fix(context, settings)
        c = report.counts()
        fixed = c['FIXED'] + c['FIXED_WARN']
        level = {'WARNING'} if (c['BLOCKED'] or c['FAILED']) else {'INFO'}
        self.report(
            level,
            f"Fixed {fixed}, need decision {c['BLOCKED']}, not changed {c['FAILED']}, "
            f"already OK {c['CLEAN']}. See '{REPORT_TEXT_NAME}'.",
        )
        return {'FINISHED'}


class SCALE_OT_verify(bpy.types.Operator):
    bl_idname = "scale_normalizer.verify"
    bl_label = "Verify"
    bl_description = "Check that no object in scope is still mirrored. Changes nothing"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = settings_of(context)
        if not _needs_objects(self, context, settings):
            return {'CANCELLED'}
        problems = run_verify(context, settings)
        if problems:
            self.report({'WARNING'}, f"{problems} object(s) still mirrored. See '{REPORT_TEXT_NAME}'.")
        else:
            self.report({'INFO'}, "No mirrored objects in scope")
        return {'FINISHED'}


class SCALE_OT_select_object(bpy.types.Operator):
    bl_idname = "scale_normalizer.select_object"
    bl_label = "Select"
    bl_description = "Select this object and make it active"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty()

    def execute(self, context):
        obj = context.view_layer.objects.get(self.name)
        if obj is None:
            self.report({'WARNING'}, f"'{self.name}' is not in this view layer")
            return {'CANCELLED'}
        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {'FINISHED'}


class SCALE_OT_show_report(bpy.types.Operator):
    bl_idname = "scale_normalizer.show_report"
    bl_label = "Open Report"
    bl_description = "Open the last report in a Text Editor window"

    def execute(self, context):
        text = bpy.data.texts.get(REPORT_TEXT_NAME)
        if text is None:
            self.report({'WARNING'}, "Run Analyze, Fix or Verify first")
            return {'CANCELLED'}
        try:
            bpy.ops.wm.window_new()
            area = context.window_manager.windows[-1].screen.areas[0]
            area.type = 'TEXT_EDITOR'
            area.spaces[0].text = text
        except (RuntimeError, IndexError, AttributeError):
            self.report({'INFO'}, f"Open a Text Editor and choose '{REPORT_TEXT_NAME}'")
        return {'FINISHED'}


# ============================================================================
# UI PANEL
# ============================================================================

class SCALE_PT_normalizer(bpy.types.Panel):
    bl_label = "Safe Scale Normalizer"
    bl_idname = "SCALE_PT_normalizer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Scale Fix"

    def draw(self, context):
        layout = self.layout
        settings = settings_of(context)

        layout.prop(settings, "scope", expand=True)

        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("scale_normalizer.fix", icon='CHECKMARK')
        row = col.row(align=True)
        row.operator("scale_normalizer.analyze", icon='VIEWZOOM')
        row.operator("scale_normalizer.verify", icon='FILE_TICK')

        summary = LAST_RUN.get("summary")
        if summary:
            box = layout.box()
            box.label(text=f"Last run: {LAST_RUN['title'].split(' (')[0].title()}", icon='INFO')
            grid = box.grid_flow(columns=2, align=True)
            for name, count in summary:
                grid.label(text=f"{name}: {count}")

            rows = LAST_RUN.get("rows", [])
            for name, status, reason in rows[:MAX_PANEL_ROWS]:
                row = box.row(align=True)
                op = row.operator(
                    "scale_normalizer.select_object", text=name, icon=STATUS_ICON[status],
                    emboss=False,
                )
                op.name = name
                row.label(text=reason[:48])
            if len(rows) > MAX_PANEL_ROWS:
                box.label(text=f"... {len(rows) - MAX_PANEL_ROWS} more in the report")
            box.operator("scale_normalizer.show_report", icon='TEXT')


class SCALE_PT_options(bpy.types.Panel):
    bl_label = "Options"
    bl_idname = "SCALE_PT_options"
    bl_parent_id = "SCALE_PT_normalizer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Scale Fix"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = settings_of(context)
        col = layout.column(align=True)
        col.prop(settings, "include_ancestors")
        col.prop(settings, "fix_armatures")
        col.prop(settings, "fix_pose_bones")
        col.prop(settings, "verify_geometry")
        col.prop(settings, "tolerance")
        col.separator()
        col.label(text="Changes you can see:")
        col.prop(settings, "allow_visible_changes")
        col.prop(settings, "convert_text")


# ============================================================================
# REGISTRATION
# ============================================================================

CLASSES = (
    ScaleFixSettings,
    SCALE_OT_analyze,
    SCALE_OT_fix,
    SCALE_OT_verify,
    SCALE_OT_select_object,
    SCALE_OT_show_report,
    SCALE_PT_normalizer,
    SCALE_PT_options,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.scale_fix_settings = PointerProperty(type=ScaleFixSettings)


def unregister():
    if hasattr(bpy.types.Scene, "scale_fix_settings"):
        del bpy.types.Scene.scale_fix_settings
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
