# =============================================================================
#  RealMotion Retarget (standalone) - v1.71, Blender 5.1
# =============================================================================
#  v1.71 fixes / hardening
#   * Bake invalidates only the spacing helper cache; selection baselines survive bake.
#   * Removed redundant Bake cache reset and dead registration tagging in unregister().
#   * Bind captures the final helper name once, after its transactional rename.
#   * Registration recognizes legacy RealMotion classes across versioned module filenames
#     during Reload Scripts while still rejecting foreign classes.
#   * Spacing helper Action isolation uses the actual sentinel contract explicitly.
#   * Minor registration formatting/clarity cleanup.
#
#  v1.70 and earlier
#   * See prior release history; functional retargeting behavior is intentionally preserved.
# =============================================================================
import bpy
import math
import os

from bpy.props import StringProperty
from bpy.types import Operator
from bpy.app.handlers import persistent
from bpy_extras.io_utils import ImportHelper

bl_info = {
    "name": "RealMotion Retarget Standalone (Exact Source)",
    "author": "Venkatesh Sanku",
    "version": (1, 71),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > RealMotion Pro",
    "description": "Animation retargeting between armatures with Blender 5.1 compatibility and action-slot support",
    "category": "Animation",
}

RETARGET_ID = "_RM"
CONSTRAINT_SOURCE_TAG = "_real_motion_retarget_constraint_source"
HELPER_TAG = "_real_motion_retarget_helper"
HELPER_SOURCE_NAME = "_real_motion_retarget_source"

COPY_ROTATION_NAME = "Copy Rotation" + RETARGET_ID
COPY_LOCATION_NAME = "Copy Location" + RETARGET_ID
CONSTRAINT_TAG = "_real_motion_retarget_constraint"


# -----------------------------------------------------------------------------
# Animation helpers (slotted actions: Blender 4.4+ / 5.x)
# -----------------------------------------------------------------------------

def get_channelbags(id_block):
    """Return all keyframe-strip channelbags for the block's assigned Action Slot."""
    anim = getattr(id_block, "animation_data", None)
    if anim is None or anim.action is None or anim.action_slot is None:
        return []

    action = anim.action
    slot = anim.action_slot
    result = []
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != 'KEYFRAME':
                continue
            channelbag = strip.channelbag(slot)
            if channelbag is not None:
                result.append(channelbag)
    return result


def get_fcurves(id_block):
    """All F-Curves of the object's active Action Slot (empty list if none)."""
    return [fcurve for bag in get_channelbags(id_block) for fcurve in list(bag.fcurves)]


def remove_fcurves(id_block, should_remove):
    """Remove every F-Curve for which should_remove(fcurve) is True."""
    removed = 0
    for channelbag in get_channelbags(id_block):
        # Never mutate the RNA collection while iterating over it.
        doomed = [fc for fc in channelbag.fcurves if should_remove(fc)]
        for fcurve in doomed:
            channelbag.fcurves.remove(fcurve)
            removed += 1
    return removed


def get_keyframe_range(id_block):
    """Return (first, last) keyframe time for the assigned Action Slot, or None."""
    frames = [
        kp.co[0]
        for fc in get_fcurves(id_block)
        for kp in fc.keyframe_points
    ]
    return (min(frames), max(frames)) if frames else None


def ensure_single_user_action(id_block):
    """
    Make the assigned Action single-user before direct F-Curve edits.
    Returns the resulting Action, or None when the block has no active Action.
    """
    anim = getattr(id_block, "animation_data", None)
    if anim is None or anim.action is None:
        return None

    action = anim.action
    if getattr(action, "users", 1) <= 1 and getattr(action, "library", None) is None:
        return action

    # Direct F-Curve edits must remain local. A linked Action is therefore
    # intentionally copied into the current .blend rather than editing the library.

    old_slot_identifier = None
    if anim.action_slot is not None:
        old_slot_identifier = getattr(anim.action_slot, "identifier", None)

    new_action = action.copy()
    anim.action = new_action

    # Preserve the exact logical slot where possible.
    if old_slot_identifier:
        for slot in new_action.slots:
            if getattr(slot, "identifier", None) == old_slot_identifier:
                anim.action_slot = slot
                break

    return new_action


def clone_action_for_transaction(id_block):
    """Copy an assigned Action and preserve its logical slot, for safe pre-commit edits."""
    anim = getattr(id_block, "animation_data", None)
    if anim is None or anim.action is None:
        return None, None, None

    original_action = anim.action
    original_slot_identifier = (
        getattr(anim.action_slot, "identifier", None)
        if anim.action_slot is not None else None
    )

    working_action = original_action.copy()
    anim.action = working_action

    restored_slot = None
    if original_slot_identifier:
        for slot in working_action.slots:
            if getattr(slot, "identifier", None) == original_slot_identifier:
                restored_slot = slot
                break

    if restored_slot is not None:
        anim.action_slot = restored_slot
    elif len(working_action.slots) == 0:
        try:
            anim.action_slot = working_action.slots.new('OBJECT', id_block.name)
        except (AttributeError, RuntimeError, TypeError):
            anim.action_slot = None
    else:
        anim.action_slot = next(iter(working_action.slots), None)

    return original_action, original_slot_identifier, working_action


def restore_action_transaction(id_block, original_action, original_slot_identifier, working_action):
    """Restore an Action replaced by clone_action_for_transaction()."""
    if original_action is None:
        return

    anim = id_block.animation_data_create()
    anim.action = original_action

    restored_slot = None
    if original_slot_identifier:
        for slot in original_action.slots:
            if getattr(slot, "identifier", None) == original_slot_identifier:
                restored_slot = slot
                break
    if restored_slot is not None:
        anim.action_slot = restored_slot

    if (
        working_action is not None
        and working_action != original_action
        and getattr(working_action, "users", 0) == 0
        and getattr(working_action, "library", None) is None
    ):
        try:
            bpy.data.actions.remove(working_action)
        except RuntimeError:
            pass



def remove_unused_local_action(action):
    """Remove one local Action when it no longer has any users."""
    if action is None:
        return False
    try:
        if getattr(action, "library", None) is not None:
            return False
        if getattr(action, "users", 0) != 0:
            return False
        bpy.data.actions.remove(action)
        return True
    except (AttributeError, RuntimeError, ReferenceError):
        return False


def cleanup_new_orphan_actions(before_actions):
    """Remove only newly-created, unused local Actions from a transaction.

    A missing snapshot is treated as "do not clean". Failing closed here prevents
    a snapshot error from ever turning into a purge of pre-existing orphan Actions.
    """
    if before_actions is None:
        return 0

    try:
        before_actions = set(before_actions)
        current_actions = set(bpy.data.actions)
    except (AttributeError, RuntimeError, TypeError):
        return 0

    removed = 0
    for action in current_actions - before_actions:
        if remove_unused_local_action(action):
            removed += 1
    return removed


# Object-transform animation paths. Pose-bone paths are intentionally excluded.
OBJECT_TRANSFORM_PATHS = {
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "rotation_axis_angle",
    "scale",
    "delta_location",
    "delta_rotation_euler",
    "delta_rotation_quaternion",
    "delta_scale",
}


def has_object_transform_animation(id_block):
    """Return True when the active Action Slot contains object-level transform animation."""
    for fcurve in get_fcurves(id_block):
        if fcurve.data_path in OBJECT_TRANSFORM_PATHS:
            return True
    return False


OBJECT_SCALE_PATHS = {
    "scale",
    "delta_scale",
}

OBJECT_LOCATION_ROTATION_PATHS = {
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "rotation_axis_angle",
    "delta_location",
    "delta_rotation_euler",
    "delta_rotation_quaternion",
}


def _object_action_paths(id_block):
    """Return animated object-transform paths in the active Action Slot."""
    return {
        fcurve.data_path
        for fcurve in get_fcurves(id_block)
        if fcurve.data_path in OBJECT_TRANSFORM_PATHS
    }


def scale_pose_location_fcurves(armature, scale):
    """Scale pose-bone location keys to compensate for static object scale.

    The multiplication direction is intentional: transform_apply() transfers the
    object's static scale into the armature-data transform, so multiplying each
    armature-local location component by that same scale preserves the original
    world-space displacement. Static non-uniform scale is handled component-wise;
    animated object scale is rejected by normalize_imported_armature_transforms().
    """
    factors = (float(scale[0]), float(scale[1]), float(scale[2]))
    changed = 0
    for fcurve in get_fcurves(armature):
        path = fcurve.data_path
        if not (path.startswith('pose.bones["') and path.endswith('"].location')):
            continue
        axis = int(getattr(fcurve, "array_index", -1))
        if not 0 <= axis < 3:
            continue
        factor = factors[axis]
        if abs(factor - 1.0) < 1e-12:
            continue
        for kp in fcurve.keyframe_points:
            kp.co[1] *= factor
            kp.handle_left[1] *= factor
            kp.handle_right[1] *= factor
        fcurve.update()
        changed += 1
    return changed


def normalize_imported_armature_transforms(armature):
    """Normalize imported armature transforms while preserving static-scale pose animation.

    Returns ``(applied, skipped_scale, warning)``:
      * animated object scale -> ``(False, True, message)``; no transform_apply is run
      * animated object location/rotation with static scale != 1 -> ``(True, False, None)``
      * animated object location/rotation with scale == 1 -> ``(False, False, None)``
      * no object transform animation -> ``(True, False, None)``
    """
    scale = tuple(float(v) for v in armature.scale)
    scale_is_one = all(abs(v - 1.0) < 1e-8 for v in scale)

    animated_paths = _object_action_paths(armature)
    scale_animated = bool(animated_paths & OBJECT_SCALE_PATHS)
    loc_rot_animated = bool(animated_paths & OBJECT_LOCATION_ROTATION_PATHS)

    if scale_animated:
        return False, True, (
            "Imported armature has animated object scale; transform scale normalization "
            "was skipped to preserve animation. Apply/retime the object scale manually."
        )

    if not scale_is_one:
        ensure_single_user_action(armature)
        scale_pose_location_fcurves(armature, scale)

    if loc_rot_animated:
        # Static scale can still be applied independently without disturbing animated
        # object location/rotation channels. Pose locations were compensated above.
        if not scale_is_one:
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            return True, False, None
        return False, False, None

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return True, False, None


# -----------------------------------------------------------------------------
# Scene helpers
# -----------------------------------------------------------------------------

def object_mode():
    """Leave Edit/Pose mode of the active object."""
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def select_only(*objects):
    """Select only valid given objects; the last one becomes active."""
    valid_objects = [obj for obj in objects if obj is not None]
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    for obj in valid_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = valid_objects[-1] if valid_objects else None


def reset_pose(armature):
    """Clear location/rotation/scale of every pose bone."""
    if armature is None or armature.type != 'ARMATURE':
        return
    for pb in armature.pose.bones:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


def reset_pose_bones(armature, bone_names):
    """Reset only the named pose bones, preserving the rest of the target rig."""
    if armature is None or armature.type != 'ARMATURE':
        return
    for name in bone_names:
        pb = armature.pose.bones.get(name)
        if pb is None:
            continue
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


def is_retarget_constraint(con, helper=None):
    """Identify this add-on's constraint; when helper is supplied, require that exact target."""
    target = getattr(con, "target", None)

    # A helper filter must apply to tagged constraints too. This is important
    # when multiple source rigs are bound into the same target scene.
    if helper is not None and target != helper:
        return False

    if (
        con.type in {'COPY_ROTATION', 'COPY_LOCATION'}
        and con.get(CONSTRAINT_TAG, False)
    ):
        return True

    # v1.58 used these exact names but did not tag constraints.
    # Keep the legacy-name fallback constrained to the copy types we create.
    if (
        con.type in {'COPY_ROTATION', 'COPY_LOCATION'}
        and con.name in (COPY_ROTATION_NAME, COPY_LOCATION_NAME)
    ):
        return True

    # Legacy constraints could be auto-renamed because the user already had
    # a constraint with the same name, so accept the expected prefix as long
    # as it targets the requested helper.
    if (
        helper is not None
        and con.type in {'COPY_ROTATION', 'COPY_LOCATION'}
        and (
            con.name.startswith("Copy Rotation" + RETARGET_ID)
            or con.name.startswith("Copy Location" + RETARGET_ID)
        )
    ):
        return True

    return False


def has_retarget_constraint(pose_bone, helper=None):
    """Return True when this bone has a constraint created by this add-on."""
    return any(is_retarget_constraint(con, helper) for con in pose_bone.constraints)


def has_retarget_constraint_for_source(pose_bone, source_name, helper_name=None):
    """Return True when this bone has an RM constraint attributable to one source."""
    if pose_bone is None or not source_name:
        return False

    legacy_names = {COPY_ROTATION_NAME, COPY_LOCATION_NAME}
    for con in pose_bone.constraints:
        if con.type not in {'COPY_ROTATION', 'COPY_LOCATION'}:
            continue

        if (con.get(CONSTRAINT_TAG, False) and
                con.get(CONSTRAINT_SOURCE_TAG, "") == source_name):
            return True

        target = getattr(con, "target", None)
        if helper_name and getattr(target, "name", None) == helper_name:
            if (con.name in legacy_names or
                    con.name.startswith(("Copy Rotation" + RETARGET_ID,
                                         "Copy Location" + RETARGET_ID))):
                return True

    return False


def remove_retarget_constraints(armature, helper=None, exclude_helper=None):
    """Remove RM constraints; helper restricts removal to that exact helper target."""
    if armature is None or armature.type != 'ARMATURE':
        return 0

    removed = 0
    for pb in armature.pose.bones:
        doomed = []
        for con in pb.constraints:
            if exclude_helper is not None and getattr(con, "target", None) == exclude_helper:
                continue
            if is_retarget_constraint(con, helper):
                doomed.append(con)
        for con in doomed:
            pb.constraints.remove(con)
            removed += 1
    return removed


def remove_retarget_constraints_for_source(
    armature, source_name, helper_name=None, exclude_helper=None
):
    """Remove RM constraints for one source, optionally excluding a new helper target."""
    if armature is None or armature.type != 'ARMATURE' or not source_name:
        return 0

    removed = 0
    legacy_names = {COPY_ROTATION_NAME, COPY_LOCATION_NAME}
    for pb in armature.pose.bones:
        doomed = []
        for con in pb.constraints:
            if con.type not in {'COPY_ROTATION', 'COPY_LOCATION'}:
                continue

            target = getattr(con, "target", None)
            if exclude_helper is not None and target == exclude_helper:
                continue

            if (con.get(CONSTRAINT_TAG, False) and
                    con.get(CONSTRAINT_SOURCE_TAG, "") == source_name):
                doomed.append(con)
                continue

            if helper_name and getattr(target, "name", None) == helper_name:
                if (con.name in legacy_names or
                        con.name.startswith(("Copy Rotation" + RETARGET_ID,
                                             "Copy Location" + RETARGET_ID))):
                    doomed.append(con)

        for con in doomed:
            pb.constraints.remove(con)
            removed += 1
    return removed


def remove_helper(obj):
    """Delete a helper armature, its unused armature data, and any orphaned helper Action."""
    if obj is None or obj.type != 'ARMATURE':
        return

    arm_data = obj.data
    helper_action = None
    try:
        anim = obj.animation_data
        if anim is not None:
            helper_action = anim.action
    except (AttributeError, ReferenceError, RuntimeError):
        helper_action = None

    bpy.data.objects.remove(obj, do_unlink=True)

    if arm_data is not None and arm_data.users == 0:
        bpy.data.armatures.remove(arm_data)

    remove_unused_local_action(helper_action)


def find_helper_for_source(source_armature, target_hint=None):
    """Find the active RealMotion helper for a source, including legacy helpers.

    A target_hint should be supplied by context-aware callers. The canonical helper
    name is preferred when several source-tagged helpers exist.
    """
    if source_armature is None:
        return None

    expected_name = source_armature.name + RETARGET_ID

    def referenced_helper(target_obj):
        if target_obj is None or target_obj.type != 'ARMATURE':
            return None
        for pb in target_obj.pose.bones:
            for con in pb.constraints:
                helper = getattr(con, "target", None)
                if helper is None or helper.type != 'ARMATURE' or helper.parent != source_armature:
                    continue
                if con.get(CONSTRAINT_TAG, False):
                    if con.type in {'COPY_ROTATION', 'COPY_LOCATION'}:
                        source_tag = con.get(CONSTRAINT_SOURCE_TAG, "")
                        # v1.60 tagged constraints did not yet store a source tag.
                        # The helper's explicit parent relationship is still enough
                        # to identify those legacy constraints safely.
                        if source_tag in {"", source_armature.name}:
                            return helper
                elif is_retarget_constraint(con, helper):
                    return helper
        return None

    # Normal path: only inspect the active target. This keeps Spacing redraws fast.
    helper = referenced_helper(target_hint)
    if helper is not None:
        return helper

    candidate = bpy.data.objects.get(expected_name)
    if candidate is not None and candidate.type == 'ARMATURE':
        if candidate.parent == source_armature:
            return candidate
        if (candidate.get(HELPER_TAG) and
                candidate.get(HELPER_SOURCE_NAME) == source_armature.name):
            return candidate

    tagged = []
    legacy = []
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE' or obj.parent != source_armature:
            continue
        if obj.get(HELPER_TAG):
            tagged.append(obj)
        elif obj.name == expected_name:
            legacy.append(obj)

    if tagged:
        return sorted(tagged, key=lambda obj: obj.name)[0]
    if legacy:
        return legacy[0]

    # Multi-target fallback.
    for target_obj in bpy.data.objects:
        if target_obj.type != 'ARMATURE' or target_obj == target_hint:
            continue
        helper = referenced_helper(target_obj)
        if helper is not None:
            return helper

    return None


def copy_rest_pose(context, source_armature):
    """Duplicate the source armature without linking armature data."""
    select_only(source_armature)
    if not source_armature.select_get():
        return None

    try:
        result = bpy.ops.object.duplicate(linked=False)
    except RuntimeError as exc:
        print(f"Error duplicating source armature: {exc}")
        return None

    if 'FINISHED' not in result:
        return None

    helper_armature = context.view_layer.objects.active
    if (
        helper_armature is None
        or helper_armature == source_armature
        or helper_armature.type != 'ARMATURE'
    ):
        return None

    # Extra safety in case Blender/preferences resulted in linked armature data.
    if helper_armature.data == source_armature.data:
        helper_armature.data = source_armature.data.copy()

    return helper_armature


def get_preset_dir():
    """Default add-on-specific folder for bone mapping presets (created on demand)."""
    candidates = []
    try:
        datafiles_dir = bpy.utils.user_resource(
            'DATAFILES',
            path=os.path.join("RealMotionRetarget", "Presets"),
            create=True,
        )
        if datafiles_dir:
            candidates.append(datafiles_dir)
    except Exception:
        pass

    candidates.append(os.path.join(os.path.expanduser("~"), "RealMotionPresets"))
    for path in candidates:
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            continue
    return os.path.expanduser("~")


def import_fbx_file(filepath):
    """Import an FBX with whichever importer this Blender build provides."""
    importers = (
        ("Python importer", lambda: bpy.ops.import_scene.fbx(filepath=filepath)),
        ("native importer", lambda: bpy.ops.wm.fbx_import(filepath=filepath)),
    )
    errors = []
    for label, run in importers:
        try:
            result = run()
            if 'FINISHED' in result:
                return
            errors.append(f"{label}: operator returned {sorted(result)}")
        except (AttributeError, RuntimeError, TypeError) as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError("FBX import failed - " + "; ".join(errors))


# -----------------------------------------------------------------------------
# Properties
# -----------------------------------------------------------------------------

class BoneMappingItem(bpy.types.PropertyGroup):
    source_bone: bpy.props.StringProperty(name="Source Bone")
    target_bone: bpy.props.StringProperty(
        name="Target Bone",
        default="",
        description="Select target bone",
    )


class RetargetProperties(bpy.types.PropertyGroup):
    source_armature: bpy.props.PointerProperty(
        name="Source Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE',
    )
    target_armature: bpy.props.PointerProperty(
        name="Target Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE',
    )
    root_bone: bpy.props.StringProperty(name="Root Bone", default="")
    bone_mappings: bpy.props.CollectionProperty(type=BoneMappingItem)
    active_bone_index: bpy.props.IntProperty(name="Active Bone Index", default=0)


class SIMPLE_RETARGET_UL_BoneMappingList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "source_bone", text="", emboss=False)
        target = context.scene.simple_retarget.target_armature
        if target is not None and target.type == 'ARMATURE':
            row.prop_search(item, "target_bone", target.pose, "bones", text="")
        else:
            row.prop(item, "target_bone", text="")


# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------

class SavePresetOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.save_preset"
    bl_label = "Save Preset"
    bl_description = "Save the current bone mappings as a preset"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        props = context.scene.simple_retarget

        if not self.filepath.lower().endswith(".txt"):
            self.filepath += ".txt"

        try:
            parent_dir = os.path.dirname(os.path.abspath(self.filepath))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(self.filepath, 'w', encoding='utf-8') as file:
                for mapping in props.bone_mappings:
                    source = mapping.source_bone.strip()
                    target = mapping.target_bone.strip()
                    file.write(f"{source},{target}\n")
        except OSError as exc:
            self.report({'ERROR'}, f"Could not save preset: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Preset saved to {self.filepath}")
        return {'FINISHED'}

    def invoke(self, context, event):
        self.filepath = os.path.join(get_preset_dir(), "preset.txt")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class LoadPresetOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.load_preset"
    bl_label = "Load Preset"
    bl_description = "Load a bone mappings preset from a file"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    def execute(self, context):
        props = context.scene.simple_retarget
        source_armature = props.source_armature
        target_armature = props.target_armature

        try:
            with open(self.filepath, 'r', encoding='utf-8-sig') as file:
                lines = file.read().splitlines()
        except OSError as exc:
            self.report({'ERROR'}, f"Could not read preset: {exc}")
            return {'CANCELLED'}

        source_bone_names = (
            set(source_armature.data.bones.keys())
            if source_armature and source_armature.type == 'ARMATURE'
            else set()
        )
        target_bone_names = (
            set(target_armature.data.bones.keys())
            if target_armature and target_armature.type == 'ARMATURE'
            else set()
        )

        parsed = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(',', 1)
            if len(parts) != 2:
                continue

            source_bone = parts[0].strip()
            target_bone = parts[1].strip()
            if not source_bone:
                continue
            if source_bone not in source_bone_names:
                continue

            if target_bone not in target_bone_names:
                target_bone = ""
            parsed.append((source_bone, target_bone))

        bone_mappings = props.bone_mappings
        bone_mappings.clear()
        for source_bone, target_bone in parsed:
            new_mapping = bone_mappings.add()
            new_mapping.source_bone = source_bone
            new_mapping.target_bone = target_bone

        # Do not leave a root bone pointing at a source bone that is no longer mapped.
        if not any(m.source_bone == props.root_bone for m in bone_mappings):
            props.root_bone = ""

        props.active_bone_index = max(0, min(props.active_bone_index, len(bone_mappings) - 1)) if bone_mappings else 0

        self.report(
            {'INFO'},
            f"Preset loaded from {self.filepath} ({len(parsed)} valid source mappings)",
        )
        return {'FINISHED'}

    def invoke(self, context, event):
        self.directory = get_preset_dir()
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# -----------------------------------------------------------------------------
# Highlight / main panel
# -----------------------------------------------------------------------------

class SIMPLE_RETARGET_OT_HighlightBone(bpy.types.Operator):
    bl_idname = "simple_retarget.highlight_bone"
    bl_label = "Highlight Selected Bone"
    bl_description = "Highlight the selected bone in the bone mapping list (source or target)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.simple_retarget

        if context.mode != 'POSE':
            self.report({'ERROR'}, "Please enter Pose Mode to select a bone.")
            return {'CANCELLED'}

        selected_bone = context.active_pose_bone
        if selected_bone is None:
            self.report({'ERROR'}, "No bone selected. Please select a bone in Pose Mode.")
            return {'CANCELLED'}

        if not props.source_armature:
            self.report({'ERROR'}, "Source armature is not set.")
            return {'CANCELLED'}
        if not props.target_armature:
            self.report({'ERROR'}, "Target armature is not set.")
            return {'CANCELLED'}

        if selected_bone.id_data not in (props.source_armature, props.target_armature):
            self.report({'ERROR'}, "Selected bone does not belong to the source or target armature.")
            return {'CANCELLED'}

        for index, mapping in enumerate(props.bone_mappings):
            if mapping.source_bone == selected_bone.name or mapping.target_bone == selected_bone.name:
                props.active_bone_index = index
                self.report({'INFO'}, f"Highlighted bone: {selected_bone.name}")
                return {'FINISHED'}

        self.report({'WARNING'}, "Selected bone not found in the bone mapping list.")
        return {'CANCELLED'}


class SimpleRetargetPanel(bpy.types.Panel):
    bl_label = "RealMotion Retarget"
    bl_idname = "VIEW3D_PT_simple_retarget"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RealMotion Pro"
    bl_order = 5

    def draw(self, context):
        layout = self.layout
        props = context.scene.simple_retarget

        layout.prop(props, "source_armature")
        layout.prop(props, "target_armature")
        layout.operator("simple_retarget.build_bone_list", text="Build Bone List")

        row = layout.row()
        row.template_list(
            "SIMPLE_RETARGET_UL_BoneMappingList",
            "",
            props,
            "bone_mappings",
            props,
            "active_bone_index",
        )

        col = row.column(align=True)
        col.operator("simple_retarget.add_mapping", icon='ADD', text="")
        col.operator("simple_retarget.remove_mapping", icon='REMOVE', text="")

        row = layout.row()
        row.operator("simple_retarget.highlight_bone", icon='VIEWZOOM', text="Highlight Selected Bone")

        row = layout.row()
        row.operator("simple_retarget.load_preset", text="Load Preset")
        row.operator("simple_retarget.save_preset", text="Save Preset")

        layout.label(text="Root Bone:")
        row = layout.row()
        row.alert = not bool(props.root_bone)
        row.label(
            text=props.root_bone if props.root_bone else "None Selected",
            icon='BONE_DATA',
        )

        row = layout.row()
        row.operator("simple_retarget.set_root_bone", text="Set Root Bone")
        row.operator("simple_retarget.calculate_scale", text="Calculate Scale")

        row = layout.row()
        row.operator("simple_retarget.bind_constraints", text="Bind Constraints")
        row.operator("simple_retarget.unbind_constraints", text="Unbind Constraints")


# -----------------------------------------------------------------------------
# Mapping validation
# -----------------------------------------------------------------------------

def collect_valid_mapping_data(props):
    """
    Validate the current mapping list before any scene edits.

    Returns (pairs, root_source, root_target, affected_target_names, error).
    """
    source_armature = props.source_armature
    target_armature = props.target_armature

    if not source_armature or source_armature.type != 'ARMATURE':
        return None, None, None, None, "Source armature is not assigned."
    if not target_armature or target_armature.type != 'ARMATURE':
        return None, None, None, None, "Target armature is not assigned."
    if source_armature == target_armature:
        return None, None, None, None, "Source and Target must be different armatures."

    source_names = set(source_armature.data.bones.keys())
    target_names = set(target_armature.data.bones.keys())
    root_source = props.root_bone.strip()

    pairs = []
    seen_targets = set()
    root_target = ""

    for index, mapping in enumerate(props.bone_mappings):
        source_name = mapping.source_bone.strip()
        target_name = mapping.target_bone.strip()

        if not source_name and not target_name:
            continue
        if not source_name:
            return None, None, None, None, f"Mapping row {index + 1}: source bone is empty."
        if not target_name:
            # Empty target is allowed for an intentionally unmapped row.
            continue
        if source_name not in source_names:
            return None, None, None, None, f"Mapping row {index + 1}: source bone '{source_name}' does not exist."
        if target_name not in target_names:
            return None, None, None, None, f"Mapping row {index + 1}: target bone '{target_name}' does not exist."
        if target_name in seen_targets:
            return None, None, None, None, f"Target bone '{target_name}' is mapped more than once."

        seen_targets.add(target_name)
        pairs.append((source_name, target_name))

        if source_name == root_source:
            root_target = target_name

    if not root_source:
        return None, None, None, None, "Root bone is not set."
    if root_source not in source_names:
        return None, None, None, None, f"Root bone '{root_source}' does not exist on the source armature."
    if not root_target:
        return None, None, None, None, "Root bone is not properly mapped to a target bone."
    if not pairs:
        return None, None, None, None, "No valid source/target bone mappings were provided."

    return pairs, root_source, root_target, seen_targets, None


# -----------------------------------------------------------------------------
# Bind / Unbind
# -----------------------------------------------------------------------------

class BindConstraintsOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.bind_constraints"
    bl_label = "Bind Constraints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = None
        source_armature = None
        target_armature = None
        mapped_target_names = set()

        # Keep basic property/preflight access inside a protected scope. A malformed or
        # partially-reloaded RNA state should be reported through the operator instead of
        # escaping as a raw traceback before the transactional bind can begin.
        try:
            props = context.scene.simple_retarget
            source_armature = props.source_armature
            target_armature = props.target_armature

            pairs, root_source, root_target, mapped_target_names, error = collect_valid_mapping_data(props)
            if error:
                self.report({'WARNING'}, error)
                return {'CANCELLED'}

            # The bind workflow requires both rigs to remain editable for the Pose-mode
            # operations below, so viewport visibility is intentionally an invariant.
            if not source_armature.visible_get() or not target_armature.visible_get():
                self.report({'WARNING'}, "Source and Target armatures must be visible in the viewport.")
                return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Bind preflight failed: {exc}")
            return {'CANCELLED'}

        old_helper = None
        new_helper = None
        new_helper_name = None
        created_constraints = []
        commit_started = False
        original_target_action = None
        original_target_slot_identifier = None
        working_target_action = None
        affected_target_names = set(mapped_target_names)
        before_actions = None

        try:
            before_actions = set(bpy.data.actions)
            old_helper = find_helper_for_source(source_armature, target_armature)
            if old_helper is not None:
                old_affected = {
                    pb.name
                    for pb in target_armature.pose.bones
                    if has_retarget_constraint(pb, old_helper)
                }
            else:
                # Never treat all tagged RM constraints as the old bind. When no helper
                # is discoverable, only constraints attributable to this source may be
                # included in the replacement's affected-bone set.
                old_affected = {
                    pb.name
                    for pb in target_armature.pose.bones
                    if has_retarget_constraint_for_source(
                        pb, source_armature.name, source_armature.name + RETARGET_ID
                    )
                }
            affected_target_names |= old_affected
            object_mode()

            # Build the replacement helper completely before touching the old bind.
            new_helper = copy_rest_pose(context, source_armature)
            if new_helper is None:
                raise RuntimeError(
                    "Failed to copy source armature. It may be hidden, linked, or unavailable."
                )

            # Keep the temporary helper name unique while the old helper still exists.
            # Blender may append .001 when the canonical name is already occupied;
            # the canonical name is reclaimed after the old helper is removed.
            base_helper_name = source_armature.name + RETARGET_ID
            if old_helper is not None:
                new_helper.name = f"{base_helper_name}.BUILD"
            else:
                new_helper.name = base_helper_name

            # Capture the final datablock name now. The exception path uses this string
            # rather than new_helper.name because a later transaction failure may
            # invalidate the RNA reference before cleanup runs.
            new_helper_name = new_helper.name
            new_helper[HELPER_TAG] = True
            new_helper[HELPER_SOURCE_NAME] = source_armature.name

            # Spacing edits directly modify this helper's rotation F-Curves.
            # Never allow those edits to write into the source rig's shared Action.
            ensure_single_user_action(new_helper)

            select_only(new_helper, source_armature)
            result = bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
            if 'FINISHED' not in result:
                raise RuntimeError("Could not parent the helper armature to the source armature.")

            new_helper.hide_set(True)

            # Record target rest-bone positions in helper armature space.
            select_only(target_armature)
            bpy.ops.object.mode_set(mode='EDIT')
            try:
                to_helper = new_helper.matrix_world.inverted() @ target_armature.matrix_world
                bone_transforms = {}
                for bone in target_armature.data.edit_bones:
                    bone_transforms[bone.name] = (
                        to_helper @ bone.head,
                        to_helper @ bone.tail,
                        bone.roll,
                    )
            finally:
                bpy.ops.object.mode_set(mode='OBJECT')

            # Add target-shaped helper bones, parented to the matching source bones.
            new_helper.hide_set(False)
            select_only(new_helper)
            bpy.ops.object.mode_set(mode='EDIT')

            rm_bone_names = {}
            try:
                edit_bones = new_helper.data.edit_bones
                for source_bone_name, target_bone_name in pairs:
                    source_bone = edit_bones.get(source_bone_name)
                    if source_bone is None:
                        raise RuntimeError(
                            f"Source helper bone '{source_bone_name}' was not found."
                        )

                    transform_data = bone_transforms.get(target_bone_name)
                    if transform_data is None:
                        raise RuntimeError(
                            f"Target edit bone '{target_bone_name}' was not found."
                        )

                    head, tail, roll = transform_data
                    if (tail - head).length <= 1e-8:
                        raise RuntimeError(
                            f"Target bone '{target_bone_name}' has zero length and cannot be used for retargeting."
                        )
                    base_name = target_bone_name + RETARGET_ID
                    new_bone_name = base_name
                    suffix = 1
                    while edit_bones.get(new_bone_name) is not None:
                        new_bone_name = f"{base_name}.{suffix:03d}"
                        suffix += 1

                    new_bone = edit_bones.new(new_bone_name)
                    new_bone.head = head
                    new_bone.tail = tail
                    new_bone.roll = roll
                    new_bone.parent = source_bone
                    new_bone.use_connect = False
                    rm_bone_names[target_bone_name] = new_bone.name
            finally:
                bpy.ops.object.mode_set(mode='OBJECT')

            new_helper.hide_set(True)
            bpy.context.view_layer.update()

            # Create replacement constraints MUTED while the old bind remains live.
            # If any constraint setup fails, only these new muted constraints are removed.
            select_only(target_armature)
            for source_bone_name, target_bone_name in pairs:
                target_pose_bone = target_armature.pose.bones.get(target_bone_name)
                helper_bone_name = rm_bone_names.get(target_bone_name)
                if target_pose_bone is None or helper_bone_name is None:
                    raise RuntimeError(
                        f"Could not build constraint target for target bone '{target_bone_name}'."
                    )
                if new_helper.pose.bones.get(helper_bone_name) is None:
                    raise RuntimeError(
                        f"Helper bone '{helper_bone_name}' was removed or is no longer available."
                    )

                rot_con = target_pose_bone.constraints.new('COPY_ROTATION')
                rot_con.name = COPY_ROTATION_NAME
                rot_con[CONSTRAINT_TAG] = True
                rot_con[CONSTRAINT_SOURCE_TAG] = source_armature.name
                rot_con.target = new_helper
                rot_con.subtarget = helper_bone_name
                rot_con.owner_space = 'POSE'
                rot_con.target_space = 'POSE'
                rot_con.mute = True
                created_constraints.append(rot_con)

                if target_bone_name == root_target:
                    # Root translation intentionally follows the ORIGINAL source bone,
                    # because the helper bone is target-shaped and is used for rotation.
                    loc_con = target_pose_bone.constraints.new('COPY_LOCATION')
                    loc_con.name = COPY_LOCATION_NAME
                    loc_con[CONSTRAINT_TAG] = True
                    loc_con[CONSTRAINT_SOURCE_TAG] = source_armature.name
                    loc_con.target = new_helper
                    loc_con.subtarget = source_bone_name
                    loc_con.owner_space = 'WORLD'
                    loc_con.target_space = 'WORLD'
                    loc_con.mute = True
                    created_constraints.append(loc_con)

            # Prepare target animation edits without touching the old Action when a
            # previous bind exists. This makes pre-commit failures reversible.
            if old_helper is not None:
                original_target_action, original_target_slot_identifier, working_target_action = (
                    clone_action_for_transaction(target_armature)
                )
            else:
                ensure_single_user_action(target_armature)

            affected_transform_paths = tuple(
                f'pose.bones["{bpy.utils.escape_identifier(name)}"].{property_name}'
                for name in affected_target_names
                for property_name in (
                    "location",
                    "rotation_euler",
                    "rotation_quaternion",
                    "rotation_axis_angle",
                    "scale",
                )
            )
            if affected_transform_paths:
                affected_transform_set = set(affected_transform_paths)
                remove_fcurves(
                    target_armature,
                    lambda fc: fc.data_path in affected_transform_set,
                )

            # Commit protocol:
            #   1) Reset affected pose values.
            #   2) Activate the replacement constraints.
            #   3) Mark the transaction committed.
            #   4) Mute/remove the old constraints and helper as best-effort cleanup.
            # This ordering guarantees that a late cleanup failure cannot leave the
            # target with no live retarget bind.
            reset_pose_bones(target_armature, affected_target_names)

            for con in created_constraints:
                con.mute = False

            commit_started = True

            # The old bind may coexist with the new bind for this short RNA-edit
            # transaction, but we do not force a dependency-graph update until the
            # old constraints have been muted. Treat this as cleanup rather than a
            # failure of the already-prepared replacement bind.
            if old_helper is not None:
                try:
                    for pb in target_armature.pose.bones:
                        for con in pb.constraints:
                            if (
                                is_retarget_constraint(con, old_helper)
                                and con not in created_constraints
                            ):
                                con.mute = True
                except Exception as mute_exc:
                    print(f"RealMotion Bind old-constraint mute warning: {mute_exc}")

            try:
                bpy.context.view_layer.update()
            except Exception as update_exc:
                print(f"RealMotion Bind update warning: {update_exc}")

            # Old-bind cleanup is always source-specific. If the helper lookup failed,
            # never call the unfiltered remove_retarget_constraints(..., None) path because
            # that would remove other sources' binds from the same target.
            try:
                if old_helper is not None:
                    remove_retarget_constraints(
                        target_armature, old_helper, exclude_helper=new_helper
                    )
                else:
                    remove_retarget_constraints_for_source(
                        target_armature,
                        source_armature.name,
                        helper_name=base_helper_name,
                        exclude_helper=new_helper,
                    )
            except Exception as cleanup_exc:
                print(f"RealMotion Bind old-constraint cleanup warning: {cleanup_exc}")

            if old_helper is not None and old_helper != new_helper:
                try:
                    remove_helper(old_helper)
                    old_helper = None
                except Exception as cleanup_exc:
                    print(f"RealMotion Bind old-helper cleanup warning: {cleanup_exc}")

            # Reclaim the canonical helper name now that the old helper is gone.
            # If an unrelated object owns that name, retain a safe suffix.
            if (
                old_helper is None
                and bpy.data.objects.get(base_helper_name) is None
            ):
                new_helper.name = base_helper_name

            # Remove the pre-bind snapshot if it became unused, and also remove any other
            # Action created during this transaction that ended up truly orphaned. The active
            # target/helper Actions are retained because they still have a datablock user.
            current_action = None
            try:
                current_anim = target_armature.animation_data
                current_action = current_anim.action if current_anim is not None else None
            except (AttributeError, ReferenceError, RuntimeError):
                current_action = None

            if original_target_action is not None and original_target_action != current_action:
                remove_unused_local_action(original_target_action)
            if working_target_action is not None and working_target_action != current_action:
                remove_unused_local_action(working_target_action)
            try:
                bpy.context.view_layer.update()
            except Exception as update_exc:
                print(f"RealMotion Bind final update warning: {update_exc}")
            if before_actions is not None:
                cleanup_new_orphan_actions(before_actions)

            self.report({'INFO'}, "Retarget constraints applied successfully")
            return {'FINISHED'}

        except Exception as exc:
            print(f"RealMotion Bind error: {exc}")
            try:
                object_mode()
            except Exception:
                pass

            # Before commit, remove only constraints explicitly created by this attempt
            # and restore any working Action copy. After commit has started, do not undo
            # the new bind because that can leave the target with no valid bind at all.
            if not commit_started:
                for pb in target_armature.pose.bones:
                    doomed = [con for con in pb.constraints if con in created_constraints]
                    for con in doomed:
                        try:
                            pb.constraints.remove(con)
                        except (RuntimeError, ReferenceError):
                            pass

                try:
                    restore_action_transaction(
                        target_armature,
                        original_target_action,
                        original_target_slot_identifier,
                        working_target_action,
                    )
                except Exception:
                    pass

                if new_helper_name:
                    try:
                        helper_obj = bpy.data.objects.get(new_helper_name)
                    except (AttributeError, RuntimeError, ReferenceError):
                        helper_obj = None
                    if helper_obj is not None:
                        try:
                            remove_helper(helper_obj)
                        except Exception:
                            pass

            self.report({'ERROR'}, f"Bind failed: {exc}")
            return {'CANCELLED'}
        finally:
            # Helper topology may have changed even when the transaction failed before commit.
            # Do not discard the target-selection session merely because Bind touched helper data.
            _clear_spacing_helper_cache()


class UnbindConstraintsOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.unbind_constraints"
    bl_label = "Unbind Constraints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.simple_retarget
        target_armature = props.target_armature

        if not target_armature or target_armature.type != 'ARMATURE':
            self.report({'WARNING'}, "No Target Armature selected")
            return {'CANCELLED'}

        try:
            object_mode()
        except RuntimeError as exc:
            self.report({'ERROR'}, f"Could not enter Object Mode: {exc}")
            return {'CANCELLED'}

        helper_obj = None
        source_armature = props.source_armature
        if source_armature:
            helper_obj = find_helper_for_source(source_armature, target_armature)

        if helper_obj is not None:
            removed_count = remove_retarget_constraints(target_armature, helper_obj)
        elif source_armature is not None:
            # Do not fall back to removing every RM constraint when a specific source
            # helper is missing; that could unbind another source's retarget setup.
            removed_count = remove_retarget_constraints_for_source(
                target_armature, source_armature.name, source_armature.name + RETARGET_ID
            )
        else:
            # Preserve the original no-source workflow: with no source selected,
            # unbind all recognisable RM constraints on the target.
            removed_count = remove_retarget_constraints(target_armature, None)

        helper_removed = False
        if helper_obj is not None:
            try:
                remove_helper(helper_obj)
                helper_removed = True
            except Exception as exc:
                self.report({'WARNING'}, f"Helper cleanup failed: {exc}")

        if helper_removed:
            self.report({'INFO'}, f"Removed helper armature and {removed_count} constraints")
        else:
            self.report({'INFO'}, f"{removed_count} constraints removed")

        _clear_spacing_helper_cache()
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Mapping list operators
# -----------------------------------------------------------------------------

class SetRootBoneOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.set_root_bone"
    bl_label = "Set Root Bone"
    bl_description = "Set the selected bone as the root bone"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.simple_retarget
        index = props.active_bone_index
        if 0 <= index < len(props.bone_mappings):
            source_name = props.bone_mappings[index].source_bone.strip()
            if source_name:
                props.root_bone = source_name
                return {'FINISHED'}
        self.report({'WARNING'}, "Select a mapping row containing a source bone first.")
        return {'CANCELLED'}


class BuildBoneListOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.build_bone_list"
    bl_label = "Build Bone List"
    bl_description = "Replace the current bone mappings with all bones from the selected source armature"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        props = context.scene.simple_retarget
        if len(props.bone_mappings) == 0:
            return self.execute(context)
        return context.window_manager.invoke_confirm(
            self,
            event,
            title="Rebuild Bone List?",
            confirm_text="Replace Mappings",
        )

    def execute(self, context):
        props = context.scene.simple_retarget
        source_armature = props.source_armature

        if not source_armature or source_armature.type != 'ARMATURE':
            self.report({'WARNING'}, "Select a valid Source Armature")
            return {'CANCELLED'}

        props.bone_mappings.clear()
        for bone in source_armature.pose.bones:
            new_mapping = props.bone_mappings.add()
            new_mapping.source_bone = bone.name

        # Normalize the stored root name and clear it if the source no longer contains it.
        root_source = props.root_bone.strip()
        if root_source not in {bone.name for bone in source_armature.data.bones}:
            props.root_bone = ""
        else:
            props.root_bone = root_source

        props.active_bone_index = 0
        return {'FINISHED'}


def get_armature_dim(arm_obj):
    """Rest-pose height (world Z extent) of an armature object."""
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        return 0.0

    world = arm_obj.matrix_world
    heights = []
    for bone in arm_obj.data.bones:
        heights.append((world @ bone.head_local)[2])
        heights.append((world @ bone.tail_local)[2])
    return (max(heights) - min(heights)) if heights else 0.0


class CalculateScaleOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.calculate_scale"
    bl_label = "Calculate Scale"
    bl_description = "Scale the source rig to match the target rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.simple_retarget
        source_rig = props.source_armature
        target_rig = props.target_armature

        if not source_rig or not target_rig:
            self.report({'WARNING'}, "Select both Source and Target Armatures")
            return {'CANCELLED'}
        if source_rig.type != 'ARMATURE' or target_rig.type != 'ARMATURE':
            self.report({'WARNING'}, "Source and Target must both be armatures")
            return {'CANCELLED'}
        if source_rig == target_rig:
            self.report({'WARNING'}, "Source and Target must be different armatures")
            return {'CANCELLED'}

        source_dim = get_armature_dim(source_rig)
        target_dim = get_armature_dim(target_rig)
        if source_dim < 1e-6 or target_dim < 1e-6:
            self.report({'WARNING'}, "Could not measure the rigs (no bones or zero height)")
            return {'CANCELLED'}

        scale_factor = target_dim / source_dim

        # The source object's scale is intentionally controlled here, so remove
        # only its active-action object-scale curves. Guard shared Actions first.
        ensure_single_user_action(source_rig)
        remove_fcurves(source_rig, lambda fc: fc.data_path == "scale")

        source_rig.scale *= scale_factor
        context.view_layer.update()
        return {'FINISHED'}


class AddBoneMappingOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.add_mapping"
    bl_label = "Add Bone Mapping"
    bl_description = "Add a new bone mapping entry"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.simple_retarget
        props.bone_mappings.add()
        props.active_bone_index = len(props.bone_mappings) - 1
        return {'FINISHED'}


class RemoveBoneMappingOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.remove_mapping"
    bl_label = "Remove Bone Mapping"
    bl_description = "Remove the selected bone mapping"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.simple_retarget
        index = props.active_bone_index
        if 0 <= index < len(props.bone_mappings):
            removed_source = props.bone_mappings[index].source_bone
            props.bone_mappings.remove(index)
            props.active_bone_index = (
                max(0, min(index, len(props.bone_mappings) - 1))
                if props.bone_mappings else 0
            )
            if removed_source == props.root_bone:
                # Do not silently choose a new root; require the user to set it.
                props.root_bone = ""
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Spacing (edits the rotation F-Curves of the helper rig)
# -----------------------------------------------------------------------------


def get_rm_fcurve(armature, bone_name, rotation_type, axis):
    if rotation_type not in ('rotation_euler', 'rotation_quaternion'):
        return None

    escaped_name = bpy.utils.escape_identifier(bone_name)
    data_path = f'pose.bones["{escaped_name}"].{rotation_type}'
    for channelbag in get_channelbags(armature):
        fcurve = channelbag.fcurves.find(data_path, index=axis)
        if fcurve is not None:
            return fcurve
    return None


_NO_HELPER = object()


def get_mapped_rm_source_bone(context, target_bone, helper_hint=None):
    props = context.scene.simple_retarget
    source_rig = props.source_armature
    target_rig = props.target_armature

    if not source_rig or not target_rig or target_bone.id_data != target_rig:
        return None, None

    source_bone_name = None
    for mapping in props.bone_mappings:
        if mapping.target_bone.strip() == target_bone.name:
            source_bone_name = mapping.source_bone.strip()
            break

    if not source_bone_name:
        return None, None

    if helper_hint is _NO_HELPER:
        return None, None

    rm_rig = helper_hint
    if rm_rig is None:
        rm_rig = find_helper_for_source(source_rig, target_rig)
    elif (rm_rig.type != 'ARMATURE' or
          rm_rig.parent != source_rig):
        return None, None

    if rm_rig is None or rm_rig.type != 'ARMATURE':
        return None, None

    return rm_rig, rm_rig.pose.bones.get(source_bone_name)


_EULER_PROPS = ("rotation_x", "rotation_y", "rotation_z")
_QUAT_PROPS = ("rotation_w", "rotation_qx", "rotation_qy", "rotation_qz")


# Spacing selection-session state is intentionally runtime-only. It is transient UI
# state and should not dirty or write the .blend merely because the user changes
# bone selection.
_spacing_runtime_sessions = {}
_spacing_helper_cache = {}


def _runtime_object_key(obj):
    try:
        pointer = obj.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError):
        pointer = id(obj)
    try:
        name = obj.name
    except (AttributeError, ReferenceError, RuntimeError):
        name = ""
    return pointer, name


def get_spacing_helper_cached(source_rig, target_rig):
    """Return cached helper lookup, including a safe negative cache for the no-helper case."""
    if source_rig is None or target_rig is None:
        return _NO_HELPER

    key = (_runtime_object_key(source_rig), _runtime_object_key(target_rig))
    if key in _spacing_helper_cache:
        cached = _spacing_helper_cache[key]
        if cached is _NO_HELPER:
            expected_name = source_rig.name + RETARGET_ID
            canonical = bpy.data.objects.get(expected_name)
            canonical_plausible = False
            if canonical is not None:
                try:
                    canonical_plausible = (
                        canonical.type == 'ARMATURE'
                        and canonical.parent == source_rig
                    )
                except (AttributeError, ReferenceError, RuntimeError):
                    canonical_plausible = True

            if not canonical_plausible:
                # Current helpers are parented to the source. Only rescan when a plausible
                # helper child appears; otherwise the negative cache remains valid.
                try:
                    possible_child = any(
                        child.type == 'ARMATURE' and
                        (child.get(HELPER_TAG, False) or
                         child.name.startswith(expected_name))
                        for child in source_rig.children
                    )
                except (AttributeError, ReferenceError, RuntimeError):
                    possible_child = True
                if not possible_child:
                    return _NO_HELPER
        else:
            try:
                if (
                    cached.type == 'ARMATURE'
                    and cached.parent == source_rig
                    and bpy.data.objects.get(cached.name) is cached
                ):
                    return cached
            except (AttributeError, ReferenceError, RuntimeError):
                pass

    found = find_helper_for_source(source_rig, target_rig)
    result = found if found is not None else _NO_HELPER
    _spacing_helper_cache[key] = result
    return result


def _spacing_scene_key(scene):
    """Return a defensive transient identity for a Scene runtime-state entry."""
    try:
        pointer = scene.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError):
        pointer = id(scene)

    try:
        name = scene.name
    except (AttributeError, ReferenceError, RuntimeError):
        name = ""
    return pointer, name


def _spacing_selection_token(target, selected_bones):
    """Return a lightweight runtime identity for the selected target pose bones.

    RNA pointers are stable for the lifetime of their datablocks. Bone names are
    included as a defensive guard against the unlikely case of pointer-address
    recycling after an in-place rig edit.
    """
    try:
        target_key = target.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError):
        target_key = id(target)

    try:
        target_name = target.name
    except (AttributeError, ReferenceError, RuntimeError):
        target_name = ""

    bone_keys = set()
    for bone in selected_bones or ():
        if bone.id_data != target:
            continue
        try:
            bone_key = bone.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError):
            bone_key = id(bone)
        try:
            bone_name = bone.name
        except (AttributeError, ReferenceError, RuntimeError):
            bone_name = ""
        bone_keys.add((bone_key, bone_name))
    return target_key, target_name, frozenset(bone_keys)


def get_spacing_selection_token(context):
    """Return the current target-rig selection identity for spacing state."""
    try:
        retarget_props = context.scene.simple_retarget
        target = getattr(retarget_props, "target_armature", None)
    except (AttributeError, ReferenceError, RuntimeError):
        return None

    if target is None or target.type != 'ARMATURE':
        return None

    try:
        if (context.view_layer.objects.active == target and
                target.mode == 'POSE'):
            return _spacing_selection_token(target, context.selected_pose_bones)
    except (AttributeError, ReferenceError, RuntimeError):
        pass

    return None


def get_scene_spacing_selection_token(scene):
    """Read the active target-rig selection without relying on panel drawing."""
    if scene is None:
        return None

    try:
        retarget_props = scene.simple_retarget
    except (AttributeError, ReferenceError, RuntimeError):
        return None

    target = getattr(retarget_props, "target_armature", None)
    if target is None or target.type != 'ARMATURE':
        return None

    # In normal interactive use Blender already maintains the selected-pose-bone
    # collection on the current context. Prefer it over scanning every pose bone.
    try:
        if (bpy.context.scene == scene and
                bpy.context.view_layer is not None and
                bpy.context.view_layer.objects.active == target and
                target.mode == 'POSE'):
            return _spacing_selection_token(target, bpy.context.selected_pose_bones)
    except (AttributeError, ReferenceError, RuntimeError):
        pass

    # Fallback for handlers invoked without a matching UI context. This is slower,
    # but preserves correctness for scripts/multi-scene situations. If multiple view
    # layers match the same active target, the first matching view layer is sufficient
    # because matching active view layers are expected to expose the same armature
    # pose-bone selection data in this fallback path.
    try:
        view_layers = list(scene.view_layers)
    except (AttributeError, ReferenceError, RuntimeError):
        return None

    for view_layer in view_layers:
        try:
            active = view_layer.objects.active
        except (AttributeError, ReferenceError, RuntimeError):
            continue

        if active != target or target.mode != 'POSE':
            continue

        selected = [
            pb for pb in target.pose.bones
            if pb.bone.select
        ]
        return _spacing_selection_token(target, selected)

    return None


def _get_spacing_runtime_state(scene):
    key = _spacing_scene_key(scene)
    state = _spacing_runtime_sessions.get(key)
    if state is None:
        # The first observation is deliberately baseline-pending. We must first
        # synchronize prev_* with the visible slider values so an absolute UI value
        # can never be mistaken for a delta when a file/session has just been opened.
        state = {"observed_token": None, "baseline_pending": True}
        _spacing_runtime_sessions[key] = state
    return state


def _clear_spacing_helper_cache():
    """Invalidate cached helper objects without resetting target-selection sessions."""
    _spacing_helper_cache.clear()


def _clear_spacing_runtime_state():
    """Reset all transient spacing state used by the runtime handlers/UI."""
    _spacing_runtime_sessions.clear()
    _spacing_helper_cache.clear()


def _remove_spacing_handler(handler_list, handler_name):
    """Remove matching handlers and return whether at least one was removed."""
    removed = False
    for handler in list(handler_list):
        if getattr(handler, "__name__", "") != handler_name:
            continue
        try:
            handler_list.remove(handler)
            removed = True
        except ValueError:
            # Another registration/unregistration path removed it first.
            pass
    return removed


@persistent
def real_motion_spacing_selection_handler(scene, depsgraph):
    """Detect spacing-selection changes even when the Spacing panel is not visible."""
    del depsgraph

    token = get_scene_spacing_selection_token(scene)
    state = _get_spacing_runtime_state(scene)

    # Object-mode / invalid-target states are deliberately ignored. Leaving Pose
    # mode should not by itself consume a spacing-slider edit. A fresh baseline is
    # required only when moving between two concrete target-selection tokens.
    if token is None:
        return

    observed = state["observed_token"]
    if observed is None:
        state["observed_token"] = token
        state["baseline_pending"] = True
        return

    if token != observed:
        state["observed_token"] = token
        state["baseline_pending"] = True


@persistent
def real_motion_spacing_load_post(*_args):
    """Clear transient spacing-session state whenever a new .blend is loaded."""
    _clear_spacing_runtime_state()


_SPACING_VALUE_PROPS = (
    "rotation_x", "rotation_y", "rotation_z",
    "rotation_w", "rotation_qx", "rotation_qy", "rotation_qz",
)


def reset_spacing_previous_values(props, zero=False):
    """Reset slider baselines to zero or to their current visible values."""
    for prop_name in _SPACING_VALUE_PROPS:
        value = 0.0 if zero else getattr(props, prop_name)
        setattr(props, "prev_" + prop_name, value)


def update_rotation_values(context, rotation_type, axis):
    props = context.scene.simple_retarget_spacing
    selected_bones = context.selected_pose_bones
    if not selected_bones:
        return

    # Selection changes are observed by the persistent depsgraph handler so the
    # same-bones-deselect/reselect sequence cannot reuse an old session baseline.
    # The first edit after a detected selection change establishes a fresh baseline;
    # it is intentionally swallowed rather than applied as a delta.
    token = get_spacing_selection_token(context)
    state = _get_spacing_runtime_state(context.scene)

    if token is None:
        return

    if (state["baseline_pending"] or
            state["observed_token"] != token):
        state["observed_token"] = token
        state["baseline_pending"] = False
        reset_spacing_previous_values(props, zero=False)
        return

    if rotation_type == 'rotation_euler':
        prop_name = _EULER_PROPS[axis]
    elif rotation_type == 'rotation_quaternion':
        prop_name = _QUAT_PROPS[axis]
    else:
        return

    current = getattr(props, prop_name)
    previous = getattr(props, "prev_" + prop_name)
    delta = current - previous

    retarget_props = context.scene.simple_retarget
    source_rig = retarget_props.source_armature
    target_rig = retarget_props.target_armature
    helper_cache = _NO_HELPER
    if source_rig and target_rig:
        helper_cache = get_spacing_helper_cached(source_rig, target_rig)

    # get_spacing_helper_cached() returns either the _NO_HELPER sentinel or a helper.
    # The helper is shared by all selected target bones, so isolate its Action once.
    if helper_cache is not _NO_HELPER:
        ensure_single_user_action(helper_cache)

    for bone in selected_bones:
        rm_rig, rm_source_bone = get_mapped_rm_source_bone(
            context, bone, helper_hint=helper_cache
        )
        if rm_rig is None or rm_source_bone is None:
            continue

        fcurve = get_rm_fcurve(rm_rig, rm_source_bone.name, rotation_type, axis)
        if fcurve is None:
            continue

        for kp in fcurve.keyframe_points:
            kp.co[1] += delta
            kp.handle_left[1] += delta
            kp.handle_right[1] += delta
        fcurve.update()

    setattr(props, "prev_" + prop_name, current)
    if context.area:
        context.area.tag_redraw()


class SimpleRetargetSpacingProperties(bpy.types.PropertyGroup):
    rotation_x: bpy.props.FloatProperty(
        name="X Rotation",
        subtype='ANGLE',
        step=0.1,
        update=lambda self, context: update_rotation_values(context, 'rotation_euler', 0),
    )
    rotation_y: bpy.props.FloatProperty(
        name="Y Rotation",
        subtype='ANGLE',
        step=0.1,
        update=lambda self, context: update_rotation_values(context, 'rotation_euler', 1),
    )
    rotation_z: bpy.props.FloatProperty(
        name="Z Rotation",
        subtype='ANGLE',
        step=0.1,
        update=lambda self, context: update_rotation_values(context, 'rotation_euler', 2),
    )
    rotation_w: bpy.props.FloatProperty(
        name="W Rotation",
        step=0.1,
        update=lambda self, context: update_rotation_values(context, 'rotation_quaternion', 0),
    )
    rotation_qx: bpy.props.FloatProperty(
        name="X Quat",
        step=0.1,
        update=lambda self, context: update_rotation_values(context, 'rotation_quaternion', 1),
    )
    rotation_qy: bpy.props.FloatProperty(
        name="Y Quat",
        step=0.1,
        update=lambda self, context: update_rotation_values(context, 'rotation_quaternion', 2),
    )
    rotation_qz: bpy.props.FloatProperty(
        name="Z Quat",
        step=0.1,
        update=lambda self, context: update_rotation_values(context, 'rotation_quaternion', 3),
    )
    prev_rotation_x: bpy.props.FloatProperty()
    prev_rotation_y: bpy.props.FloatProperty()
    prev_rotation_z: bpy.props.FloatProperty()
    prev_rotation_w: bpy.props.FloatProperty()
    prev_rotation_qx: bpy.props.FloatProperty()
    prev_rotation_qy: bpy.props.FloatProperty()
    prev_rotation_qz: bpy.props.FloatProperty()


class SIMPLERETARGET_PT_SpacingPanel(bpy.types.Panel):
    bl_label = "Spacing"
    bl_idname = "SIMPLERETARGET_PT_SpacingPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RealMotion Pro'
    bl_order = 1
    bl_parent_id = "VIEW3D_PT_simple_retarget"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.simple_retarget_spacing
        retarget_props = scene.simple_retarget
        selected_bones = context.selected_pose_bones

        if not selected_bones:
            layout.label(text="Select target rig bones")
            return

        source_rig = retarget_props.source_armature
        target_rig = retarget_props.target_armature
        helper_cache = _NO_HELPER
        if source_rig and target_rig:
            helper_cache = get_spacing_helper_cached(source_rig, target_rig)

        has_valid_mapping = False
        rotation_modes = set()
        for bone in selected_bones:
            rm_rig, rm_source_bone = get_mapped_rm_source_bone(
                context, bone, helper_hint=helper_cache
            )
            if rm_rig is not None and rm_source_bone is not None:
                has_valid_mapping = True
                rotation_modes.add(rm_source_bone.rotation_mode)

        if not has_valid_mapping:
            layout.label(text="No valid bone mappings found in _RM rig.")
            return

        layout.label(text="First slider move after opening/changing selection sets baseline.", icon='INFO')

        if 'QUATERNION' in rotation_modes:
            layout.label(text="Adjust Quaternion F-Curves for mapped _RM bones:")
            layout.prop(props, "rotation_w")
            layout.prop(props, "rotation_qx")
            layout.prop(props, "rotation_qy")
            layout.prop(props, "rotation_qz")

        if len(rotation_modes) > 1 or 'QUATERNION' not in rotation_modes:
            layout.label(text="Adjust Euler F-Curves for mapped _RM bones:")
            layout.prop(props, "rotation_x")
            layout.prop(props, "rotation_y")
            layout.prop(props, "rotation_z")


# -----------------------------------------------------------------------------
# Bake
# -----------------------------------------------------------------------------


def ensure_action_for_bake(id_block):
    """Ensure the object has an active slotted Action suitable for nla.bake."""
    anim = id_block.animation_data_create()

    if anim.action is None:
        action = bpy.data.actions.new(f"{id_block.name}_RetargetBake")
        slot = action.slots.new('OBJECT', id_block.name)
        layer = action.layers.new("RealMotion Bake")
        strip = layer.strips.new(type='KEYFRAME')
        strip.channelbag(slot, ensure=True)
        anim.action = action
        anim.action_slot = slot
        return action

    if anim.action_slot is None:
        suitable = list(getattr(anim, "action_suitable_slots", []))
        if suitable:
            anim.action_slot = suitable[0]
        elif hasattr(anim.action, "slots"):
            anim.action_slot = anim.action.slots.new('OBJECT', id_block.name)

    slot = anim.action_slot
    if slot is not None:
        # Ensure there is at least one keyframe strip/channelbag for the assigned slot.
        channelbag = None
        for layer in anim.action.layers:
            for strip in layer.strips:
                if strip.type == 'KEYFRAME':
                    channelbag = strip.channelbag(slot, ensure=True)
                    break
            if channelbag is not None:
                break
        if channelbag is None:
            layer = anim.action.layers.new("RealMotion Bake")
            strip = layer.strips.new(type='KEYFRAME')
            strip.channelbag(slot, ensure=True)

    return anim.action


def get_bake_range(source_armature):
    """Determine a safe integer bake range from the source Action Slot."""
    key_range = get_keyframe_range(source_armature)
    if key_range is None:
        return None, None, "Source animation has no keyframes in the active Action Slot."

    source_start = math.floor(key_range[0])
    source_end = math.ceil(key_range[1])

    if source_start < 0:
        return None, None, (
            "Source animation starts before frame 0. Blender's nla.bake "
            "operator does not accept a negative frame_start. Move the source "
            "animation to frame 0 or later before baking."
        )
    if source_end > 300000:
        return None, None, (
            "Source animation extends beyond frame 300000. Blender's nla.bake "
            "operator does not accept frame_end values above 300000."
        )

    frame_start = max(0, source_start)
    frame_end = max(frame_start, source_end)
    if frame_end < 1:
        frame_end = 1

    return frame_start, frame_end, None


def bake_animation(context=None):
    """Bake constrained target bones to keyframes and clean only RM data."""
    context = context or bpy.context

    # Validation is intentionally outside the broad operational exception handler so
    # programmer errors in basic property access are not silently presented as bake failures.
    props = context.scene.simple_retarget
    source_armature = props.source_armature
    target_armature = props.target_armature

    if not source_armature or source_armature.type != 'ARMATURE':
        return False, "Source armature not set!"
    if not target_armature or target_armature.type != 'ARMATURE':
        return False, "Target armature not set!"
    if source_armature == target_armature:
        return False, "Source and Target must be different armatures."
    if not target_armature.visible_get():
        return False, "Target armature must be visible in the viewport."

    frame_start, frame_end, range_error = get_bake_range(source_armature)
    if range_error:
        return False, range_error

    # Validation returns above occur before helper topology or animation data are
    # modified, so the spacing helper cache remains valid on those paths.
    try:
        helper = find_helper_for_source(source_armature, target_armature)
        if helper is None:
            return False, "Retarget helper rig was not found. Bind constraints again before baking."

        ensure_single_user_action(target_armature)
        ensure_action_for_bake(target_armature)

        driven_bone_names = {
            pb.name
            for pb in target_armature.pose.bones
            if any(is_retarget_constraint(con, helper) for con in pb.constraints)
        }
        if not driven_bone_names:
            return False, "No retarget constraints found on the target rig - bind first."

        # nla.bake works on selected objects and (with only_selected) selected bones.
        object_mode()
        select_only(target_armature)
        bpy.ops.object.mode_set(mode='POSE')

        for pb in target_armature.pose.bones:
            pb.bone.select = pb.name in driven_bone_names

        # Do NOT use clear_constraints=True: Blender removes all constraints from
        # keyed bones, not just this add-on's constraints.
        result = bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            only_selected=True,
            visual_keying=True,
            clear_constraints=False,
            use_current_action=True,
            clean_curves=True,
            bake_types={'POSE'},
        )
        if 'FINISHED' not in result:
            object_mode()
            return False, "Bake was cancelled - no keyframes were created."

        anim = target_armature.animation_data
        if anim and anim.action:
            anim.action.name = f"{source_armature.name}_Bake"
            print(f"Renamed baked action to: {anim.action.name}")

        # Remove only RM constraints after visual baking has completed.
        removed_count = remove_retarget_constraints(target_armature, helper)
        object_mode()

        # Remove only the helper associated with this source rig.
        remove_helper(helper)

        print(
            f"Baking and cleanup complete! {len(driven_bone_names)} driven bones, "
            f"{removed_count} constraints removed."
        )
        return True, "Baking and cleanup complete!"

    except Exception as exc:
        try:
            object_mode()
        except Exception:
            pass
        print(f"RealMotion Bake error: {exc}")
        return False, f"Bake failed: {exc}"
    finally:
        # Bake may remove or recreate helpers. Invalidate only the helper lookup cache;
        # the target-selection baseline is independent of helper topology.
        _clear_spacing_helper_cache()


class SIMPLE_RETARGET_PT_BakePanel(bpy.types.Panel):
    bl_label = "Bake Animation"
    bl_idname = "SIMPLE_RETARGET_PT_BakePanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RealMotion Pro"
    bl_order = 2
    bl_parent_id = "VIEW3D_PT_simple_retarget"

    def draw(self, context):
        layout = self.layout
        layout.operator("simple_retarget.bake_animation", text="Bake")


class SIMPLE_RETARGET_OT_BakeOperator(bpy.types.Operator):
    bl_idname = "simple_retarget.bake_animation"
    bl_label = "Bake Animation"
    bl_description = "Bakes animation from source to target rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ok, message = bake_animation(context)
        self.report({'INFO'} if ok else {'WARNING'}, message)
        return {'FINISHED'} if ok else {'CANCELLED'}


# -----------------------------------------------------------------------------
# FBX import
# -----------------------------------------------------------------------------

class SimpleRetargetImportFbx(Operator, ImportHelper):
    """Import an FBX file and normalize static imported armature transforms without corrupting pose animation."""
    bl_idname = "simple_retarget.import_fbx"
    bl_label = "Import FBX"
    bl_options = {'PRESET', 'UNDO'}
    filename_ext = ".fbx"

    filter_glob: StringProperty(
        default="*.fbx",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context):
        # Snapshot scene objects so we can identify what this import actually added.
        before_objects = set(bpy.data.objects)

        try:
            import_fbx_file(self.filepath)
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        new_objects = [obj for obj in bpy.data.objects if obj not in before_objects]
        new_armatures = [obj for obj in new_objects if obj.type == 'ARMATURE']

        imported_object = None
        active = context.view_layer.objects.active
        if active in new_armatures:
            imported_object = active
        elif new_armatures:
            # Prefer a selected newly imported armature, otherwise the first new one.
            imported_object = next(
                (obj for obj in context.selected_objects if obj in new_armatures),
                new_armatures[0],
            )

        if imported_object is None:
            self.report({'WARNING'}, "No newly imported armature found. No transform changes were made.")
            return {'FINISHED'}

        try:
            object_mode()
            select_only(imported_object)

            # Reset pose-bone transforms independently of object transforms.
            reset_pose(imported_object)

            # Blender's armature transform_apply does not rescale pose-bone animation curves.
            # Compensate pose-location keys by the static imported object scale before applying it,
            # so root/hip translation keeps the same world-space amplitude after normalization.
            applied, skipped_scale, warning = normalize_imported_armature_transforms(imported_object)
            if warning:
                self.report({'WARNING'}, warning)
                return {'FINISHED'}

            if applied:
                self.report(
                    {'INFO'},
                    "Imported armature transforms normalized; static object scale was compensated in pose-location keys.",
                )
            elif skipped_scale:
                self.report({'WARNING'}, "Imported armature scale normalization was skipped.")
            else:
                self.report({'INFO'}, "Imported armature transforms were already normalized; no transform changes were needed.")
            return {'FINISHED'}

        except (RuntimeError, ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Imported armature transform setup failed: {exc}")
            return {'CANCELLED'}


def menu_func_import(self, context):
    self.layout.operator(
        SimpleRetargetImportFbx.bl_idname,
        text="RealMotion Pro FBX Import (.fbx)",
    )


classes = [
    BoneMappingItem,
    RetargetProperties,
    SIMPLE_RETARGET_UL_BoneMappingList,
    SavePresetOperator,
    LoadPresetOperator,
    SIMPLE_RETARGET_OT_HighlightBone,
    SimpleRetargetPanel,
    BindConstraintsOperator,
    UnbindConstraintsOperator,
    SetRootBoneOperator,
    BuildBoneListOperator,
    CalculateScaleOperator,
    AddBoneMappingOperator,
    RemoveBoneMappingOperator,
    SimpleRetargetSpacingProperties,
    SIMPLERETARGET_PT_SpacingPanel,
    SIMPLE_RETARGET_PT_BakePanel,
    SIMPLE_RETARGET_OT_BakeOperator,
    SimpleRetargetImportFbx,
]


def _remove_scene_property_if_present(name):
    try:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
            return True
    except (AttributeError, RuntimeError):
        pass
    return False


_REGISTRATION_OWNER = "RealMotionRetargetStandalone"
_REGISTRATION_OWNER_ATTR = "_real_motion_registration_owner"


def _mark_registration_classes():
    """Tag this module's Python classes so same-name classes from other add-ons are ignored."""
    for cls in classes:
        try:
            setattr(cls, _REGISTRATION_OWNER_ATTR, _REGISTRATION_OWNER)
        except Exception:
            pass


def _registered_type(cls):
    """Return this add-on's registered class with the same RNA/Python name, if any.

    A same-name class belonging to another add-on is deliberately treated as a foreign
    collision and is never unregistered by this module.
    """
    try:
        candidate = getattr(bpy.types, cls.__name__, None)
    except (AttributeError, RuntimeError):
        return None

    if candidate is None:
        return None
    if candidate is cls:
        return candidate

    try:
        if getattr(candidate, _REGISTRATION_OWNER_ATTR, None) == _REGISTRATION_OWNER:
            return candidate
    except (AttributeError, RuntimeError, ReferenceError):
        pass

    # Compatibility with classes registered by older RealMotion versions before the
    # ownership marker existed. Versioned source files can have different module names
    # during Reload Scripts (for example realmotion_retarget_v1_68 -> realmotion_retarget_v1_70),
    # so compare the module basename against the RealMotion module family instead of
    # requiring an exact module-name match.
    try:
        candidate_module = getattr(candidate, "__module__", "")
        cls_module = getattr(cls, "__module__", "")
        candidate_basename = candidate_module.rsplit(".", 1)[-1]
        cls_basename = cls_module.rsplit(".", 1)[-1]

        # Legacy RealMotion releases were often renamed/copied by users. Current
        # releases carry an explicit owner marker, but older registrations do not.
        # Accept the known RealMotion module family here so upgrades such as
        # realmotion_retarget_v1_68 -> realmotion_retarget or
        # realmotion_retarget_venkatesh can still replace stale legacy classes.
        # The exact-class identity check in register() remains mandatory for the
        # preservation fast path, so legacy classes never make a new module appear
        # fully registered.
        if (
            candidate_basename.startswith("realmotion_retarget")
            and cls_basename.startswith("realmotion_retarget")
        ):
            return candidate
    except (AttributeError, RuntimeError, ReferenceError):
        pass

    return None


def _foreign_registration_collisions():
    """Return same-name bpy.types classes that are not owned by this add-on."""
    collisions = []
    for cls in classes:
        try:
            candidate = getattr(bpy.types, cls.__name__, None)
        except (AttributeError, RuntimeError):
            continue
        if candidate is None or _registered_type(cls) is candidate:
            continue
        collisions.append((cls, candidate))
    return collisions


def register():
    """Register the add-on safely across normal and partial script reloads."""
    _mark_registration_classes()
    _clear_spacing_runtime_state()

    _remove_spacing_handler(
        bpy.app.handlers.depsgraph_update_post,
        "real_motion_spacing_selection_handler",
    )
    _remove_spacing_handler(
        bpy.app.handlers.load_post,
        "real_motion_spacing_load_post",
    )
    _remove_spacing_handler(
        bpy.app.handlers.load_post,
        "realmotion_spacing_load_post",
    )
    try:
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    except (AttributeError, RuntimeError, ValueError):
        pass

    collisions = _foreign_registration_collisions()
    if collisions:
        names = ", ".join(cls.__name__ for cls, _ in collisions)
        raise RuntimeError(
            "RealMotion Retarget cannot register because another add-on already owns "
            f"these Blender RNA class names: {names}."
        )

    # Only take the preservation fast path when every currently registered RNA class is
    # literally this module's current Python class object. Older RealMotion versions may
    # be recognized by _registered_type() for replacement during Reload Scripts, but they
    # must not make the new module look fully registered.
    all_registered = all(_registered_type(cls) is cls for cls in classes)
    if all_registered:
        # Common re-entry path: keep PropertyGroups/mappings intact and only reinstall
        # transient handlers/menu entries.
        try:
            if not hasattr(bpy.types.Scene, "simple_retarget"):
                bpy.types.Scene.simple_retarget = bpy.props.PointerProperty(type=RetargetProperties)
            if not hasattr(bpy.types.Scene, "simple_retarget_spacing"):
                bpy.types.Scene.simple_retarget_spacing = bpy.props.PointerProperty(type=SimpleRetargetSpacingProperties)
        except (AttributeError, RuntimeError):
            pass

        bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
        bpy.app.handlers.depsgraph_update_post.append(real_motion_spacing_selection_handler)
        bpy.app.handlers.load_post.append(real_motion_spacing_load_post)
        return

    # Partial reload: remove every currently registered class that belongs to this add-on,
    # including classes where the registered object is literally the same Python class.
    # A stale Scene PropertyGroup instance cannot safely be preserved here: Blender's API
    # requires the PropertyGroup class to be registered before the property is assigned,
    # and class definitions themselves are not stored in the .blend. The fully-registered
    # fast path above is therefore the only path that promises to preserve live mappings.
    # This is the case that previously allowed register_class() to receive an already-
    # registered class and raise RuntimeError.
    _remove_scene_property_if_present("simple_retarget_spacing")
    _remove_scene_property_if_present("simple_retarget")

    owned_registered = []
    for cls in classes:
        registered = _registered_type(cls)
        if registered is not None:
            owned_registered.append(registered)

    for registered in reversed(owned_registered):
        try:
            bpy.utils.unregister_class(registered)
        except (RuntimeError, ValueError):
            pass

    registered_now = []
    try:
        for cls in classes:
            bpy.utils.register_class(cls)
            registered_now.append(cls)
    except (RuntimeError, ValueError) as exc:
        for registered in reversed(registered_now):
            try:
                bpy.utils.unregister_class(registered)
            except (RuntimeError, ValueError):
                pass
        raise RuntimeError(f"RealMotion Retarget registration failed: {exc}") from exc

    bpy.types.Scene.simple_retarget = bpy.props.PointerProperty(type=RetargetProperties)
    bpy.types.Scene.simple_retarget_spacing = bpy.props.PointerProperty(type=SimpleRetargetSpacingProperties)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.app.handlers.depsgraph_update_post.append(real_motion_spacing_selection_handler)
    bpy.app.handlers.load_post.append(real_motion_spacing_load_post)


def unregister():
    # Clear transient state before any RNA/class teardown so even a partial failure
    # cannot leave stale spacing-session data for the next registration attempt.
    _clear_spacing_runtime_state()

    _remove_spacing_handler(
        bpy.app.handlers.depsgraph_update_post,
        "real_motion_spacing_selection_handler",
    )
    _remove_spacing_handler(
        bpy.app.handlers.load_post,
        "real_motion_spacing_load_post",
    )
    _remove_spacing_handler(
        bpy.app.handlers.load_post,
        "realmotion_spacing_load_post",
    )
    try:
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    except (AttributeError, RuntimeError, ValueError):
        pass
    _remove_scene_property_if_present("simple_retarget_spacing")
    _remove_scene_property_if_present("simple_retarget")
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass


if __name__ == "__main__":
    register()
