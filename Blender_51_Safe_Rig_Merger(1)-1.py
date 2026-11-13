import bpy
from bpy_extras import anim_utils
from mathutils import Matrix


# =============================================================================
# Blender 5.1 - Conservative Armature + Mesh Merger
#
# Usage:
#   1. Save a backup of your .blend file.
#   2. Object Mode: select all armatures and the meshes they control.
#   3. Make the armature that should survive the merge ACTIVE.
#   4. Run this script from the Text Editor.
#
# Important:
#   - Each mesh must be controlled by at most ONE selected armature.
#   - Armatures sharing the same Armature datablock are rejected.
#   - Rig-specific constraints, drivers, custom properties, bone collections,
#     custom shapes, and external references require manual post-merge checks.
#   - The script does not apply transforms or alter mesh vertex coordinates
#     directly; Blender's Join operator converts joined data into the survivor's
#     local space.
# =============================================================================


PREFIX_SEPARATOR = "_"
COMBINED_ACTION_NAME = "MERGED_Combined_Action"


def log(message):
    print("[Safe Rig Merger] " + str(message))


def unique_name(collection, base):
    if base not in collection:
        return base
    i = 1
    while f"{base}.{i:03d}" in collection:
        i += 1
    return f"{base}.{i:03d}"


def deselect_all():
    for obj in bpy.context.view_layer.objects:
        if obj.select_get():
            obj.select_set(False)


def find_slot(action, identifier):
    if not action:
        return None
    for slot in action.slots:
        if slot.identifier == identifier:
            return slot
    return None


def selected_armature_for_mesh(mesh, armatures):
    """Find the one selected rig that controls this mesh."""
    found = []

    for modifier in mesh.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object in armatures:
            if modifier.object not in found:
                found.append(modifier.object)

    if mesh.parent in armatures and mesh.parent not in found:
        found.append(mesh.parent)

    if len(found) > 1:
        raise RuntimeError(
            f'Mesh "{mesh.name}" is linked to more than one selected armature. '
            "This script will not guess which rig owns its weights."
        )

    return found[0] if found else None


def build_bone_mapping(armature):
    """Build old-name -> unique-prefixed-name mapping."""
    mapping = {}
    prefix = armature.name + PREFIX_SEPARATOR

    for bone in armature.data.bones:
        old = bone.name
        new = prefix + old
        mapping[old] = new

    return mapping


def rename_vertex_groups(mesh, mapping):
    """Rename only groups matching bones from this mesh's original rig."""
    jobs = []
    existing = {group.name for group in mesh.vertex_groups}

    for group in mesh.vertex_groups:
        new_name = mapping.get(group.name)
        if new_name and new_name != group.name:
            if new_name in existing:
                raise RuntimeError(
                    f'Mesh "{mesh.name}" already has vertex group "{new_name}". '
                    "Rename the conflicting group manually before merging."
                )
            jobs.append((group, new_name))

    # Temporary names avoid collisions during the rename operation.
    for index, (group, _) in enumerate(jobs):
        group.name = f"__RIG_MERGE_TEMP_{index:05d}__"

    for group, new_name in jobs:
        group.name = new_name


def remap_bone_path(data_path, mapping):
    """Remap pose.bones['OldName'] path segments to the new bone names."""
    result = data_path

    for old_name, new_name in mapping.items():
        old_escaped = bpy.utils.escape_identifier(old_name)
        new_escaped = bpy.utils.escape_identifier(new_name)

        old_token = f'pose.bones["{old_escaped}"]'
        new_token = f'pose.bones["{new_escaped}"]'

        result = result.replace(old_token, new_token)

    return result


def copy_fcurve(src, dst_channelbag, mapping):
    """Copy keyframes and common F-Curve settings into a destination channelbag."""
    data_path = remap_bone_path(src.data_path, mapping)
    index = src.array_index

    if dst_channelbag.fcurves.find(data_path, index=index) is not None:
        raise RuntimeError(
            f"Animation channel collision: {data_path}[{index}]. "
            "The source actions contain overlapping channels."
        )

    group_name = src.group.name if src.group else ""
    dst = dst_channelbag.fcurves.new(
        data_path, index=index, group_name=group_name
    )

    dst.extrapolation = src.extrapolation
    dst.auto_smoothing = src.auto_smoothing
    dst.mute = src.mute
    dst.lock = src.lock
    dst.select = src.select

    count = len(src.keyframe_points)
    if count:
        dst.keyframe_points.add(count)

        for source_key, target_key in zip(
            src.keyframe_points, dst.keyframe_points
        ):
            target_key.co = source_key.co
            target_key.handle_left = source_key.handle_left
            target_key.handle_right = source_key.handle_right
            target_key.interpolation = source_key.interpolation
            target_key.easing = source_key.easing
            target_key.type = source_key.type
            target_key.handle_left_type = source_key.handle_left_type
            target_key.handle_right_type = source_key.handle_right_type

            # Bezier easing parameters, where applicable.
            for attr in ("back", "amplitude", "period"):
                if hasattr(source_key, attr) and hasattr(target_key, attr):
                    try:
                        setattr(target_key, attr, getattr(source_key, attr))
                    except (AttributeError, TypeError, ValueError):
                        pass

    # Copy F-Curve modifiers and their writable RNA settings.
    # Some modifier settings are read-only or context-dependent; those are skipped.
    for source_modifier in src.modifiers:
        try:
            target_modifier = dst.modifiers.new(source_modifier.type)
        except Exception as exc:
            log(f"Could not copy F-Curve modifier {source_modifier.type}: {exc}")
            continue

        for prop in source_modifier.bl_rna.properties:
            identifier = prop.identifier
            if identifier in {"rna_type", "type"} or prop.is_readonly:
                continue
            try:
                value = getattr(source_modifier, identifier)
                setattr(target_modifier, identifier, value)
            except Exception:
                pass

    dst.update()


def get_action_channelbag(action, slot):
    if not action or not slot:
        return None
    return anim_utils.action_get_channelbag_for_slot(action, slot)


def collect_animation_sources(armatures, bone_maps):
    """Capture source action/slot/F-Curve data before joining the armatures."""
    sources = []

    for armature in armatures:
        anim_data = armature.animation_data
        if not anim_data or not anim_data.action:
            log(f'{armature.name}: no active action; animation skipped.')
            continue

        action = anim_data.action
        slot = anim_data.action_slot

        if slot is None:
            log(f'{armature.name}: active action has no slot; animation skipped.')
            continue

        channelbag = get_action_channelbag(action, slot)
        if channelbag is None:
            log(f'{armature.name}: no channelbag for active slot; skipped.')
            continue

        sources.append({
            "armature": armature,
            "mapping": bone_maps[armature],
            "action": action,
            "slot_identifier": slot.identifier,
            "frame_start": action.frame_start,
            "frame_end": action.frame_end,
            "use_frame_range": action.use_frame_range,
            "fcurves": list(channelbag.fcurves),
        })

    return sources


def create_combined_action(sources, survivor):
    if not sources:
        return None

    action = bpy.data.actions.new(
        name=unique_name(bpy.data.actions, COMBINED_ACTION_NAME)
    )
    action.use_fake_user = True

    slot = action.slots.new(id_type='OBJECT', name=survivor.name)
    channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)

    frame_starts = []
    frame_ends = []

    for source in sources:
        if source["use_frame_range"]:
            frame_starts.append(source["frame_start"])
            frame_ends.append(source["frame_end"])

        for fcurve in source["fcurves"]:
            copy_fcurve(fcurve, channelbag, source["mapping"])

    if frame_starts:
        action.use_frame_range = True
        action.frame_start = min(frame_starts)
        action.frame_end = max(frame_ends)

    return action, slot


def validate_context():
    if bpy.app.version < (5, 1, 0):
        raise RuntimeError("This script is intended for Blender 5.1 or newer.")

    if bpy.context.mode != 'OBJECT':
        raise RuntimeError("Switch to Object Mode before running this script.")

    selected = list(bpy.context.selected_objects)
    armatures = [obj for obj in selected if obj.type == 'ARMATURE']
    meshes = [obj for obj in selected if obj.type == 'MESH']

    if len(armatures) < 2:
        raise RuntimeError("Select at least two armature objects.")

    if not meshes:
        log("No meshes selected. Only the armatures will be merged.")

    active = bpy.context.view_layer.objects.active
    survivor = active if active in armatures else armatures[0]

    # Joining two objects that share the same armature datablock can duplicate
    # or ambiguously merge the same bone data. Reject this instead of guessing.
    data_ids = [arm.data.as_pointer() for arm in armatures]
    if len(data_ids) != len(set(data_ids)):
        raise RuntimeError(
            "Two or more selected armature objects share the same Armature "
            "datablock. Make their data single-user or select only one object."
        )

    return selected, armatures, meshes, survivor


def main():
    selected_before = list(bpy.context.selected_objects)
    active_before = bpy.context.view_layer.objects.active

    try:
        selected, armatures, meshes, survivor = validate_context()

        log(f"Survivor armature: {survivor.name}")

        # Build the original mesh -> rig relationships before modifying anything.
        mesh_rigs = {}
        for mesh in meshes:
            mesh_rigs[mesh] = selected_armature_for_mesh(mesh, armatures)

        # Build unique bone names and mappings before changing bone names.
        bone_maps = {
            armature: build_bone_mapping(armature)
            for armature in armatures
        }

        # Record world matrices before detaching/joining.
        mesh_world = {mesh: mesh.matrix_world.copy() for mesh in meshes}

        # Capture source actions and their curves before joining objects.
        animation_sources = collect_animation_sources(armatures, bone_maps)

        # Rename bones and matching mesh vertex groups.
        for armature in armatures:
            mapping = bone_maps[armature]

            for bone in armature.data.bones:
                bone.name = mapping[bone.name]

        for mesh, rig in mesh_rigs.items():
            if rig is not None:
                rename_vertex_groups(mesh, bone_maps[rig])

        # Detach meshes while retaining their world transforms.
        for mesh in meshes:
            mesh.parent = None
            mesh.matrix_world = mesh_world[mesh]

            # Remove only armature modifiers that point to selected rigs.
            for modifier in list(mesh.modifiers):
                if (
                    modifier.type == 'ARMATURE'
                    and modifier.object in armatures
                ):
                    mesh.modifiers.remove(modifier)

        # Join meshes. Blender transforms joined geometry into the active
        # survivor mesh's local space and keeps its materials/UVs/shape keys
        # subject to Blender's normal Join compatibility rules.
        joined_mesh = None
        if meshes:
            active_mesh = (
                active_before
                if active_before in meshes
                else meshes[0]
            )

            deselect_all()
            for mesh in meshes:
                mesh.select_set(True)

            bpy.context.view_layer.objects.active = active_mesh
            bpy.ops.object.join()

            joined_mesh = active_mesh
            joined_mesh.name = unique_name(
                bpy.data.objects, survivor.name + "_JoinedMesh"
            )

        # Join armatures. The survivor's object transform is retained;
        # Blender converts the other armature data into its local space.
        deselect_all()
        for armature in armatures:
            armature.select_set(True)

        bpy.context.view_layer.objects.active = survivor
        bpy.ops.object.join()

        survivor.name = unique_name(
            bpy.data.objects, survivor.name + "_CombinedRig"
        )

        # Build and assign a new action after the join.
        combined = create_combined_action(animation_sources, survivor)

        if combined:
            combined_action, combined_slot = combined
            anim_data = (
                survivor.animation_data
                or survivor.animation_data_create()
            )
            anim_data.action = combined_action
            anim_data.action_slot = combined_slot

        # Attach the joined mesh while preserving its world transform.
        if joined_mesh:
            world = joined_mesh.matrix_world.copy()

            joined_mesh.parent = survivor
            joined_mesh.parent_type = 'OBJECT'
            joined_mesh.parent_bone = ""
            joined_mesh.matrix_world = world

            arm_modifier = joined_mesh.modifiers.new(
                name="Combined Armature",
                type='ARMATURE'
            )
            arm_modifier.object = survivor
            arm_modifier.use_vertex_groups = True
            arm_modifier.use_bone_envelopes = False

        # Warn about external references that may still point to old rigs.
        for obj in bpy.data.objects:
            anim_data = obj.animation_data
            if not anim_data:
                continue

            for driver in anim_data.drivers:
                for variable in driver.driver.variables:
                    for target in variable.targets:
                        if target.id in armatures and target.id != survivor:
                            log(
                                f'WARNING: driver on "{obj.name}" still targets '
                                f'"{target.id.name}". Review manually.'
                            )

        # Restore a useful selection state.
        deselect_all()
        survivor.select_set(True)
        if joined_mesh:
            joined_mesh.select_set(True)
        bpy.context.view_layer.objects.active = survivor

        try:
            bpy.ops.ed.undo_push(message="Safe Armature and Mesh Merge")
        except Exception:
            pass

        log("Merge completed.")
        log("Check the character in Pose Mode and inspect all deformations.")
        log("Review constraints, drivers, custom shapes and external rig links.")

    except Exception as exc:
        log("MERGE STOPPED: " + str(exc))
        log(
            "If any changes occurred before the error, use Ctrl+Z immediately "
            "or reopen your saved backup."
        )
        raise


if __name__ == "__main__":
    main()
