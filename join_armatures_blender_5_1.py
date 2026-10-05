bl_info = {
    "name": "Join Armatures - Keep Animation (Blender 5.1)",
    "author": "OpenAI",
    "version": (2, 0, 0),
    "blender": (5, 1, 0),
    "location": "Text Editor > Run Script",
    "description": "Joins selected armatures and meshes while preserving bone animation and skin weights.",
    "category": "Animation",
}

import bpy
import re

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
PREFIX_BONES = True
PREFIX_VERTEX_GROUPS = True
JOIN_MESHES = True
RENAME_JOINED_OBJECT = True
KEEP_SOURCE_ACTIONS = True


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def unique_object_name(base_name):
    if base_name not in bpy.data.objects:
        return base_name
    i = 1
    while f"{base_name}.{i:03d}" in bpy.data.objects:
        i += 1
    return f"{base_name}.{i:03d}"


def safe_copy(value):
    try:
        return value.copy()
    except Exception:
        return value


def get_selected_objects():
    return list(bpy.context.selected_objects)


def get_source_mesh_armature(mesh):
    """Return the armature object used by this mesh, when detectable."""
    for mod in mesh.modifiers:
        if mod.type == 'ARMATURE' and mod.object and mod.object.type == 'ARMATURE':
            return mod.object
    if mesh.parent and mesh.parent.type == 'ARMATURE':
        return mesh.parent
    return None


def rename_vertex_groups_for_armature(mesh, bone_map):
    if not PREFIX_VERTEX_GROUPS:
        return 0
    renamed = 0
    for vg in mesh.vertex_groups:
        new_name = bone_map.get(vg.name)
        if new_name and new_name != vg.name:
            vg.name = new_name
            renamed += 1
    return renamed


def rewrite_pose_bone_paths(data_path, bone_map):
    """Rewrite pose.bones[\"Bone\"] segments after bone renaming."""
    if not bone_map or not data_path:
        return data_path

    pattern = re.compile(r'pose\\.bones\\[(?:"([^"]*)"|\'([^\']*)\')\\]')

    def replace(match):
        old_name = match.group(1) if match.group(1) is not None else match.group(2)
        new_name = bone_map.get(old_name)
        if new_name is None:
            return match.group(0)
        escaped = new_name.replace('\\', '\\\\').replace('"', '\\"')
        return f'pose.bones["{escaped}"]'

    return pattern.sub(replace, data_path)


def get_action_fcurves(action):
    """Yield source F-Curves from Blender 5.1 actions and older legacy actions."""
    if action is None:
        return

    # Blender 5.x slotted/layered action system.
    layers = getattr(action, "layers", None)
    if layers and len(layers) > 0:
        for layer in layers:
            for strip in layer.strips:
                channelbags = getattr(strip, "channelbags", None)
                if channelbags:
                    for channelbag in channelbags:
                        for fcurve in channelbag.fcurves:
                            yield fcurve, channelbag
        return

    # Legacy compatibility path.
    fcurves = getattr(action, "fcurves", None)
    if fcurves:
        for fcurve in fcurves:
            yield fcurve, None


def copy_fcurve_settings(src, dst):
    """Copy non-keyframe F-Curve settings where Blender exposes them."""
    attrs = (
        "extrapolation",
        "color_mode",
        "auto_smoothing",
        "mute",
        "lock",
        "select",
        "hide",
    )
    for attr in attrs:
        try:
            setattr(dst, attr, getattr(src, attr))
        except Exception:
            pass

    # Copy modifiers.
    try:
        for src_mod in src.modifiers:
            dst_mod = dst.modifiers.new(src_mod.type)
            for prop in src_mod.bl_rna.properties:
                name = prop.identifier
                if name in {"rna_type", "type"} or prop.is_readonly:
                    continue
                try:
                    setattr(dst_mod, name, safe_copy(getattr(src_mod, name)))
                except Exception:
                    pass
    except Exception:
        pass


def copy_keyframes(src, dst):
    """Copy keyframe points including interpolation and handles."""
    # new_from_fcurve may already copy keyframes; this function is used only
    # for legacy/manual creation paths.
    for kp in src.keyframe_points:
        new_kp = dst.keyframe_points.insert(
            frame=kp.co.x,
            value=kp.co.y,
            options={'FAST'},
        )
        try:
            new_kp.interpolation = kp.interpolation
        except Exception:
            pass
        try:
            new_kp.handle_left_type = kp.handle_left_type
            new_kp.handle_right_type = kp.handle_right_type
            new_kp.handle_left = kp.handle_left
            new_kp.handle_right = kp.handle_right
        except Exception:
            pass
    try:
        dst.update()
    except Exception:
        pass


def create_combined_action(target_armature, source_actions, bone_maps):
    """Create one Blender 5.1 action containing all source bone animation."""
    action_name = f"{target_armature.name}_JOINED_ANIMATION"
    existing = bpy.data.actions.get(action_name)
    if existing:
        bpy.data.actions.remove(existing, do_unlink=True)

    combined = bpy.data.actions.new(action_name)

    # Blender 5.1 Action Slots require an OBJECT slot because animation_data on
    # an armature object belongs to the Object ID data-block.
    use_slotted_api = hasattr(combined, "slots") and hasattr(combined, "layers")

    target_slot = None
    target_layer = None
    target_strip = None
    target_channelbag = None

    if use_slotted_api:
        target_slot = combined.slots.new(id_type='OBJECT', name=target_armature.name)
        combined.slots.active = target_slot
        target_layer = combined.layers.new("Joined Animation")
        target_strip = target_layer.strips.new(type='KEYFRAME')
        target_channelbag = target_strip.channelbag(target_slot, ensure=True)

    copied_curves = 0
    source_count = 0

    for source_obj, source_action in source_actions:
        if source_action is None:
            continue
        source_count += 1
        bone_map = bone_maps.get(source_obj.name, {})

        for src_fcurve, src_channelbag in get_action_fcurves(source_action):
            data_path = rewrite_pose_bone_paths(src_fcurve.data_path, bone_map)

            # Bone F-Curves from every source are safe to merge because their
            # bone names are prefixed. Object-level transform curves, however,
            # all use paths such as "location" and "rotation_euler" and would
            # conflict in one final object. Keep those only from the master
            # armature.
            is_pose_curve = "pose.bones[" in src_fcurve.data_path
            is_object_transform = src_fcurve.data_path in {
                "location",
                "rotation_euler",
                "rotation_quaternion",
                "rotation_axis_angle",
                "scale",
            }
            if is_object_transform and source_obj is not target_armature:
                continue
            if not is_pose_curve and not is_object_transform:
                # Preserve custom-property / non-transform animation when it
                # does not create an obvious object-channel collision.
                pass

            group_name = ""
            try:
                if src_fcurve.group:
                    group_name = src_fcurve.group.name
            except Exception:
                group_name = ""

            try:
                if use_slotted_api:
                    # Copy the entire F-Curve through the Blender 5.x API.
                    dst = target_channelbag.fcurves.new_from_fcurve(
                        src_fcurve,
                        data_path=data_path,
                    )
                else:
                    dst = combined.fcurves.new(
                        data_path=data_path,
                        index=src_fcurve.array_index,
                        action_group=group_name,
                    )
                    copy_keyframes(src_fcurve, dst)

                # new_from_fcurve carries keyframes; still restore settings.
                copy_fcurve_settings(src_fcurve, dst)
                copied_curves += 1
            except Exception as exc:
                print(
                    f"[Join Armatures] Warning: could not copy F-Curve "
                    f"'{src_fcurve.data_path}[{src_fcurve.array_index}]': {exc}"
                )

    # If no animations were present, keep a valid empty action rather than
    # assigning None.
    if target_armature.animation_data is None:
        target_armature.animation_data_create()

    target_armature.animation_data.action = combined
    try:
        if target_slot is not None:
            target_armature.animation_data.action_slot = target_slot
    except Exception:
        pass

    # Preserve active source object's action range when possible.
    frame_ranges = []
    for _, act in source_actions:
        if act is not None:
            try:
                frame_ranges.append(tuple(act.frame_range))
            except Exception:
                pass
    if frame_ranges:
        start = min(r[0] for r in frame_ranges)
        end = max(r[1] for r in frame_ranges)
        try:
            combined.use_frame_range = True
            combined.frame_start = start
            combined.frame_end = end
        except Exception:
            pass

    print(f"[Join Armatures] Created '{combined.name}' with {copied_curves} F-Curves from {source_count} action(s).")
    return combined


def rename_bones_and_prepare_maps(armatures):
    bone_maps = {}
    for armature in armatures:
        bone_map = {}
        original_names = [bone.name for bone in armature.data.bones]

        for old_name in original_names:
            if PREFIX_BONES:
                new_name = f"{armature.name}_{old_name}"
            else:
                new_name = old_name
            bone_map[old_name] = new_name

        # Assigning names one at a time can collide with a currently existing
        # name. Use indexed temporary names first for reliable renaming.
        if PREFIX_BONES:
            temp_to_final = {}
            for index, bone in enumerate(armature.data.bones):
                old_name = original_names[index]
                temp_name = f"__JOIN_TMP__{index:06d}"
                temp_to_final[temp_name] = bone_map[old_name]
                bone.name = temp_name
            for bone in armature.data.bones:
                bone.name = temp_to_final[bone.name]

        bone_maps[armature.name] = bone_map
    return bone_maps


def make_mesh_modifiers_safe(meshes):
    """Record and remove old armature modifiers before mesh joining."""
    records = []
    for mesh in meshes:
        source_armature = get_source_mesh_armature(mesh)
        for mod in list(mesh.modifiers):
            if mod.type != 'ARMATURE':
                continue
            records.append((mesh, source_armature))
            mesh.modifiers.remove(mod)
    return records


def add_final_armature_modifier(mesh, armature):
    # Avoid duplicate armature modifiers targeting the same armature.
    for mod in mesh.modifiers:
        if mod.type == 'ARMATURE' and mod.object == armature:
            return mod
    mod = mesh.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = armature
    return mod


def preserve_parent_world_transform(obj, new_parent):
    world = obj.matrix_world.copy()
    obj.parent = new_parent
    obj.matrix_world = world


def select_only(objects, active=None):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        if obj and obj.name in bpy.context.view_layer.objects:
            obj.select_set(True)
    if active:
        bpy.context.view_layer.objects.active = active


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    if bpy.app.version < (5, 1, 0):
        raise RuntimeError("This script is intended for Blender 5.1 or newer.")

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    selected = get_selected_objects()
    armatures = [obj for obj in selected if obj.type == 'ARMATURE']
    meshes = [obj for obj in selected if obj.type == 'MESH']

    if len(armatures) < 2:
        raise RuntimeError("Select at least 2 armature objects.")

    # The first selected armature is the active/master armature.
    first_armature = armatures[0]
    original_first_name = first_armature.name

    print("=" * 72)
    print("JOIN ARMATURES - BLENDER 5.1")
    print(f"Master armature: {original_first_name}")
    print(f"Armatures: {len(armatures)} | Meshes: {len(meshes)}")
    print("=" * 72)

    # Snapshot animations BEFORE changing bone names.
    source_actions = []
    for armature in armatures:
        action = None
        if armature.animation_data:
            action = armature.animation_data.action
        source_actions.append((armature, action))
        if action:
            print(f"Animation found: {armature.name} -> {action.name}")
        else:
            print(f"No active action: {armature.name}")

    # Build bone maps before renaming so vertex groups and F-Curves can be
    # updated consistently.
    bone_maps = rename_bones_and_prepare_maps(armatures)

    # Update mesh vertex groups according to the armature that deforms them.
    for mesh in meshes:
        source_armature = get_source_mesh_armature(mesh)
        if source_armature and source_armature.name in bone_maps:
            count = rename_vertex_groups_for_armature(
                mesh,
                bone_maps[source_armature.name],
            )
            if count:
                print(f"Vertex groups renamed: {mesh.name} -> {count}")

    # Build the combined action BEFORE joining armatures. This avoids losing
    # animation when Blender removes the secondary armature objects.
    combined_action = create_combined_action(
        first_armature,
        source_actions,
        bone_maps,
    )

    # Preserve the master object's current transform if possible.
    master_world = first_armature.matrix_world.copy()

    # Remove old armature modifiers only; keep all other mesh modifiers.
    # This prevents a stale modifier from pointing to a deleted armature.
    make_mesh_modifiers_safe(meshes)

    # Detach meshes while keeping their current world transforms. This avoids
    # leaving the joined mesh parented to a secondary armature that will be
    # deleted by the armature join.
    for mesh in meshes:
        world = mesh.matrix_world.copy()
        mesh.parent = None
        mesh.matrix_parent_inverse.identity()
        mesh.matrix_world = world

    # Join armatures.
    select_only(armatures, active=first_armature)
    bpy.ops.object.join()
    joined_armature = bpy.context.object

    if RENAME_JOINED_OBJECT:
        desired_name = f"{original_first_name}_JOINED_ARMATURE"
        joined_armature.name = unique_object_name(desired_name)

    # Restore master object transform because joining can make object selection
    # and active-object context affect the final transform.
    try:
        joined_armature.matrix_world = master_world
    except Exception:
        pass

    # Re-assign the combined action after join; joining may have created or
    # changed animation data on the active object.
    if joined_armature.animation_data is None:
        joined_armature.animation_data_create()
    joined_armature.animation_data.action = combined_action

    # Blender 5.1 uses action slots; select the slot compatible with the joined
    # armature object when the assignment created multiple possible slots.
    try:
        suitable = joined_armature.animation_data.action_suitable_slots
        if suitable:
            joined_armature.animation_data.action_slot = suitable[0]
    except Exception:
        pass

    # Join meshes into a single mesh if requested.
    joined_mesh = None
    if JOIN_MESHES and meshes:
        select_only(meshes, active=meshes[0])
        bpy.ops.object.join()
        joined_mesh = bpy.context.object
        joined_mesh.name = unique_object_name(f"{original_first_name}_JOINED_MESH")

        # Parent while preserving the current world transform.
        preserve_parent_world_transform(joined_mesh, joined_armature)
        joined_mesh.parent_type = 'OBJECT'

        # New single armature modifier drives all prefixed vertex groups.
        add_final_armature_modifier(joined_mesh, joined_armature)

    # Save the original actions as named backups for recovery/debugging.
    if KEEP_SOURCE_ACTIONS:
        for source_obj, source_action in source_actions:
            if source_action is None:
                continue
            try:
                backup = source_action.copy()
                backup.name = f"KEEP_{source_obj.name}_{source_action.name}"
            except Exception as exc:
                print(f"Could not create action backup for {source_obj.name}: {exc}")

    # Make the final selection clean.
    final_selection = [joined_armature]
    if joined_mesh:
        final_selection.append(joined_mesh)
    select_only(final_selection, active=joined_armature)

    # Update the current frame so the viewport evaluates the new action.
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(current_frame)

    print("=" * 72)
    print("SUCCESS")
    print(f"Joined armature: {joined_armature.name}")
    if joined_mesh:
        print(f"Joined mesh:     {joined_mesh.name}")
    print(f"Action:          {combined_action.name}")
    print(f"Frame range:     {combined_action.frame_range[:]}" if hasattr(combined_action, 'frame_range') else "Frame range:     n/a")
    print("Bone animation paths were rewritten for renamed bones.")
    print("Vertex groups were rewritten to match the new bone names.")
    print("The final mesh has a new Armature modifier targeting the joined armature.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[Join Armatures] ERROR:", exc)
        raise
