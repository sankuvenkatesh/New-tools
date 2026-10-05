import bpy
from bpy_extras import anim_utils

# Updated for Blender 5.1 (works on 5.0 and newer).
# The legacy Action API (action.fcurves / action.groups) was removed in Blender 5.0.
# F-Curves now live in a "channelbag" inside the action, one channelbag per action slot.


def find_slot(action, identifier):
    """Return the slot of `action` with the given identifier, or None."""
    for slot in action.slots:
        if slot.identifier == identifier:
            return slot
    return None


def copy_fcurve(src_fcurve, dst_channelbag):
    """Copy an F-Curve (keys, interpolation, handles) into a channelbag."""
    data_path = src_fcurve.data_path
    index = src_fcurve.array_index

    if dst_channelbag.fcurves.find(data_path, index=index) is not None:
        print(f"Skipping duplicate F-Curve: {data_path}[{index}]")
        return

    group_name = src_fcurve.group.name if src_fcurve.group else ""
    new_fcurve = dst_channelbag.fcurves.new(data_path, index=index, group_name=group_name)
    new_fcurve.extrapolation = src_fcurve.extrapolation
    new_fcurve.auto_smoothing = src_fcurve.auto_smoothing

    for key in src_fcurve.keyframe_points:
        new_key = new_fcurve.keyframe_points.insert(frame=key.co.x, value=key.co.y, options={'FAST'})
        new_key.interpolation = key.interpolation
        new_key.easing = key.easing
        new_key.back = key.back
        new_key.amplitude = key.amplitude
        new_key.period = key.period
        new_key.type = key.type
        new_key.handle_left_type = key.handle_left_type
        new_key.handle_right_type = key.handle_right_type
        new_key.handle_left = key.handle_left
        new_key.handle_right = key.handle_right

    # 'FAST' inserts skip handle calculation, so recalculate once at the end
    new_fcurve.update()


def main():
    if bpy.app.version < (5, 0, 0):
        print("This script needs Blender 5.0 or newer (layered Action API).")
        return

    selected_objects = bpy.context.selected_objects

    # Separate armatures and mesh objects
    armatures = [obj for obj in selected_objects if obj.type == 'ARMATURE']
    meshes = [obj for obj in selected_objects if obj.type == 'MESH']

    if len(armatures) < 2:
        print("Please select at least 2 armatures.")
        return

    # Get the first armature
    first_armature = armatures[0]
    first_armature_name = first_armature.name
    action_sources = []  # (action name, slot identifier) of every duplicated action

    # Duplicate current actions and rename them for each armature
    for armature in armatures:
        anim_data = armature.animation_data
        if not (anim_data and anim_data.action):
            continue

        source_slot = anim_data.action_slot
        if source_slot is None:
            print(f"{armature.name}: its action has no slot assigned, skipping its animation.")
            continue
        slot_identifier = source_slot.identifier

        new_action = anim_data.action.copy()
        new_action.name = f"{armature.name}_combined_action"
        anim_data.action = new_action
        # Use the same slot in the copy. Renaming the bones below only updates
        # the F-Curve paths of the slot that is assigned to the object.
        anim_data.action_slot = find_slot(new_action, slot_identifier)
        action_sources.append((new_action.name, slot_identifier))

    # Rename bones to include armature prefix
    for armature in armatures:
        for bone in armature.data.bones:
            new_name = f"{armature.name}_{bone.name}"
            bone.name = new_name

    # Clear parent and delete armature modifier from meshes
    for mesh in meshes:
        mesh.parent = None
        # Remove armature modifiers
        modifiers_to_remove = [mod for mod in mesh.modifiers if mod.type == 'ARMATURE']
        for modifier in modifiers_to_remove:
            mesh.modifiers.remove(modifier)

    # Join all meshes into one
    if meshes:
        bpy.ops.object.select_all(action='DESELECT')
        for mesh in meshes:
            mesh.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        joined_mesh = bpy.context.object
        joined_mesh.name = f"{first_armature_name}_joined_objects"
    else:
        joined_mesh = None

    # Select all armatures and set the first armature as active
    if len(armatures) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for armature in armatures:
            armature.select_set(True)
        bpy.context.view_layer.objects.active = first_armature

        # Simulate pressing Ctrl+J to join armatures
        bpy.ops.object.join()
        first_armature.name = f"{first_armature_name}_joined_armatures"

    # Combine all actions into a new action (one slot for the joined armature)
    combined_action = bpy.data.actions.new(name="joined_combined_action")
    combined_slot = combined_action.slots.new(id_type='OBJECT', name=first_armature.name)
    combined_channelbag = anim_utils.action_ensure_channelbag_for_slot(combined_action, combined_slot)

    for action_name, slot_identifier in action_sources:
        action = bpy.data.actions.get(action_name)
        if action is None:
            continue
        slot = find_slot(action, slot_identifier)
        channelbag = anim_utils.action_get_channelbag_for_slot(action, slot) if slot is not None else None
        if channelbag is None:
            continue
        for fcurve in channelbag.fcurves:
            copy_fcurve(fcurve, combined_channelbag)

    # Assign the new combined action (and its slot) to the joined armature
    anim_data = first_armature.animation_data or first_armature.animation_data_create()
    anim_data.action = combined_action
    anim_data.action_slot = combined_slot

    # Parent the joined object to the new armature
    if joined_mesh:
        joined_mesh.parent = first_armature
        joined_mesh.parent_type = 'ARMATURE'

    # Ensure the new action "joined_combined_action" is displayed
    for area in bpy.context.screen.areas:
        if area.type == 'DOPESHEET_EDITOR':
            area.spaces.active.action = combined_action

    print("Operation completed successfully.")


if __name__ == "__main__":
    main()
