import bpy
import re
from bpy_extras import anim_utils

# Blender 5.1+ conservative armature/mesh merger.
# Run in Object Mode on a saved duplicate. Select >=2 armatures and only
# their meshes. Make the intended surviving armature active.

PREFIX = "_"
ACTION_NAME = "MERGED_Combined_Action"


def log(msg):
    print("[Safe Rig Merger] " + str(msg))


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


def remap_bone_path(path, mapping):
    """Single-pass replacement prevents cascading bone-name substitutions."""
    escape = getattr(bpy.utils, "escape_identifier", None)
    if escape is None:
        raise RuntimeError(
            "bpy.utils.escape_identifier is unavailable in this Blender build."
        )
    replacements = {}
    for old, new in mapping.items():
        source = f'pose.bones["{escape(old)}"]'
        target = f'pose.bones["{escape(new)}"]'
        replacements[source] = target
    if not replacements:
        return path
    pattern = re.compile("|".join(
        re.escape(token) for token in sorted(replacements, key=len, reverse=True)
    ))
    return pattern.sub(lambda m: replacements[m.group(0)], path)


def rig_for_mesh(mesh, armatures):
    found = []
    for mod in mesh.modifiers:
        if mod.type == 'ARMATURE' and mod.object in armatures:
            if mod.object not in found:
                found.append(mod.object)
    if mesh.parent in armatures and mesh.parent not in found:
        found.append(mesh.parent)
    if len(found) > 1:
        raise RuntimeError(
            f'{mesh.name}: linked to multiple selected armatures; resolve manually.'
        )
    return found[0] if found else None


def bone_mapping(arm):
    return {b.name: arm.name + PREFIX + b.name for b in arm.data.bones}


def validate_groups(mesh, mapping):
    names = {g.name for g in mesh.vertex_groups}
    for old, new in mapping.items():
        if new in names and old not in names and old != new:
            raise RuntimeError(
                f'{mesh.name}: vertex group "{new}" already exists, but the '
                f'corresponding source group "{old}" does not. It may be an '
                "unrelated group that would accidentally bind to a renamed bone."
            )
        if old in names and new in names and old != new:
            raise RuntimeError(
                f'{mesh.name}: vertex-group collision "{new}". Resolve first.'
            )


def rename_groups(mesh, mapping):
    jobs = [(g, mapping[g.name]) for g in mesh.vertex_groups
            if g.name in mapping and mapping[g.name] != g.name]
    for i, (group, _) in enumerate(jobs):
        group.name = f"__MERGE_TEMP_{i:05d}__"
    for group, new_name in jobs:
        group.name = new_name


def copy_fcurve(src, bag, mapping):
    path = remap_bone_path(src.data_path, mapping)
    idx = src.array_index
    if bag.fcurves.find(path, index=idx) is not None:
        raise RuntimeError(f"Duplicate animation channel: {path}[{idx}]")
    group = src.group.name if src.group else None
    try:
        dst = bag.fcurves.new(path, index=idx, action_group=group)
    except TypeError as exc:
        raise RuntimeError(
            "Blender 5.1 channelbag FCurve.new(action_group=...) failed "
            "before scene joins: " + str(exc)
        )
    dst.extrapolation = src.extrapolation
    dst.auto_smoothing = src.auto_smoothing
    dst.mute = src.mute
    dst.lock = src.lock

    n = len(src.keyframe_points)
    if n:
        dst.keyframe_points.add(n)
        for a, b in zip(src.keyframe_points, dst.keyframe_points):
            b.co = a.co
            b.handle_left = a.handle_left
            b.handle_right = a.handle_right
            b.interpolation = a.interpolation
            b.easing = a.easing
            b.type = a.type
            b.handle_left_type = a.handle_left_type
            b.handle_right_type = a.handle_right_type
            for attr in ("back", "amplitude", "period"):
                if hasattr(a, attr) and hasattr(b, attr):
                    try:
                        setattr(b, attr, getattr(a, attr))
                    except Exception:
                        pass

    for src_mod in src.modifiers:
        try:
            dst_mod = dst.modifiers.new(src_mod.type)
        except Exception as exc:
            log(f"Could not copy FCurve modifier {src_mod.type}: {exc}")
            continue
        for prop in src_mod.bl_rna.properties:
            key = prop.identifier
            if key in {"rna_type", "type"} or prop.is_readonly:
                continue
            try:
                setattr(dst_mod, key, getattr(src_mod, key))
            except Exception:
                pass
    dst.update()


def collect_actions(armatures, maps):
    sources = []
    for arm in armatures:
        ad = arm.animation_data
        if not ad or not ad.action:
            log(f"{arm.name}: no active action; skipped.")
            continue
        action, slot = ad.action, ad.action_slot
        if slot is None:
            log(f"{arm.name}: active action has no slot; skipped.")
            continue
        bag = anim_utils.action_get_channelbag_for_slot(action, slot)
        if bag is None:
            log(f"{arm.name}: no channelbag; skipped.")
            continue
        # Validate every path before any destructive changes.
        for fc in bag.fcurves:
            remap_bone_path(fc.data_path, maps[arm])
        sources.append({
            "mapping": maps[arm],
            "curves": list(bag.fcurves),
            "start": action.frame_start,
            "end": action.frame_end,
            "use_range": action.use_frame_range,
        })
    return sources


def build_action(sources, survivor):
    """Create/populate the action before joins so API errors are non-destructive."""
    if not sources:
        return None, None
    action = bpy.data.actions.new(unique_name(bpy.data.actions, ACTION_NAME))
    action.use_fake_user = True
    try:
        slot = action.slots.new(id_type='OBJECT', name=survivor.name)
        bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
        starts, ends = [], []
        for source in sources:
            if source["use_range"]:
                starts.append(source["start"])
                ends.append(source["end"])
            for fc in source["curves"]:
                copy_fcurve(fc, bag, source["mapping"])
        if starts:
            action.use_frame_range = True
            action.frame_start = min(starts)
            action.frame_end = max(ends)
        return action, slot
    except Exception:
        try:
            action.use_fake_user = False
            bpy.data.actions.remove(action)
        except Exception:
            pass
        raise


def warn_rig_state(armatures, survivor):
    selected_ids = set(armatures)

    for arm in armatures:
        ad = arm.animation_data
        if ad and len(ad.nla_tracks):
            log(f"WARNING: {arm.name} has {len(ad.nla_tracks)} NLA track(s); "
                "non-survivor NLA is not merged.")

        if ad:
            for driver in ad.drivers:
                log(f"REVIEW: {arm.name} driver data_path="
                    f'"{driver.data_path}" (driver paths are not auto-remapped)')
                for variable in driver.driver.variables:
                    for target in variable.targets:
                        target_id = target.id
                        if target_id in selected_ids:
                            log(
                                f"REVIEW: {arm.name} driver -> "
                                f"{target_id.name}, "
                                f'bone_target="{target.bone_target}", '
                                f'data_path="{driver.data_path}" '
                                "(cross-rig driver references are not auto-remapped)"
                            )

        for pose_bone in arm.pose.bones:
            if arm != survivor and not pose_bone.matrix_basis.is_identity:
                log(f"WARNING: {arm.name}/{pose_bone.name} has non-identity "
                    "pose transforms; verify after joining.")

            for constraint in pose_bone.constraints:
                if constraint.subtarget:
                    target = getattr(constraint, "target", None)
                    target_name = target.name if target else "<no target>"
                    log(
                        f"REVIEW: {arm.name}/{pose_bone.name} constraint "
                        f'"{constraint.name}" target="{target_name}" '
                        f'subtarget="{constraint.subtarget}" '
                        "(bone subtargets are not auto-remapped)"
                    )


def validate():
    if bpy.app.version < (5, 1, 0):
        raise RuntimeError("Requires Blender 5.1 or newer.")
    if bpy.context.mode != 'OBJECT':
        raise RuntimeError("Switch to Object Mode.")
    selected = list(bpy.context.selected_objects)
    arms = [o for o in selected if o.type == 'ARMATURE']
    meshes = [o for o in selected if o.type == 'MESH']
    if len(arms) < 2:
        raise RuntimeError("Select at least two armature objects.")
    ids = [a.data.as_pointer() for a in arms]
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Selected armature objects share armature data. Make data single-user "
            "or deselect one of the offending objects."
        )
    active = bpy.context.view_layer.objects.active
    survivor = active if active in arms else arms[0]
    return arms, meshes, survivor


def main():
    created_action = None
    try:
        arms, meshes, survivor = validate()
        log(f"Survivor: {survivor.name}")

        mesh_rigs = {m: rig_for_mesh(m, arms) for m in meshes}
        maps = {a: bone_mapping(a) for a in arms}
        mesh_world = {m: m.matrix_world.copy() for m in meshes}

        active_mesh = (bpy.context.view_layer.objects.active
                       if bpy.context.view_layer.objects.active in meshes
                       else (meshes[0] if meshes else None))

        # Preflight all meshes before modifying the scene.
        for mesh in meshes:
            rig = mesh_rigs[mesh]
            if rig is None:
                raise RuntimeError(
                    f'{mesh.name}: no selected rig detected. Deselect it or '
                    "assign it to the correct selected armature."
                )
            validate_groups(mesh, maps[rig])
            if mesh != active_mesh:
                unsupported = [mod.name for mod in mesh.modifiers
                               if mod.type != 'ARMATURE']
                if unsupported:
                    raise RuntimeError(
                        f'{mesh.name}: non-active mesh has modifiers that Join '
                        f"would discard: {', '.join(unsupported)}. Apply or "
                        "resolve them before merging."
                    )
            for mod in mesh.modifiers:
                if (mod.type == 'ARMATURE' and mod.object == rig
                        and mod.use_bone_envelopes):
                    log(f"WARNING: {mesh.name}/{mod.name} uses bone envelopes; "
                        "the new modifier will use vertex groups only.")

        action_sources = collect_actions(arms, maps)

        # Critical fix: create/populate action before destructive joins.
        created_action, combined_slot = build_action(action_sources, survivor)

        # Warnings are captured while original objects and references still exist.
        warn_rig_state(arms, survivor)

        # Create an undo boundary immediately before the first scene mutation.
        try:
            bpy.ops.ed.undo_push(message="Safe Armature and Mesh Merge")
        except Exception:
            pass

        # Rename bones.
        for arm in arms:
            mapping = maps[arm]
            for bone in arm.data.bones:
                bone.name = mapping[bone.name]

        # Rename matching vertex groups for each mesh's original rig.
        for mesh, rig in mesh_rigs.items():
            rename_groups(mesh, maps[rig])

        # Detach meshes preserving world transforms; remove only selected-rig mods.
        for mesh in meshes:
            mesh.parent = None
            mesh.matrix_world = mesh_world[mesh]
            for mod in list(mesh.modifiers):
                if mod.type == 'ARMATURE' and mod.object in arms:
                    mesh.modifiers.remove(mod)

        # Join meshes. Only the active mesh's modifiers survive, hence preflight.
        joined_mesh = None
        if meshes:
            deselect_all()
            for mesh in meshes:
                mesh.select_set(True)
            bpy.context.view_layer.objects.active = active_mesh
            bpy.ops.object.join()
            joined_mesh = active_mesh
            joined_mesh.name = unique_name(
                bpy.data.objects, survivor.name + "_JoinedMesh"
            )

        # Join armatures into the chosen survivor.
        deselect_all()
        for arm in arms:
            arm.select_set(True)
        bpy.context.view_layer.objects.active = survivor
        bpy.ops.object.join()

        survivor.name = unique_name(
            bpy.data.objects, survivor.name + "_CombinedRig"
        )
        survivor.data.name = unique_name(
            bpy.data.armatures, survivor.name + "_ArmatureData"
        )

        if created_action:
            ad = survivor.animation_data or survivor.animation_data_create()
            ad.action = created_action
            ad.action_slot = combined_slot

        if joined_mesh:
            world = joined_mesh.matrix_world.copy()
            joined_mesh.parent = survivor
            joined_mesh.parent_type = 'OBJECT'
            joined_mesh.parent_bone = ""
            joined_mesh.matrix_world = world
            mod = joined_mesh.modifiers.new("Combined Armature", 'ARMATURE')
            mod.object = survivor
            mod.use_vertex_groups = True
            mod.use_bone_envelopes = False

        deselect_all()
        survivor.select_set(True)
        if joined_mesh:
            joined_mesh.select_set(True)
        bpy.context.view_layer.objects.active = survivor
        log("Merge completed. Thoroughly test deformation, animation, constraints, "
            "drivers, NLA and custom rig controls before production use.")

    except Exception as exc:
        log("STOPPED: " + str(exc))
        if created_action and created_action.name in bpy.data.actions:
            try:
                created_action.use_fake_user = False
                if created_action.users == 0:
                    bpy.data.actions.remove(created_action)
            except Exception:
                pass
        log("If any scene changes occurred, use Ctrl+Z or reopen your backup.")
        raise


if __name__ == "__main__":
    main()
