"""Retarget the solved 3D joints onto the pose bones of a Rigify armature.

By default only the "rotation" channels are written (keeping the armature proportions unchanged); root translation + keyframes are optional.
The pose matrix of every bone is decided by two parts:
  * swing (aim): the minimal rotation from the rest direction -> the target direction;
  * twist: align the roll with a reference axis (shoulder line / hip line / arm bend axis / eye line ...) so arms / head do not get wrung like a rope.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

from . import reconstruction as recon_mod
from . import rigify_limbs, utils

FINGER_CHAIN = {
    "thumb": ("thumb.01", "thumb.02", "thumb.03"),
    "index": ("f_index.01", "f_index.02", "f_index.03"),
    "middle": ("f_middle.01", "f_middle.02", "f_middle.03"),
    "ring": ("f_ring.01", "f_ring.02", "f_ring.03"),
    "pinky": ("f_pinky.01", "f_pinky.02", "f_pinky.03"),
}


def build_targets(recon: dict, bmap, metrics: dict) -> dict:
    """role -> (start 3D, end 3D, reference axis name). Coordinates are in the "camera / character" world frame."""
    joints = recon["joints"]
    spine_points = recon["spine_points"]
    count = max(len(bmap.spine_bones()), 1)
    pts = recon_mod._resample_polyline(spine_points, count)

    targets = {"hips": (spine_points[0], pts[1], "hip_line")}
    for index in range(count):
        targets["spine_%02d" % index] = (pts[index], pts[index + 1], "shoulder_line")

    targets["neck"] = (joints.get("neck_start", joints["chest"]), joints["head_base"],
                       "shoulder_line")
    targets["head"] = (joints["head_base"], joints["head_tip"], "eye_line")

    for side in ("L", "R"):
        origin = joints.get("shoulder_origin.%s" % side, joints["chest"])
        targets["shoulder.%s" % side] = (origin, joints["shoulder.%s" % side],
                                         "shoulder_line")
        targets["upper_arm.%s" % side] = (joints["shoulder.%s" % side],
                                          joints["elbow.%s" % side], "armbend.%s" % side)
        targets["forearm.%s" % side] = (joints["elbow.%s" % side], joints["wrist.%s" % side],
                                        "armbend.%s" % side)
        # The roll of palm / fingers also follows the "arm bend plane":
        # the bend axis of wrist flexion degenerates when the wrist is straight (the cross product tends to 0), and aligning the roll with it jitters by tens of degrees;
        # using the elbow bend axis is equivalent to "no extra twist at the wrist", consistent with the relation between palm and arm plane in the rest pose
        targets["hand.%s" % side] = (joints["wrist.%s" % side], joints["hand_tip.%s" % side],
                                     "armbend.%s" % side)
        targets["palm.%s" % side] = targets["hand.%s" % side]
        targets["thigh.%s" % side] = (joints["hip.%s" % side], joints["knee.%s" % side],
                                      "hip_line")
        targets["shin.%s" % side] = (joints["knee.%s" % side], joints["ankle.%s" % side],
                                     "kneebend.%s" % side)
        targets["foot.%s" % side] = (joints["ankle.%s" % side], joints["ball.%s" % side],
                                     "hip_line")
        # Toes are not written separately: a single image cannot tell how the toes bend, so keeping the rest relation to the foot is safer

    for side, per_hand in recon.get("fingers", {}).items():
        for finger, chain in per_hand.items():
            names = FINGER_CHAIN.get(finger)
            if not names or len(chain) < 4:
                continue
            for index, base in enumerate(names):
                targets["%s.%s" % (base, side)] = (chain[index], chain[index + 1],
                                                   "armbend.%s" % side)
    return targets


def solve_orientation(rest_mat3: Matrix, target_dir: Vector, ref: Vector = None,
                      rest_ref: Vector = None, use_twist: bool = True,
                      twist_blend: float = 1.0) -> Matrix:
    """Return the target pose rotation matrix (3x3) of a bone in armature space.

    Swing (aim): the minimal rotation from the rest direction -> the target direction.
    Twist: carry the rest reference axis along with the swing, then rotate around the bone axis to the target reference axis.
      * The "reference axis itself" must be carried; the matrix column closest to it must not be used instead: Rigify's elbow bend axis
        is tens of degrees away from every column, and approximating it with a column shifts the roll as a whole.
      * The alignment must be "signed": when a limb folds the opposite way (e.g. the elbow points forward instead of backward),
        the roll has to turn by nearly 180 degrees, and taking the nearest side would mirror the bend plane.
    """
    rest_dir = rest_mat3.col[1].normalized()
    aim = rest_dir.rotation_difference(target_dir.normalized())
    matrix = aim.to_matrix() @ rest_mat3

    if not use_twist or ref is None or rest_ref is None or twist_blend <= 1e-6:
        return matrix
    if ref.length < 1e-6 or rest_ref.length < 1e-6:
        return matrix

    axis = target_dir.normalized()
    local_ref = rest_mat3.inverted() @ rest_ref.normalized()
    carried = matrix @ local_ref
    carried = carried - axis * carried.dot(axis)
    if carried.length < 1e-6:
        return matrix
    carried.normalize()

    desired = ref - axis * ref.dot(axis)
    if desired.length < 1e-6:
        return matrix
    desired.normalize()

    angle = math.atan2(carried.cross(desired).dot(axis), carried.dot(desired))
    return Matrix.Rotation(angle * twist_blend, 3, axis) @ matrix


def solve_role_matrix(role: str, target, refs: dict, rest_refs: dict, rest_mat3: Matrix,
                      to_arm, use_twist: bool = True, twist_blend: float = 1.0):
    """Target pose matrix of one role (armature-space 3x3); returns None when the direction is degenerate.

    ``target`` is ``(start, end, reference axis name)``; ``refs`` / ``rest_refs`` are the reference axis dicts of the "target frame" and
    the "rest frame". The IK direct write and the FK write share the same maths, so that the results of both sides line up.
    """
    p_from, p_to, ref_name = target[0], target[1], target[2]
    direction = Vector(p_to) - Vector(p_from)
    if direction.length < 1e-6:
        return None
    ref_arm = rest_ref = None
    if ref_name:
        ref_vec = refs.get(ref_name)
        # Note: the reference axes of the solved result are numpy arrays, which have no .length attribute,
        # so they must be wrapped in Vector() first, otherwise they are judged "invalid reference axes" here and the twist is silently skipped.
        try:
            ref_vec = Vector(ref_vec) if ref_vec is not None else None
        except (TypeError, ValueError):
            ref_vec = None
        if ref_vec is not None and ref_vec.length > 1e-9:
            ref_arm = to_arm(ref_vec)
            rest_ref = rest_refs.get(ref_name)
    return solve_orientation(rest_mat3, to_arm(direction), ref_arm, rest_ref,
                             use_twist, twist_blend)


# Thresholds for checking an IK direct write: beyond them we fall back to the old "switch to FK + Rigify snap" path.
IK_DIRECT_POS_TOL = 0.02     # metres
IK_DIRECT_ANGLE_TOL = 10.0   # degrees


# --------------------------------------------------------------------------- apply pose
def limb_mode_option(options: dict) -> str:
    """Whether to stay on IK (Rigify's default controls) or FK after this write.

    Compatible with the old key ``fk_mode``: historically True meant "automatically switch to FK mode".
    """
    mode = str(options.get("limb_mode") or "").upper()
    if mode in rigify_limbs.MODES:
        return mode
    return rigify_limbs.MODE_FK if options.get("fk_mode") else rigify_limbs.MODE_IK


def _limb_bone_names(spec: dict) -> set:
    """All bone names that a limb may be written to by us (FK chain / IK chain / controls)."""
    names = set(spec.get("fk_bones") or [])
    for key in ("ik_bones", "ctrl_bones", "tail_bones", "extra_ctrls"):
        names |= set(spec.get(key) or [])
    return names


def _limbs_written(written_names, specs: dict) -> dict:
    """Pick the Rigify limbs that were really written this time (limbs with a bone name in their FK chain / IK controls that was written).

    Only limbs that were written get the IK snap / direct write: limbs that were not written must not have their pose changed by us,
    otherwise their current IK placement would be overwritten with the FK rest pose.
    """
    out = {}
    written = set(written_names)
    for key, spec in (specs or {}).items():
        if _limb_bone_names(spec) & written:
            out[key] = spec
    return out


def _limb_keyframe_bones(limbs: dict, mode: str) -> list[str]:
    """IK controls that must be keyframed together in IK mode (the visible ones, without MCH-)."""
    if mode != rigify_limbs.MODE_IK:
        return []
    out: list[str] = []
    for spec in (limbs or {}).values():
        for name in spec["ctrl_bones"] + spec["tail_bones"] + spec["extra_ctrls"]:
            if name not in out:
                out.append(name)
    return out


def close_limbs(arm_obj, limbs: dict, mode: str, report: dict, options: dict,
                specs: dict = None, direct: dict = None) -> None:
    """Wrap-up: in IK mode align the IK controls (of limbs that could not be written directly) to the FK pose, then sync the controller visibility according to the mode.

    * ``direct``: limbs whose pose was **directly** written into the IK controls - they do not switch to FK and take no snap;
    * the other limbs still take the old path in IK mode: switch to FK -> Rigify's "IK->FK" snap -> back to IK;
    * FK mode needs no snap - the pose already sits on the FK bones.
    """
    if not limbs:
        # Only "a Rigify-generated rig where not a single IK chain can be found" is worth a hint
        if mode == rigify_limbs.MODE_IK and not (specs or {}):
            if rigify_limbs.rig_id(arm_obj):
                report["warnings"].append("No Rigify IK control chain found in the armature, writing as plain FK")
        return

    report["limb_mode"] = mode
    if mode == rigify_limbs.MODE_IK:
        snap = {key: spec for key, spec in limbs.items() if key not in (direct or {})}
        if snap:
            # Rigify's snap requires "the pose on the FK chain is the current pose", so switch to FK first and switch back after the snap.
            rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_FK, report, snap, sync=False)
            view_layer = bpy.context.view_layer
            if view_layer is not None:
                view_layer.update()
            snapped = rigify_limbs.snap_limbs(arm_obj, snap, report)
            report["ik_snapped"] = snapped
            missing = [spec["fk_bones"][0] for key, spec in snap.items()
                       if key not in snapped]
            if missing:
                report["warnings"].append("IK snap was not done for these limbs: %s" % ", ".join(missing))
        rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_IK, report, limbs, sync=False)
    else:
        report["fk_switched"] = sorted(spec["prop_bone"] for spec in limbs.values())

    if bool(options.get("sync_visibility", True)):
        touched = rigify_limbs.sync_visibility(arm_obj)
        if touched:
            report["visibility"] = touched


# --------------------------------------------------------------------------- IK direct write
def direct_limb_plans(bmap, limbs: dict) -> dict:
    """Pick the limbs for which the IK controls can be written directly this time: ``key -> {"spec", "plan"}``.

    Two conditions:
      1. the user's mapping really points to the IK controls of this limb (this is what automatic matching does in IK mode;
         if the user mapped the limb to FK bones by hand we do not overstep and change other bones);
      2. Rigify's chain information recognises a writable control (chain root / end) and the IK target bone.
    """
    out: dict = {}
    for key, spec in (limbs or {}).items():
        plan = rigify_limbs.ik_write_plan(spec)
        if not plan:
            continue
        names = set(plan["controls"].values()) | {plan["target"]}
        if not any(bmap.get(role) in names for role in plan["roles"]):
            continue
        out[key] = {"spec": spec, "plan": plan}
    return out


def _update_view() -> None:
    view_layer = bpy.context.view_layer
    if view_layer is not None:
        view_layer.update()


def _write_limb_fk_chain(arm_obj, plan: dict, targets: dict, refs: dict, metrics: dict,
                         to_arm, use_twist: bool, twist_blend: float) -> None:
    """Write the FK chain of one limb the way FK mode does (the rest matrices are those of the FK bones themselves).

    In IK mode this chain is hidden, but (a) the pose is not lost when the user drags the IK_FK slider back to FK, (b) when falling back to Rigify's
    snap it is the "current pose" the snap reads, (c) when checking the direct-write result it is the expected value. It must be written in chain order.
    """
    rest_refs = metrics.get("ref") or {}
    for role in plan["roles"]:
        name = plan["fk_of_role"].get(role) or ""
        bone = arm_obj.data.bones.get(name)
        pbone = arm_obj.pose.bones.get(name)
        target = targets.get(role)
        if bone is None or pbone is None or target is None:
            continue
        mat3 = solve_role_matrix(role, target, refs, rest_refs,
                                 bone.matrix_local.to_3x3(), to_arm, use_twist, twist_blend)
        if mat3 is None:
            continue
        pbone.matrix = Matrix.Translation(pbone.head.copy()) @ mat3.to_4x4()
        _update_view()


def _fk_pose_delta(arm_obj, fk_name: str):
    """The "world delta" matrix of an FK bone in armature space (rest -> current); returns None for a missing bone.

    Rigify's own "IK->FK" snap carries exactly this delta of the FK chain over to the IK controls. Using deltas instead of absolute
    coordinates automatically carries over however Rigify builds things internally (MCH bones, COPY_LOCATION handle bones), and the root bone
    translation / parent pose are already included in the pose of the FK chain.
    """
    bone = arm_obj.data.bones.get(fk_name)
    pbone = arm_obj.pose.bones.get(fk_name)
    if bone is None or pbone is None:
        return None
    return pbone.matrix @ bone.matrix_local.inverted()


def _limb_check_score(arm_obj, spec: dict, roles) -> float:
    """Turn the check deviation into one comparable score (smaller is better); if it cannot be checked it counts as infinity.

    Joint positions get a higher weight: the visible error of a pose is in the joints, and the chain bones' own roll being off by tens of degrees is normal
    (Rigify's own snap has it too; in testing ``DEF-upper_arm`` was off by 64.7 degrees), so do not let roll lead the search astray.
    """
    check = _ik_direct_check(arm_obj, spec, roles)
    if check is None:
        return float("inf")
    return (4.0 * check[0] / IK_DIRECT_POS_TOL
            + math.radians(check[1]) / math.radians(IK_DIRECT_ANGLE_TOL))


def _search_root_roll(arm_obj, spec: dict, plan: dict, steps: int = 16, rounds: int = 5) -> bool:
    """Search the "twist" (rotation around its own bone axis) of the chain root control so the check deviation is minimal - the idea is copied from Rigify's ``correct_rotation``.

    The bend plane of the IK chain is decided by the twist of the chain root (which way the knee / elbow points), and Rigify's snap searches it the same way.
    It is only called when a direct write failed the check, so limbs that pass do not pay for it.
    Returns whether a smaller deviation was really found.
    """
    roles = plan.get("roles") or []
    root_name = plan["controls"].get(roles[0]) if roles else None
    pbone = arm_obj.pose.bones.get(root_name or "")
    if pbone is None:
        return False
    matrix = pbone.matrix.copy()
    base, translation = matrix.to_3x3(), matrix.translation.copy()

    def apply(angle: float) -> None:
        rolled = base @ Matrix.Rotation(angle, 3, "Y")
        pbone.matrix = Matrix.Translation(translation) @ rolled.to_4x4()
        _update_view()

    start = _limb_check_score(arm_obj, spec, roles)
    best_angle, best_score = 0.0, start
    step = 2.0 * math.pi / steps
    for index in range(1, steps + 1):
        angle = -math.pi + step * index
        apply(angle)
        score = _limb_check_score(arm_obj, spec, roles)
        if score < best_score:
            best_angle, best_score = angle, score
    for _ in range(rounds):                  # after the coarse search finds a direction, refine with a shrinking step
        step /= 2.0
        for delta in (-step, step):
            apply(best_angle + delta)
            score = _limb_check_score(arm_obj, spec, roles)
            if score < best_score:
                best_angle, best_score = best_angle + delta, score
    apply(best_angle)
    return best_score < start


def _write_limb_ik_controls(arm_obj, spec: dict, plan: dict) -> int:
    """IK controls: write directly using the algorithm of Rigify's "IK->FK" snap (``RigifyLimbIk2FkBase.apply_frame_state``).

    Three things done the same way as Rigify:

      * the end controls ``hand_ik.L`` / ``foot_ik.L`` **are the end of this IK chain** (Rigify attaches them to the IK chain with a MCH bone
        carrying ``COPY_LOCATION``), so they need both the position **and** the orientation of the FK hand / foot -
        if only rotation were written, the wrist / ankle would not move at all and the joints on the chain would all be wrong;
      * the chain root ``upper_arm_ik.L`` / ``thigh_ik.L`` and the tail control ``toe_ik.L`` likewise get the delta of the matching FK bone;
      * ``*_ik_target`` is the **pole target** (it decides the IK bend plane, it is not a handle at the chain end): when
        ``pole_vector`` is enabled it is placed into the plane of the FK chain following Rigify's ``match_pole_target``
        (see :func:`rigify_limbs.place_pole`); when it is not enabled it is left alone - Rigify only places it when needed, too.

    The middle IK solve bones (``MCH-*``) are decided by the solver and are not written. Returns the number of controls written.
    """
    written = 0
    for role, control in plan["controls"].items():
        delta = _fk_pose_delta(arm_obj, plan["fk_of_role"].get(role) or "")
        bone = arm_obj.data.bones.get(control)
        pbone = arm_obj.pose.bones.get(control)
        if delta is None or bone is None or pbone is None:
            continue
        # Write position + rotation together: the rest matrix of the control follows the world delta of the FK bone as a whole
        pbone.matrix = delta @ bone.matrix_local
        _update_view()
        written += 1

    # Pole target (``*_ik_target``): the IK bend plane depends on it, and Rigify's snap places it too (see place_pole).
    if written and rigify_limbs.pole_enabled(arm_obj, spec):
        roles = plan.get("roles") or []
        root = arm_obj.pose.bones.get(plan["fk_of_role"].get(roles[0]) or "") if roles else None
        if root is not None:
            rigify_limbs.place_pole(arm_obj, spec, root.matrix)
            _update_view()
    return written


def _ik_direct_check(arm_obj, spec: dict, roles) -> tuple | None:
    """Check the direct-write result: the ``DEF-*`` deform bones driven by the IK chain vs the expected pose just written into the FK chain.

    It compares **joint positions** (the heads) and **bone directions**, not the tails: DEF bones driven by the IK chain get lengthened by their own
    stretch constraints, and Rigify's own snap path does the same (in testing joint error 0.000 m / tail error 0.27 m),
    so using the tail as the criterion would wrongly classify a result that is "as correct as the snap" as a failure.

    Returns ``(max position error in m, max direction error in degrees)``; returns ``None`` when not a single bone can be checked
    (the caller treats that as "failed" and falls back to the old path, which measured 0.000000 m).
    """
    side = str(spec.get("side") or "")
    if not side:
        return None
    suffix = ".%s" % side
    matrix = arm_obj.matrix_world
    wanted = set(roles or ())
    worst_pos = 0.0
    worst_deg = 0.0
    checked = 0
    for fk_name in spec.get("fk_bones") or []:
        stem = fk_name[:-len(suffix)] if fk_name.endswith(suffix) else ""
        if stem.endswith("_fk"):
            stem = stem[:-len("_fk")]
        if not stem or ("%s.%s" % (stem, side)) not in wanted:
            continue
        fk_pb = arm_obj.pose.bones.get(fk_name)
        def_pb = arm_obj.pose.bones.get("DEF-%s.%s" % (stem, side))
        if fk_pb is None or def_pb is None:
            continue
        worst_pos = max(worst_pos, (matrix @ fk_pb.head - matrix @ def_pb.head).length)
        fk_dir, df_dir = fk_pb.tail - fk_pb.head, def_pb.tail - def_pb.head
        if fk_dir.length > 1e-6 and df_dir.length > 1e-6:
            worst_deg = max(worst_deg, math.degrees(fk_dir.angle(df_dir)))
        checked += 1
    if not checked:
        return None
    return worst_pos, worst_deg


def apply_ik_direct(arm_obj, plans: dict, targets: dict, refs: dict, metrics: dict,
                    options: dict, report: dict) -> dict:
    """IK mode: write the pose directly into the IK controls (no FK switch, no call to Rigify's IK->FK snap).

    Right after writing, check against the ``DEF-*`` deform bones; limbs that do not match are left out of the return value and the caller sends them down the old path
    (switch to FK + Rigify snap), so the worst case is only "the direct write did not take effect", never a wrong pose.
    Returns ``{limb key: spec}`` (those really written directly) and writes the results into ``report``.
    """
    if not plans:
        return {}
    use_twist = bool(options.get("twist", True))
    twist_blend = float(options.get("twist_blend", 1.0))
    obj_mat3_inv = arm_obj.matrix_world.to_3x3().inverted()

    def to_arm(vec) -> Vector:
        return (obj_mat3_inv @ Vector(vec)).normalized()

    # A direct write only makes sense if these limbs are on IK (otherwise the result would be invisible)
    rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_IK, report,
                          {key: item["spec"] for key, item in plans.items()}, sync=False)
    _update_view()

    failed: list = []
    ready: list = []
    # 1) First write the FK chains of these limbs completely: if the direct write fails the check later and we fall back to Rigify's snap,
    #    the snap reads the pose on this chain (otherwise the IK controls would be aligned to the rest pose).
    for key, item in plans.items():
        try:
            _write_limb_fk_chain(arm_obj, item["plan"], targets, refs, metrics, to_arm,
                                 use_twist, twist_blend)
            ready.append(key)
        except Exception as exc:  # noqa: BLE001
            failed.append((item["spec"]["fk_bones"][0], "writing the FK chain failed: %s" % exc))
    _update_view()

    # 2) Then write the IK controls and check them against the DEF-* deform bones
    done: dict = {}
    measured: dict = {}
    for key in ready:
        item = plans[key]
        spec, plan = item["spec"], item["plan"]
        name = spec["fk_bones"][0]
        try:
            written = _write_limb_ik_controls(arm_obj, spec, plan)
            _update_view()
            check = _ik_direct_check(arm_obj, spec, plan["roles"])
        except Exception as exc:  # noqa: BLE001 a problem in the direct write must not take down the whole write
            failed.append((name, "direct write error: %s" % exc))
            continue
        if written <= 0:
            failed.append((name, "no control suitable for a direct write was found"))
            continue
        if check is not None:
            measured[name] = [round(float(check[0]), 4), round(float(check[1]), 2)]
        if check is None or check[0] > IK_DIRECT_POS_TOL or check[1] > IK_DIRECT_ANGLE_TOL:
            # Failed: search the twist of the chain root (the bend plane) the way Rigify does, and save it if that works
            if check is not None and _search_root_roll(arm_obj, spec, plan):
                check = _ik_direct_check(arm_obj, spec, plan["roles"])
                measured[name] = ([round(float(check[0]), 4), round(float(check[1]), 2)]
                                  if check is not None else None)
        if check is None or check[0] > IK_DIRECT_POS_TOL or check[1] > IK_DIRECT_ANGLE_TOL:
            failed.append((name, "cannot be verified (no DEF-* deform bones)" if check is None
                           else "deviation %.3f m / %.1f deg" % (check[0], check[1])))
            continue
        done[key] = spec

    report["ik_direct"] = [spec["fk_bones"][0] for spec in done.values()]
    if measured:
        report["ik_direct_error"] = measured
    if failed:
        report["warnings"].append(
            "These limbs use Rigify's IK snap instead (no direct write): %s"
            % "; ".join("%s %s" % (name, text) for name, text in failed))
    return done


def _apply_root_translation(arm_obj, bmap, metrics, recon, options, report) -> None:
    """Write the person's horizontal position / standing height into the translation channels of the root bone (optional)."""
    root_name = bmap.get("root")
    if not root_name:
        report["warnings"].append("The armature has no root/torso bone, root translation skipped")
        return
    pbone = arm_obj.pose.bones.get(root_name)
    if pbone is None:
        report["warnings"].append("Root bone %s not found" % root_name)
        return

    joints = recon["joints"]
    rest_pelvis_world = arm_obj.matrix_world @ Vector(metrics["pelvis"])
    pelvis = Vector(joints["pelvis"])
    target = Vector((pelvis.x, rest_pelvis_world.y, rest_pelvis_world.z))

    if bool(options.get("ground_align", True)):
        lows = [Vector(joints["ankle.%s" % side]) for side in ("L", "R")]
        lows += [Vector(joints["ball.%s" % side]) for side in ("L", "R")]
        lowest = min(vec.z for vec in lows)
        # Keep the "pelvis above the lowest foot point" height, putting the feet back on the ground level of the rest armature
        target.z = metrics["lowest_foot_z"] + max(pelvis.z - lowest, 0.0)

    delta_world = target - rest_pelvis_world
    if bool(options.get("keep_depth", True)):
        delta_world.y = 0.0

    delta_arm = arm_obj.matrix_world.to_3x3().inverted() @ delta_world
    matrix = pbone.matrix.copy()
    matrix.translation = Vector(arm_obj.data.bones[root_name].head_local) + delta_arm
    pbone.matrix = matrix
    if bpy.context.view_layer is not None:
        bpy.context.view_layer.update()
    report["root_offset"] = tuple(round(float(v), 4) for v in delta_arm)


def insert_keyframes(arm_obj, bmap, roles, frame: int, report: dict, extra_bones=None,
                     prop_bones=None) -> None:
    """Insert keyframes for the bones that were written.

    * ``extra_bones``: IK controls to be keyed together in IK mode (otherwise only the FK bones would be recorded and
      the animation would show no change);
    * ``prop_bones``: Rigify's ``IK_FK`` slider property (switching IK/FK should be recorded into the animation too).
    """
    extra = list(extra_bones or [])
    names: list[str] = []
    for role in set(list(roles) + ["root"]):
        name = bmap.get(role)
        if name and name not in names:
            names.append(name)
    for name in extra:
        if name not in names:
            names.append(name)

    count = 0
    for name in names:
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            continue
        if name == bmap.get("root") or name in extra:
            pbone.keyframe_insert(data_path="location", frame=frame)
        path = "rotation_quaternion" if pbone.rotation_mode == "QUATERNION" \
            else "rotation_euler"
        pbone.keyframe_insert(data_path=path, frame=frame)
        if name in extra:
            pbone.keyframe_insert(data_path="scale", frame=frame)
        count += 1

    for name in (prop_bones or []):
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None or "IK_FK" not in pbone.keys():
            continue
        pbone.keyframe_insert(data_path='["IK_FK"]', frame=frame)
        count += 1
    report["keyframes"] = count


def apply_pose(arm_obj, bmap, metrics, recon, options=None) -> dict:
    """Main entry: write the solved result to the pose bones and return an execution report."""
    options = options or {}
    report = {"ok": False, "applied": [], "skipped": [],
              "warnings": list(metrics.get("warnings", [])), "message": ""}

    if metrics.get("errors"):
        report["message"] = "; ".join(metrics["errors"])
        return report
    if not recon or not recon.get("ok"):
        report["message"] = (recon or {}).get("message", "Pose reconstruction failed")
        return report

    use_twist = bool(options.get("twist", True))
    twist_blend = float(options.get("twist_blend", 1.0))
    obj_mat3_inv = arm_obj.matrix_world.to_3x3().inverted()

    def to_arm(vec) -> Vector:
        return (obj_mat3_inv @ Vector(vec)).normalized()

    targets = build_targets(recon, bmap, metrics)
    refs = recon["refs"]
    plan = []

    for role, (p_from, p_to, ref_name) in targets.items():
        name = bmap.get(role)
        if not name:
            report["skipped"].append((role, "not mapped to a bone"))
            continue
        pbone = arm_obj.pose.bones.get(name)
        bone = arm_obj.data.bones.get(name)
        if pbone is None or bone is None:
            report["skipped"].append((role, "the armature has no %s" % name))
            continue
        direction = Vector(p_to) - Vector(p_from)
        if direction.length < 1e-6:
            report["skipped"].append((role, "degenerate direction"))
            continue

        rest_mat = metrics["rest_mat"].get(role)
        if rest_mat is None:
            rest_mat = bone.matrix_local.to_3x3()
        mat3 = solve_role_matrix(role, (p_from, p_to, ref_name), refs, metrics["ref"],
                                 rest_mat, to_arm, use_twist, twist_blend)
        if mat3 is None:
            report["skipped"].append((role, "degenerate direction"))
            continue
        plan.append((len(bone.parent_recursive), role, name, mat3))

    plan.sort(key=lambda item: item[0])

    view_layer = bpy.context.view_layer

    # The arms / thighs of a Rigify-generated rig hang on the IK controls by default. In IK mode, limbs whose IK controls can be written directly are
    # written directly (no FK switch, no snap); the other limbs (written as FK / falling back to the snap) still have to be switched to FK first,
    # otherwise the IK chain would cover the FK result, and the later IK snap would also read a wrong pose.
    limb_mode = limb_mode_option(options)
    limb_specs = rigify_limbs.limb_specs(arm_obj)
    written_limbs = _limbs_written({item[2] for item in plan}, limb_specs)
    direct_plans = (direct_limb_plans(bmap, written_limbs)
                    if limb_mode == rigify_limbs.MODE_IK else {})
    if written_limbs and not direct_plans:
        rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_FK, report, written_limbs,
                              sync=False)
        if view_layer is not None:
            view_layer.update()

    pivot = Vector(metrics.get("pelvis") or Vector((0.0, 0.0, 0.0)))
    hip_line_rest = metrics["ref"].get("hip_line")
    hip_line_target = recon["refs"].get("hip_line")
    for _depth, role, name, mat3 in plan:
        pbone = arm_obj.pose.bones[name]
        head = pbone.head.copy()
        bone = arm_obj.data.bones[name]
        rest_mat = metrics["rest_mat"].get(role) or bone.matrix_local.to_3x3()

        if role == "hips" and hip_line_rest is not None and hip_line_target is not None:
            # Rigify's torso is a "spine IK control" whose rest direction is not the body axis (measured: it points along +Y),
            # so aligning by direction would turn it by 90 degrees and fling the whole person away. Instead:
            # derive the pelvis rotation from the hip line and stack it onto the bone's rest pose.
            rotation = hip_line_rest.normalized().rotation_difference(
                to_arm(hip_line_target))
            mat3 = rotation.to_matrix() @ rest_mat
        else:
            rotation = mat3 @ rest_mat.inverted()

        matrix = Matrix.Translation(head) @ mat3.to_4x4()
        if role == "hips":
            # Keep the pelvis in place (equivalent to rotating around the pelvis rather than around the bone head)
            offset = pivot - Vector(bone.head_local)
            matrix.translation = head + (offset - rotation @ offset)
        try:
            pbone.matrix = matrix
        except Exception as exc:  # noqa: BLE001
            report["skipped"].append((role, "write failed: %s" % exc))
            continue
        if view_layer is not None:
            view_layer.update()
        report["applied"].append(role)

    if options.get("root_motion", False):
        _apply_root_translation(arm_obj, bmap, metrics, recon, options, report)

    # IK mode: limbs that can be written directly get the pose written straight into the IK controls (no FK switch, no snap)
    direct_done: dict = {}
    if direct_plans:
        direct_done = apply_ik_direct(arm_obj, direct_plans, targets, recon["refs"],
                                      metrics, options, report)

    close_limbs(arm_obj, written_limbs, limb_mode, report, options, limb_specs,
                direct=direct_done)

    if options.get("keyframe", False):
        frame = int(options.get("frame", bpy.context.scene.frame_current))
        insert_keyframes(arm_obj, bmap, list(report["applied"]), frame, report,
                         extra_bones=_limb_keyframe_bones(written_limbs, limb_mode),
                         prop_bones=[spec["prop_bone"] for spec in written_limbs.values()])

    report["ok"] = bool(report["applied"])
    report["message"] = "Wrote %d bones" % len(report["applied"])
    return report


