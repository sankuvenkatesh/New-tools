"""Rigify bone mapping: maps "semantic roles" onto concrete bone names of the armature, and reads the rest-pose dimensions.

Two kinds of armature are supported at the same time:
  * Rigify-generated rigs: spine_fk / upper_arm_fk.L / thigh_fk.L / neck / head / torso ...
  * Rigify metarigs or ordinary armatures: spine / spine.001 / upper_arm.L / thigh.L ...
"""
from __future__ import annotations

import math

from mathutils import Matrix, Vector

# Candidate bone names per role ({s} is replaced with L / R; matched in priority order from top to bottom)
ROLE_DEFS = [
    ("root", "Root bone / overall translation", ["root", "torso"]),
    ("hips", "Pelvis / torso orientation (needs a Rigify-generated rig)", ["torso"]),
    ("neck", "Neck", ["neck", "neck_fk", "spine.005", "ORG-spine.005", "DEF-spine.005"]),
    ("head", "Head", ["head", "head_fk", "spine.006", "ORG-spine.006", "DEF-spine.006"]),
    ("shoulder.{s}", "Clavicle {s}", ["shoulder.{s}", "shoulder_fk.{s}", "ORG-shoulder.{s}",
                                  "DEF-shoulder.{s}"]),
    ("upper_arm.{s}", "Upper arm {s}", ["upper_arm_fk.{s}", "upper_arm.{s}", "upper_arm_ik.{s}",
                                   "ORG-upper_arm.{s}", "DEF-upper_arm.{s}"]),
    ("forearm.{s}", "Forearm {s}", ["forearm_fk.{s}", "forearm.{s}", "forearm_ik.{s}",
                                 "ORG-forearm.{s}", "DEF-forearm.{s}"]),
    ("hand.{s}", "Hand {s}", ["hand_fk.{s}", "hand.{s}", "hand_ik.{s}",
                              "ORG-hand.{s}", "DEF-hand.{s}"]),
    ("palm.{s}", "Palm {s} (optional)", ["palm.{s}", "palm.02.{s}", "palm.01.{s}", "DEF-palm.01.{s}"]),
    ("thigh.{s}", "Thigh {s}", ["thigh_fk.{s}", "thigh.{s}", "thigh_ik.{s}",
                               "ORG-thigh.{s}", "DEF-thigh.{s}"]),
    ("shin.{s}", "Shin {s}", ["shin_fk.{s}", "shin.{s}", "shin_ik.{s}",
                              "ORG-shin.{s}", "DEF-shin.{s}"]),
    ("foot.{s}", "Foot {s}", ["foot_fk.{s}", "foot.{s}", "foot_ik.{s}",
                              "ORG-foot.{s}", "DEF-foot.{s}"]),
    ("toe.{s}", "Toe {s}", ["toe_fk.{s}", "toe.{s}", "toe_ik.{s}",
                             "ORG-toe.{s}", "DEF-toe.{s}"]),
    ("thumb.01.{s}", "Thumb 1 {s}", ["thumb.01.{s}", "DEF-thumb.01.{s}"]),
    ("thumb.02.{s}", "Thumb 2 {s}", ["thumb.02.{s}", "DEF-thumb.02.{s}"]),
    ("thumb.03.{s}", "Thumb 3 {s}", ["thumb.03.{s}", "DEF-thumb.03.{s}"]),
    ("f_index.01.{s}", "Index 1 {s}", ["f_index.01.{s}", "DEF-f_index.01.{s}"]),
    ("f_index.02.{s}", "Index 2 {s}", ["f_index.02.{s}", "DEF-f_index.02.{s}"]),
    ("f_index.03.{s}", "Index 3 {s}", ["f_index.03.{s}", "DEF-f_index.03.{s}"]),
    ("f_middle.01.{s}", "Middle 1 {s}", ["f_middle.01.{s}", "DEF-f_middle.01.{s}"]),
    ("f_middle.02.{s}", "Middle 2 {s}", ["f_middle.02.{s}", "DEF-f_middle.02.{s}"]),
    ("f_middle.03.{s}", "Middle 3 {s}", ["f_middle.03.{s}", "DEF-f_middle.03.{s}"]),
    ("f_ring.01.{s}", "Ring 1 {s}", ["f_ring.01.{s}", "DEF-f_ring.01.{s}"]),
    ("f_ring.02.{s}", "Ring 2 {s}", ["f_ring.02.{s}", "DEF-f_ring.02.{s}"]),
    ("f_ring.03.{s}", "Ring 3 {s}", ["f_ring.03.{s}", "DEF-f_ring.03.{s}"]),
    ("f_pinky.01.{s}", "Pinky 1 {s}", ["f_pinky.01.{s}", "DEF-f_pinky.01.{s}"]),
    ("f_pinky.02.{s}", "Pinky 2 {s}", ["f_pinky.02.{s}", "DEF-f_pinky.02.{s}"]),
    ("f_pinky.03.{s}", "Pinky 3 {s}", ["f_pinky.03.{s}", "DEF-f_pinky.03.{s}"]),
]

SPINE_SLOTS = 6
SPINE_MAX_BONES = 5  # at most 5 spine segments take part
SPINE_PATTERNS = ["spine_fk", "spine", "ORG-spine", "DEF-spine"]

FINGERS = ("thumb", "f_index", "f_middle", "f_ring", "f_pinky")

# Rigify IK controls that are matched first when "Limb Write Mode = IK" (only usable when the bone really exists,
# otherwise we fall back to the normal candidates in ROLE_DEFS, so metarigs / ordinary armatures are not affected).
# See `rigify_limbs._DEFAULT_CHAINS`: the chain root and end controls are upper_arm_ik.L / hand_ik.L /
# thigh_ik.L / foot_ik.L / toe_ik.L, and the middle segments (forearm_ik / shin_ik) are decided by the IK solve.
IK_ROLE_CANDIDATES = {
    "upper_arm.{s}": ["upper_arm_ik.{s}"],
    "forearm.{s}": ["forearm_ik.{s}"],
    "hand.{s}": ["hand_ik.{s}"],
    "thigh.{s}": ["thigh_ik.{s}"],
    "shin.{s}": ["shin_ik.{s}"],
    "foot.{s}": ["foot_ik.{s}"],
    "toe.{s}": ["toe_ik.{s}"],
}


def role_label(role: str) -> str:
    for rid, label, _cands in ROLE_DEFS:
        if rid == role:
            return label
        if "{s}" in rid:
            # Sided roles are stored as "shoulder.{s}": match "shoulder.L" / "shoulder.R" and fill in the side.
            # (Without this the list showed raw ids such as "shoulder.L" instead of "Clavicle L".)
            for side in ("L", "R"):
                if rid.format(s=side) == role:
                    return label.format(s=side)
    if role.startswith("spine_"):
        return "Spine segment %d" % (int(role.split("_")[1]) + 1)
    return role


def all_role_ids(include_fingers: bool = True) -> list[str]:
    roles = []
    for role, _label, _cands in ROLE_DEFS:
        if "{s}" in role:
            if not include_fingers and any(role.startswith(f) for f in FINGERS):
                continue
            roles.extend([role.format(s="L"), role.format(s="R")])
        else:
            roles.append(role)
    roles.extend(["spine_%02d" % i for i in range(SPINE_SLOTS)])
    return roles


class BoneMap:
    """Result of the role -> actual bone name mapping."""

    def __init__(self):
        self.roles: dict[str, str] = {}
        self.missing: list[str] = []

    def get(self, role: str) -> str:
        return self.roles.get(role, "")

    def set(self, role: str, name: str) -> None:
        if name:
            self.roles[role] = name
        else:
            self.roles.pop(role, None)

    def spine_bones(self) -> list[str]:
        names = []
        for i in range(SPINE_SLOTS):
            name = self.roles.get("spine_%02d" % i, "")
            if name:
                names.append(name)
        return names

    def used_bones(self) -> list[str]:
        return list(self.roles.values())


# --------------------------------------------------------------------------- auto detection
def _collect_indexed(names: list[str], pattern: str) -> list[str]:
    """Collect bones named pattern and pattern.NN, sorted by their number."""
    import re

    rx = re.compile(r"^%s(?:\.(\d+))?$" % re.escape(pattern))
    found = []
    for name in names:
        match = rx.match(name)
        if match:
            found.append((int(match.group(1) or 0), name))
    return [name for _index, name in sorted(found)]


def _pick(name_set, candidates: list[str]) -> str:
    for cand in candidates:
        if cand in name_set:
            return cand
    return ""


def ordered_candidates(role_id: str, side: str = "", limb_mode: str = None) -> list[str]:
    """Candidate bone names of a role (``{s}`` replaced with side).

    With ``limb_mode="IK"`` the IK controls (``upper_arm_ik.L`` / ``hand_ik.L`` ...) are moved to the front:
    the arms / thighs of a Rigify-generated rig are driven by IK by default, so a pose written to the FK bones would not be visible.
    In FK mode, or when ``limb_mode`` is not given, the original FK-first order is kept (compatible with metarigs / ordinary armatures).
    """
    for rid, _label, cands in ROLE_DEFS:
        if rid != role_id:
            continue
        ordered: list[str] = []
        if str(limb_mode or "").upper() == "IK":
            ordered.extend(name.format(s=side) for name in IK_ROLE_CANDIDATES.get(rid, ()))
        for name in cands:
            resolved = name.format(s=side)
            if resolved not in ordered:
                ordered.append(resolved)
        return ordered
    return []


def auto_detect(arm_obj, include_fingers: bool = True, limb_mode: str = None) -> BoneMap:
    """Automatically match role bone names on the armature.

    ``limb_mode`` follows the "Limb Write Mode" setting: with ``"IK"`` the limbs prefer
    ``*_ik`` (IK controls), with ``"FK"`` or when not given they prefer ``*_fk``.
    """
    bones = arm_obj.data.bones
    names = [bone.name for bone in bones]
    name_set = set(names)
    bmap = BoneMap()

    for role, _label, cands in ROLE_DEFS:
        if "{s}" in role:
            if not include_fingers and any(role.startswith(f) for f in FINGERS):
                continue
            for side in ("L", "R"):
                rid = role.format(s=side)
                resolved = _pick(name_set, ordered_candidates(role, side, limb_mode))
                if resolved:
                    bmap.set(rid, resolved)
                else:
                    bmap.missing.append(rid)
        else:
            resolved = _pick(name_set, list(cands))
            if resolved:
                bmap.set(role, resolved)
            else:
                bmap.missing.append(role)

    # Spine chain: take the naming style with the most matches, and cut it off at "shoulder height"
    shoulder_z = None
    for side in ("L", "R"):
        name = bmap.get("upper_arm.%s" % side)
        if name:
            z = bones[name].head_local.z
            shoulder_z = z if shoulder_z is None else max(shoulder_z, z)

    best_chain: list[str] = []
    for pattern in SPINE_PATTERNS:  # take the first naming style that matches, in priority order
        chain = _collect_indexed(names, pattern)
        if chain:
            best_chain = chain
            break

    if shoulder_z is not None and best_chain:
        limited = [name for name in best_chain
                   if bones[name].head_local.z < shoulder_z + 1e-4]
        if limited:
            best_chain = limited
    best_chain = best_chain[:SPINE_MAX_BONES]

    for index, name in enumerate(best_chain):
        bmap.set("spine_%02d" % index, name)
    for index in range(len(best_chain), SPINE_SLOTS):
        bmap.missing.append("spine_%02d" % index)

    return bmap


def read_metrics(arm_obj, bmap: BoneMap) -> dict:
    """Read the lengths / directions / reference axes in the rest pose, used for 2D pose solving and retargeting."""
    bones = arm_obj.data.bones
    metrics: dict = {"errors": [], "warnings": []}

    def bone(role: str):
        name = bmap.get(role)
        return bones.get(name) if name else None

    spine_names = bmap.spine_bones()
    if not spine_names:
        metrics["errors"].append("No spine bones found (spine / spine_fk chain)")
        return metrics

    spine_bones = [bones[name] for name in spine_names]
    metrics["spine_names"] = spine_names
    metrics["pelvis"] = Vector(spine_bones[0].head_local)
    metrics["chest"] = Vector(spine_bones[-1].tail_local)
    metrics["spine_len"] = float(sum(item.length for item in spine_bones))

    shoulder_heads = [bone("upper_arm.%s" % side) for side in ("L", "R")]
    hip_heads = [bone("thigh.%s" % side) for side in ("L", "R")]
    if any(item is None for item in shoulder_heads):
        metrics["errors"].append("No upper arm bone found (upper_arm)")
    if any(item is None for item in hip_heads):
        metrics["errors"].append("No thigh bone found (thigh)")
    if metrics["errors"]:
        return metrics

    metrics["shoulder_joints"] = {side: Vector(item.head_local)
                                  for side, item in zip(("L", "R"), shoulder_heads)}
    metrics["hip_joints"] = {side: Vector(item.head_local)
                             for side, item in zip(("L", "R"), hip_heads)}
    metrics["shoulder_line"] = (metrics["shoulder_joints"]["L"]
                                - metrics["shoulder_joints"]["R"])
    metrics["hip_line"] = metrics["hip_joints"]["L"] - metrics["hip_joints"]["R"]

    # The Rigify spine root (hips) sits slightly below the thigh roots and the top of the chest slightly above the shoulder joints:
    # record these two offsets, and the reconstruction shifts them by the same proportion so the clavicle / spine directions look natural.
    hip_avg_z = (metrics["hip_joints"]["L"].z + metrics["hip_joints"]["R"].z) * 0.5
    shoulder_avg_z = (metrics["shoulder_joints"]["L"].z
                      + metrics["shoulder_joints"]["R"].z) * 0.5
    metrics["pelvis_offset_z"] = float(metrics["pelvis"].z - hip_avg_z)
    metrics["chest_offset_z"] = float(metrics["chest"].z - shoulder_avg_z)
    metrics["shoulder_mid"] = Vector(((metrics["shoulder_joints"]["L"]
                                       + metrics["shoulder_joints"]["R"]) * 0.5))
    metrics["hip_mid"] = Vector(((metrics["hip_joints"]["L"]
                                  + metrics["hip_joints"]["R"]) * 0.5))
    # The keypoint "hip" corresponds to the thigh root of the armature, so we measure "midpoint of the thigh roots -> midpoint of the shoulder joints"
    metrics["pelvis_to_shoulders"] = float((metrics["shoulder_mid"]
                                            - metrics["hip_mid"]).length)

    # Rigid offsets on the torso (they rotate together with the torso during reconstruction)
    chest = metrics["chest"]
    metrics["hip_offsets"] = {side: Vector(metrics["hip_joints"][side] - metrics["pelvis"])
                              for side in ("L", "R")}
    metrics["shoulder_offsets"] = {side: Vector(metrics["shoulder_joints"][side] - chest)
                                   for side in ("L", "R")}
    clavicle_offsets = {}
    for side in ("L", "R"):
        item = bone("shoulder.%s" % side)
        clavicle_offsets[side] = Vector(item.head_local - chest) if item is not None else None
    metrics["clavicle_offsets"] = clavicle_offsets
    neck_bone = bone("neck")
    metrics["neck_offset"] = (Vector(neck_bone.head_local - chest) if neck_bone is not None
                              else Vector((0.0, 0.0, 0.0)))
    head_bone = bone("head")
    metrics["head_offset"] = (Vector(head_bone.head_local - chest) if head_bone is not None
                              else Vector((0.0, 0.0, 0.0)))

    def chain_len(*roles) -> float:
        return float(sum(bone(role).length for role in roles if bone(role) is not None))

    metrics["upper_arm_len"] = {s: chain_len("upper_arm.%s" % s) for s in ("L", "R")}
    metrics["forearm_len"] = {s: chain_len("forearm.%s" % s) for s in ("L", "R")}
    metrics["hand_len"] = {s: chain_len("hand.%s" % s, "palm.%s" % s) for s in ("L", "R")}
    metrics["thigh_len"] = {s: chain_len("thigh.%s" % s) for s in ("L", "R")}
    metrics["shin_len"] = {s: chain_len("shin.%s" % s) for s in ("L", "R")}
    metrics["foot_len"] = {s: chain_len("foot.%s" % s) for s in ("L", "R")}
    metrics["toe_len"] = {s: chain_len("toe.%s" % s) for s in ("L", "R")}
    metrics["neck_len"] = chain_len("neck")
    metrics["head_len"] = chain_len("head")

    metrics["rest_dir"] = {}
    metrics["rest_mat"] = {}
    for role, name in bmap.roles.items():
        item = bones.get(name)
        if item is None:
            continue
        direction = Vector(item.tail_local) - Vector(item.head_local)
        if direction.length > 1e-9:
            metrics["rest_dir"][role] = direction.normalized()
        metrics["rest_mat"][role] = item.matrix_local.to_3x3()

    metrics["ref"] = {"hip_line": metrics["hip_line"],
                      "shoulder_line": metrics["shoulder_line"]}
    # The rest-side "eye line" (person right -> left). It lets the head roll / head turn actually take effect:
    # the head reference axis in retarget is "eye_line"; on the solved side it is now given by the facial features (eye corners /
    # mouth corners / brow tips / jaw corners voted in pairs). The rest side used to have **no** such key: solve_orientation got
    # rest_ref=None and silently skipped the twist, so all head turn / tilt information from the photo was dropped (the head always faced the rest direction).
    # Here we use the shoulder line (or the hip line if there is none): every rig has a different roll convention for the head bone itself, whereas the shoulder / hip line
    # naturally defines "person left/right" (.L is on the person's left), and in a rest A/T pose the head is aligned with the shoulder line anyway.
    head_lateral = metrics["shoulder_line"]
    if head_lateral.length < 1e-9:
        head_lateral = metrics["hip_line"]
    if head_lateral.length > 1e-9:
        metrics["ref"]["eye_line"] = head_lateral.normalized()
    # The bend axes (elbow / knee) deliberately do NOT get the "align to shoulder/hip line" sign flip:
    # what carries information is using the same cross-product convention for rest and target (upper x lower, fixed order);
    # flipping the sign by a reference line turns into noise when the bend axis is nearly perpendicular to it, making roll flip randomly.
    for side in ("L", "R"):
        upper, fore = bone("upper_arm.%s" % side), bone("forearm.%s" % side)
        thigh, shin = bone("thigh.%s" % side), bone("shin.%s" % side)
        if upper and fore:
            axis = (Vector(upper.tail_local) - Vector(upper.head_local)).cross(
                Vector(fore.tail_local) - Vector(fore.head_local))
            if axis.length > 1e-6:
                metrics["ref"]["armbend.%s" % side] = axis.normalized()
        if thigh and shin:
            axis = (Vector(thigh.tail_local) - Vector(thigh.head_local)).cross(
                Vector(shin.tail_local) - Vector(shin.head_local))
            if axis.length > 1e-6:
                metrics["ref"]["kneebend.%s" % side] = axis.normalized()

    low_z = None
    for role in ("foot.L", "foot.R", "toe.L", "toe.R"):
        item = bone(role)
        if item is None:
            continue
        for point in (item.head_local, item.tail_local):
            low_z = point.z if low_z is None else min(low_z, point.z)
    metrics["lowest_foot_z"] = 0.0 if low_z is None else float(low_z)
    metrics["pelvis_height"] = float(metrics["pelvis"].z - metrics["lowest_foot_z"])
    return metrics


