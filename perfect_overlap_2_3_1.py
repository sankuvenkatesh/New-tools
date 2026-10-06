# ##### BEGIN GPL LICENSE BLOCK #####
#
# Perfect Overlap Addon
# Version 2.5.0 - armature-space tail-spring overlap solver.
#
# Copyright 2021-2026 CaptainHansode, sakaiden.com
# Copyright 2026 Sanku Venkatesh
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Perfect Overlap",
    "author": "Sanku Venkatesh",
    "version": (2, 5, 0),
    "blender": (4, 4, 0),
    "location": "3D Viewport > Sidebar > Perfect Overlap (Pose Mode)",
    "description": (
        "Overlap and follow-through animation for bone chains. "
        "Simulates a spring-lagged bone tip and re-aims the bone at it."
    ),
    "category": "Animation",
}

import bpy
import math
import mathutils
import traceback

__author__ = "Sanku Venkatesh"
__copyright__ = (
    "Copyright 2021-2026 CaptainHansode, sakaiden.com; "
    "2026 Sanku Venkatesh"
)
__license__ = "GPL"
__status__ = "Production"

REQUIRED_BLENDER = (4, 4, 0)

EPS = 1.0e-6
ZERO_V = mathutils.Vector((0.0, 0.0, 0.0))
ONE_V = mathutils.Vector((1.0, 1.0, 1.0))
IDENTITY_Q = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))

MAX_TIP_LAG_RATIO = 1.0
MAX_HEAD_LAG_RATIO = 1.0
MAX_SWING_ANGLE = math.radians(150.0)

CYCLE_MODIFIER_NAME = "Perfect Overlap Cycle"

NON_CONTROL_PREFIXES = (
    "def-", "def_", "def.",
    "org-", "org_", "org.",
    "mch-", "mch_", "mch.",
    "wgt-", "wgt_", "wgt.",
    "vis-", "vis_", "vis.",
)

ROTATION_PATHS = (
    "rotation_quaternion",
    "rotation_euler",
    "rotation_axis_angle",
)


# ============================================================================
# Math helpers
# ============================================================================

def is_finite_vector(v):
    try:
        return all(math.isfinite(float(x)) for x in v)
    except (TypeError, ValueError, AttributeError):
        return False


def is_finite_quaternion(q):
    try:
        return all(math.isfinite(float(x)) for x in q)
    except (TypeError, ValueError, AttributeError):
        return False


def safe_vector(v, fallback=None):
    if fallback is None:
        fallback = ZERO_V
    try:
        out = v.copy()
        if not is_finite_vector(out):
            return fallback.copy()
        return out
    except Exception:
        return fallback.copy()


def safe_quaternion(q, fallback=None):
    """Normalize a quaternion. Uses .magnitude - mathutils.Quaternion
    has no .length (that was the crash bug in 2.3.x)."""
    if fallback is None:
        fallback = IDENTITY_Q
    try:
        out = q.copy()
        mag = float(out.magnitude)
        if not math.isfinite(mag) or mag < EPS or not is_finite_quaternion(out):
            return fallback.copy()
        out.normalize()
        return out
    except Exception:
        return fallback.copy()


def safe_invert(m):
    try:
        return m.inverted()
    except (ValueError, ArithmeticError):
        return mathutils.Matrix.Identity(len(m))


def quat_angle_between(a, b):
    """Smallest rotation angle between two quaternions (handles q == -q)."""
    qa = safe_quaternion(a)
    qb = safe_quaternion(b)
    dot = min(1.0, abs(float(qa.dot(qb))))
    return 2.0 * math.acos(dot)


def axis_angle_to_quaternion(aa):
    try:
        angle = float(aa[0])
        axis = mathutils.Vector((float(aa[1]), float(aa[2]), float(aa[3])))
        if axis.length < EPS or not math.isfinite(angle):
            return IDENTITY_Q.copy()
        axis.normalize()
        return safe_quaternion(mathutils.Quaternion(axis, angle))
    except (TypeError, ValueError, IndexError):
        return IDENTITY_Q.copy()


# ============================================================================
# Solver
# ============================================================================

class PerfectOverlapSolver:
    """Spring-lagged tip overlap solver.

    Pipeline:
        1. build_chains()    select the chains to overlap
        2. sample_source()   read the source animation (read-only)
        3. solve()           simulate overlap in pure math
        4. delete_keys()     remove old keys from the range
        5. write_results()   write the new rotation keys

    All math is done in armature-object space (the space of
    pose_bone.matrix). The armature object's own transform is never read.
    """

    def __init__(self):
        self.sf = 0
        self.ef = 100
        self.delay = 3.0
        self.recursion = 5.0
        self.strength = 1.0
        self.threshold = 0.001
        self.animate_translate = False
        self.debug = False

        self.key_failures = set()
        self.lookup_failures = set()
        self.pose_failures = set()
        self.translate_skipped = []

    # ------------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------------

    def configure(self, sf, ef, delay, recursion, strength, threshold,
                  animate_translate, debug=False):
        self.sf = int(sf)
        self.ef = int(ef)
        self.delay = float(delay)
        self.recursion = float(recursion)
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.animate_translate = bool(animate_translate)
        self.debug = bool(debug)
        return self

    def solver_params(self):
        # k is the per-frame pull toward the target. Larger delay = lower k.
        k = 1.0 / max(min(self.delay, 10.0), 1.0)

        # Damping from recursion: 0 -> heavy damping, 10 -> light damping.
        rec = max(0.0, min(10.0, self.recursion))
        damping = 0.76 - (rec / 10.0) * 0.34
        damping = max(0.38, min(0.76, damping))

        # Strength amplifies the tip offset before re-aim. 1 -> 1x, 10 -> 2x.
        strength = max(1.0, min(10.0, self.strength))
        amp = 1.0 + (strength - 1.0) / 9.0
        return k, damping, amp

    def _log(self, msg):
        if self.debug:
            print("[Perfect Overlap] {}".format(msg))

    def _fail(self, collection, message):
        if message:
            collection.add(str(message))

    # ------------------------------------------------------------------------
    # Rig inspection
    # ------------------------------------------------------------------------

    def _visible(self, pbn):
        try:
            if pbn.bone.hide:
                return False
        except (AttributeError, ReferenceError):
            return False
        colls = getattr(pbn.bone, "collections", None)
        if not colls:
            return True
        for c in colls:
            vis = getattr(c, "is_visible_effectively", None)
            if vis is None:
                vis = getattr(c, "is_visible_with_ancestors", None)
            if vis is None:
                vis = getattr(c, "is_visible", True)
            if vis:
                return True
        return False

    def _is_control(self, pbn):
        try:
            if pbn.name.lower().startswith(NON_CONTROL_PREFIXES):
                return False
        except (AttributeError, ReferenceError):
            return False
        return self._visible(pbn)

    def _control_children(self, pbn, depth=0):
        if depth >= 8:
            return []
        result = []
        try:
            children = pbn.children
        except (AttributeError, ReferenceError):
            return result
        for c in children:
            if self._is_control(c):
                result.append(c)
            else:
                result.extend(self._control_children(c, depth + 1))
        return result

    def _next_in_chain(self, pbn, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        for c in candidates:
            try:
                if c.bone.use_connect:
                    return c
            except (AttributeError, ReferenceError):
                pass
        try:
            tail = pbn.tail
            direction = (pbn.tail - pbn.head).normalized()
            scale = max(float(pbn.length), EPS)
        except Exception:
            return candidates[0]
        best, best_score = None, None
        for c in candidates:
            try:
                dist = (c.head - tail).length / scale
                c_vec = c.tail - c.head
                c_dir = c_vec.normalized() if c_vec.length > EPS else -direction
                align = float(c_dir.dot(direction))
                score = dist - align
                if best_score is None or score < best_score:
                    best_score, best = score, c
            except Exception:
                continue
        return best or candidates[0]

    @staticmethod
    def _depth(pbn):
        d = 0
        cur = pbn
        while cur is not None:
            d += 1
            cur = cur.parent
        return d

    # ------------------------------------------------------------------------
    # Chain building
    # ------------------------------------------------------------------------

    def build_chains(self):
        """Return a list of chains. Each chain is a list of pose bones,
        ordered parents-first. Chains are also sorted parents-first."""
        obj = bpy.context.active_object
        if obj is None or obj.type != 'ARMATURE':
            return []

        selected = []
        for pbn in (bpy.context.selected_pose_bones or []):
            try:
                if pbn.id_data == obj:
                    selected.append(pbn)
            except (AttributeError, ReferenceError):
                pass
        if not selected:
            return []

        names = {p.name for p in selected}
        ordered = sorted(selected, key=self._depth)
        used = set()
        chains = []

        for pbn in ordered:
            if pbn.name in used:
                continue
            if pbn.parent is None:
                # A parentless control is a driver, not part of the chain.
                continue
            chain = [pbn]
            used.add(pbn.name)
            cur = pbn
            while True:
                cands = [
                    c for c in self._control_children(cur)
                    if c.name in names and c.name not in used
                ]
                nxt = self._next_in_chain(cur, cands)
                if nxt is None:
                    break
                chain.append(nxt)
                used.add(nxt.name)
                cur = nxt
            chains.append(chain)

        chains.sort(key=lambda ch: self._depth(ch[0]))
        return chains

    @staticmethod
    def iter_bones(chains):
        """Unique bones across all chains, parents-first."""
        seen = set()
        for chain in chains:
            for pbn in chain:
                if pbn.name in seen:
                    continue
                seen.add(pbn.name)
                yield pbn

    # ------------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------------

    @staticmethod
    def _active_quat(pbn):
        """Read the quaternion from the bone's ACTIVE rotation channel."""
        mode = pbn.rotation_mode
        try:
            if mode == 'QUATERNION':
                return safe_quaternion(pbn.rotation_quaternion)
            if mode == 'AXIS_ANGLE':
                return axis_angle_to_quaternion(pbn.rotation_axis_angle)
            return safe_quaternion(pbn.rotation_euler.to_quaternion())
        except Exception:
            return IDENTITY_Q.copy()

    def sample_source(self, chains):
        """Read the source animation across the frame range.

        Returns a dict with:
            frames       {frame: {bone_name: {"matrix", "local_loc",
                                              "local_quat", "local_scale"}}}
            rest_rel     {bone_name: M4}
            chain_names  set of chain bone names
            driver_names set of names of parents outside the chain
        or None on failure.
        """
        scene = bpy.context.scene
        obj = bpy.context.active_object
        if obj is None:
            return None
        original_frame = scene.frame_current

        bones = list(self.iter_bones(chains))
        if not bones:
            return None
        chain_names = {pbn.name for pbn in bones}

        rest_rel = {}
        for pbn in bones:
            parent = pbn.parent
            if parent is None:
                continue
            rest_rel[pbn.name] = (
                safe_invert(parent.bone.matrix_local) @ pbn.bone.matrix_local
            )

        driver_names = set()
        for pbn in bones:
            parent = pbn.parent
            if parent is not None and parent.name not in chain_names:
                driver_names.add(parent.name)

        cache = {}
        try:
            for f in range(self.sf, self.ef + 1):
                scene.frame_set(f)
                bpy.context.view_layer.update()
                frame_data = {}
                for pbn in bones:
                    try:
                        frame_data[pbn.name] = {
                            "matrix": pbn.matrix.copy(),
                            "local_loc": safe_vector(pbn.location),
                            "local_quat": self._active_quat(pbn),
                            "local_scale": safe_vector(pbn.scale, ONE_V),
                        }
                    except (ReferenceError, AttributeError):
                        continue
                for name in driver_names:
                    pbn = obj.pose.bones.get(name)
                    if pbn is None:
                        continue
                    try:
                        frame_data.setdefault(name, {})["matrix"] = (
                            pbn.matrix.copy()
                        )
                    except (ReferenceError, AttributeError):
                        pass
                cache[f] = frame_data
        finally:
            scene.frame_set(original_frame)
            bpy.context.view_layer.update()

        if not cache:
            return None

        return {
            "frames": cache,
            "rest_rel": rest_rel,
            "chain_names": chain_names,
            "driver_names": driver_names,
            "bones": [pbn.name for pbn in bones],
        }

    # ------------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------------

    @staticmethod
    def _spring(pos, vel, target, k, damping, amp):
        v = vel * damping + (target - pos) * k
        if not is_finite_vector(v):
            v = ZERO_V.copy()
        new_pos = pos + v * amp
        return new_pos, v

    def _aim_quaternion(self, pose_x, source_quat, source_head,
                        source_tail, wanted_tail):
        """Local quaternion that re-aims the bone from its source direction
        toward the lagged direction. The source roll is preserved."""
        aim_now = source_tail - source_head
        aim_new = wanted_tail - source_head
        if aim_now.length < EPS or aim_new.length < EPS:
            return source_quat
        try:
            delta = aim_now.rotation_difference(aim_new)
            if delta.w < 0.0:
                delta.negate()
            angle = float(delta.angle)
            if angle > MAX_SWING_ANGLE:
                delta = IDENTITY_Q.slerp(delta, MAX_SWING_ANGLE / angle)
            qx = pose_x.to_quaternion()
            result = qx.conjugated() @ delta @ qx @ source_quat
            result.normalize()
            return safe_quaternion(result, source_quat)
        except (ValueError, ArithmeticError, TypeError):
            return source_quat

    def solve(self, chains, source, cycle=False, preroll=0):
        """Simulate overlap. Returns {frame: {bone: {"rotation", "location"}}}.

        No Blender data is touched.
        """
        frames = list(range(self.sf, self.ef + 1))
        cache = source["frames"]
        if frames[0] not in cache:
            return None

        k, damping, amp = self.solver_params()

        # Build a table of chain bones and their static properties.
        info = {}
        for pbn in self.iter_bones(chains):
            if pbn.name not in source["chain_names"]:
                continue
            parent = pbn.parent
            if parent is None:
                continue
            axes = self._translate_axes(pbn)
            info[pbn.name] = {
                "bone": pbn,
                "parent": parent.name,
                "rest_rel": source["rest_rel"].get(pbn.name),
                "length": max(float(pbn.length), EPS),
                "axes": axes,
            }
            if self.animate_translate and not axes:
                self.translate_skipped.append(pbn.name)

        ordered = sorted(info.keys(), key=lambda n: self._depth(info[n]["bone"]))
        if not ordered:
            return None

        # Per-bone state for the two springs (tip and optional head).
        state = {}
        for n in ordered:
            state[n] = {
                "tip": None,
                "tip_vel": ZERO_V.copy(),
                "head": None,
                "head_vel": ZERO_V.copy(),
            }

        def step_frame(frame_idx, snap=False, reset_vel=False):
            """One integration step across all bones. Returns output dict."""
            frame_data = cache.get(frame_idx, {})
            solved_poses = {}
            out = {}

            for name in ordered:
                meta = info[name]
                src = frame_data.get(name)
                if src is None:
                    continue

                parent_name = meta["parent"]
                parent_pose = solved_poses.get(parent_name)
                if parent_pose is None:
                    # Parent is a driver: use its sampled pose.
                    par = frame_data.get(parent_name)
                    if par is None:
                        continue
                    parent_pose = par.get("matrix")
                    if parent_pose is None:
                        continue

                rest_rel = meta["rest_rel"]
                if rest_rel is None:
                    continue

                pose_x = parent_pose @ rest_rel
                source_loc = src["local_loc"]
                source_quat = src["local_quat"]
                source_scale = src["local_scale"]

                target_pose = pose_x @ mathutils.Matrix.LocRotScale(
                    source_loc, source_quat, source_scale,
                )
                target_head = target_pose.translation.copy()
                target_tail = target_pose @ mathutils.Vector(
                    (0.0, meta["length"], 0.0)
                )

                st = state[name]

                # --- Optional translation spring on the head ---
                use_loc = source_loc
                if meta["axes"]:
                    if snap or st["head"] is None:
                        st["head"] = target_head.copy()
                        if reset_vel or st["head_vel"] is None:
                            st["head_vel"] = ZERO_V.copy()
                    else:
                        new_head, hvel = self._spring(
                            st["head"], st["head_vel"], target_head,
                            k, damping, amp,
                        )
                        max_lag = meta["length"] * MAX_HEAD_LAG_RATIO
                        lag = new_head - target_head
                        if lag.length > max_lag:
                            new_head = target_head + lag.normalized() * max_lag
                            hvel = ZERO_V.copy()
                        st["head"] = new_head
                        st["head_vel"] = hvel

                    try:
                        local_head = safe_invert(pose_x) @ st["head"]
                        delta_loc = local_head - source_loc
                        for ax in range(3):
                            if ax not in meta["axes"]:
                                delta_loc[ax] = 0.0
                        use_loc = source_loc + delta_loc
                    except Exception:
                        use_loc = source_loc
                else:
                    st["head"] = target_head.copy()
                    st["head_vel"] = ZERO_V.copy()

                # --- Rotation overlap on the tip ---
                new_quat = source_quat
                if snap or st["tip"] is None:
                    st["tip"] = target_tail.copy()
                    if reset_vel or st["tip_vel"] is None:
                        st["tip_vel"] = ZERO_V.copy()
                else:
                    new_tip, tvel = self._spring(
                        st["tip"], st["tip_vel"], target_tail,
                        k, damping, amp,
                    )
                    max_lag = meta["length"] * MAX_TIP_LAG_RATIO
                    lag = new_tip - target_tail
                    if lag.length > max_lag:
                        new_tip = target_tail + lag.normalized() * max_lag
                        tvel = ZERO_V.copy()
                    st["tip"] = new_tip
                    st["tip_vel"] = tvel
                    new_quat = self._aim_quaternion(
                        pose_x, source_quat,
                        target_head, target_tail, new_tip,
                    )

                out[name] = {
                    "rotation": new_quat.copy(),
                    "location": use_loc.copy() if meta["axes"] else None,
                }

                # The solved pose of this bone drives its children.
                solved_poses[name] = pose_x @ mathutils.Matrix.LocRotScale(
                    use_loc, new_quat, source_scale,
                )

            return out

        if not cycle:
            # Simple pass: snap at start, then step through the range.
            results = {
                frames[0]: step_frame(frames[0], snap=True, reset_vel=True),
            }
            for f in frames[1:]:
                results[f] = step_frame(f)
            self._log(
                "solved {} bones over {} frames".format(len(ordered), len(frames))
            )
            return results

        # Cycle mode: pre-roll to settle the momentum, then record a clean pass.
        step_frame(frames[0], snap=True, reset_vel=True)
        passes = max(1, int(preroll))
        for _ in range(passes):
            for f in frames[1:]:
                step_frame(f)
        # Record the full loop (state persists from the pre-roll).
        results = {}
        for f in frames:
            results[f] = step_frame(f)
        self._log(
            "cycle: {} pre-roll passes + 1 recorded pass".format(passes)
        )
        return results

    def _translate_axes(self, pbn):
        if not self.animate_translate:
            return []
        try:
            if pbn.bone.use_connect:
                return []
            return [i for i in range(3) if not pbn.lock_location[i]]
        except (AttributeError, ReferenceError):
            return []

    # ------------------------------------------------------------------------
    # Pose capture / restore
    # ------------------------------------------------------------------------

    def capture_pose(self, chains):
        snap = {}
        for pbn in self.iter_bones(chains):
            snap[pbn.name] = {
                "location": pbn.location.copy(),
                "rotation_quaternion": pbn.rotation_quaternion.copy(),
                "rotation_euler": pbn.rotation_euler.copy(),
                "rotation_axis_angle": tuple(
                    float(v) for v in pbn.rotation_axis_angle
                ),
                "scale": pbn.scale.copy(),
            }
        return snap

    def restore_pose(self, chains, snap):
        if not snap:
            return
        for pbn in self.iter_bones(chains):
            vals = snap.get(pbn.name)
            if vals is None:
                continue
            pbn.location = vals["location"].copy()
            pbn.rotation_quaternion = vals["rotation_quaternion"].copy()
            e = vals["rotation_euler"]
            pbn.rotation_euler = mathutils.Euler((e[0], e[1], e[2]), e.order)
            pbn.rotation_axis_angle = vals["rotation_axis_angle"]
            pbn.scale = vals["scale"].copy()
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------------
    # Slotted-Action / F-Curve access
    # ------------------------------------------------------------------------

    def _ensure_action_slot(self, id_data):
        """Assign a slot for this ID if the Action has none yet."""
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
            adt.action_slot = adt.action.slots.new(
                id_type=id_data.id_type,
                name=id_data.name,
            )
            return True
        except (AttributeError, RuntimeError, TypeError):
            return False

    def _fcurve_container(self, id_data):
        if id_data is None:
            return None
        try:
            adt = id_data.animation_data
        except AttributeError:
            return None
        if adt is None or adt.action is None:
            return None
        if not self._ensure_action_slot(id_data):
            return None

        slot = getattr(adt, "action_slot", None)
        if slot is None:
            return None

        try:
            from bpy_extras import anim_utils
            helper = getattr(
                anim_utils,
                "animdata_get_channelbag_for_assigned_slot",
                None,
            )
            channelbag = helper(adt) if helper is not None else None
            if channelbag is None:
                fallback = getattr(
                    anim_utils, "action_get_channelbag_for_slot", None,
                )
                if fallback is not None:
                    channelbag = fallback(adt.action, slot)
            if channelbag is None:
                return None
            return channelbag.fcurves
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            self._fail(self.lookup_failures, "channelbag: {}".format(exc))
            return None

    def _bones_fcurve_map(self, chains):
        obj = bpy.context.active_object
        if obj is None:
            return []
        container = self._fcurve_container(obj)
        if container is None:
            return []
        by_prefix = {
            pbn.path_from_id(): pbn for pbn in self.iter_bones(chains)
        }
        result = []
        for fc in list(container):
            for prefix, pbn in by_prefix.items():
                if fc.data_path.startswith(prefix + "."):
                    channel = fc.data_path[len(prefix) + 1:]
                    result.append((fc, pbn, channel))
                    break
        return result

    # ------------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------------

    @staticmethod
    def _rotation_path(pbn):
        if pbn.rotation_mode == 'QUATERNION':
            return 'rotation_quaternion'
        if pbn.rotation_mode == 'AXIS_ANGLE':
            return 'rotation_axis_angle'
        return 'rotation_euler'

    def _apply_rotation(self, pbn, quat, tracker):
        """Write the local rotation using the bone's active rotation mode,
        keeping it continuous with the previous frame's representation."""
        q = safe_quaternion(quat)
        mode = pbn.rotation_mode
        try:
            if mode == 'QUATERNION':
                prev = tracker.get("last_quat")
                if prev is not None and q.dot(prev) < 0.0:
                    q.negate()
                pbn.rotation_quaternion = q
                tracker["last_quat"] = q.copy()
                return

            if mode == 'AXIS_ANGLE':
                angle = float(q.angle)
                axis = q.axis.copy()
                prev = tracker.get("last_axis_angle")
                if prev is not None:
                    prev_angle = float(prev[0])
                    prev_axis = mathutils.Vector(prev[1:4])
                    if prev_axis.length >= EPS:
                        prev_axis.normalize()
                        if axis.dot(prev_axis) < 0.0:
                            axis.negate()
                            angle = -angle
                    angle += math.tau * round((prev_angle - angle) / math.tau)
                pbn.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
                tracker["last_axis_angle"] = (angle, axis.x, axis.y, axis.z)
                return

            euler = q.to_euler(mode)
            prev = tracker.get("last_euler")
            if prev is not None:
                try:
                    euler.make_compatible(prev)
                except (ValueError, ArithmeticError):
                    pass
            pbn.rotation_euler = euler
            tracker["last_euler"] = euler.copy()
        except Exception as exc:
            self._fail(
                self.pose_failures,
                "{}: rotation {}".format(pbn.name, exc),
            )

    def write_results(self, chains, results, start_pose):
        """Write the solved rotations (and optional locations) as keys."""
        scene = bpy.context.scene
        obj = bpy.context.active_object

        bones = {pbn.name: pbn for pbn in self.iter_bones(chains)}

        # Seed one continuity tracker per bone from the start pose.
        trackers = {}
        for name in bones:
            t = {}
            seed = start_pose.get(name) if start_pose else None
            if seed:
                try:
                    t["last_quat"] = safe_quaternion(seed["rotation_quaternion"])
                    t["last_euler"] = seed["rotation_euler"].copy()
                    t["last_axis_angle"] = tuple(
                        float(v) for v in seed["rotation_axis_angle"]
                    )
                except (KeyError, AttributeError, TypeError, ValueError):
                    pass
            trackers[name] = t

        self._ensure_action_slot(obj)

        for f in range(self.sf, self.ef + 1):
            frame_result = results.get(f)
            if not frame_result:
                continue
            scene.frame_set(f)
            for name, pbn in bones.items():
                res = frame_result.get(name)
                if res is None:
                    continue
                self._apply_rotation(pbn, res["rotation"], trackers[name])
                if res["location"] is not None:
                    try:
                        pbn.location = res["location"].copy()
                    except Exception as exc:
                        self._fail(
                            self.pose_failures,
                            "{}: location {}".format(name, exc),
                        )
                self._key_bone(pbn)
            bpy.context.view_layer.update()

    def _key_bone(self, pbn):
        f = bpy.context.scene.frame_current

        for ax in self._translate_axes(pbn):
            try:
                r = pbn.keyframe_insert(
                    data_path='location', index=ax, frame=f
                )
                if r is False:
                    self._fail(
                        self.key_failures,
                        "{} loc[{}]".format(pbn.name, ax),
                    )
            except Exception as exc:
                self._fail(
                    self.key_failures,
                    "{} loc[{}]: {}".format(pbn.name, ax, exc),
                )

        try:
            r = pbn.keyframe_insert(
                data_path=self._rotation_path(pbn), frame=f,
            )
            if r is False:
                self._fail(
                    self.key_failures, "{} rotation".format(pbn.name)
                )
        except Exception as exc:
            self._fail(
                self.key_failures, "{} rotation: {}".format(pbn.name, exc)
            )

    # ------------------------------------------------------------------------
    # Key deletion / reduction
    # ------------------------------------------------------------------------

    def delete_keys(self, chains):
        """Remove rotation keys and generated translation keys in [sf, ef]."""
        obj = bpy.context.active_object
        container = self._fcurve_container(obj)
        emptied = []

        for fc, pbn, channel in self._bones_fcurve_map(chains):
            if channel == 'location':
                if fc.array_index not in self._translate_axes(pbn):
                    continue
            elif channel not in ROTATION_PATHS:
                continue

            indices = [
                i for i, k in enumerate(fc.keyframe_points)
                if self.sf <= float(k.co[0]) <= self.ef
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

    def reduce_keys(self, chains):
        """Greedy per-group key reduction (rotation is treated as a group)."""
        tol = max(self.threshold, 1.0e-6)
        groups = {}
        for fc, pbn, channel in self._bones_fcurve_map(chains):
            if channel == 'location':
                if fc.array_index not in self._translate_axes(pbn):
                    continue
            elif channel not in ROTATION_PATHS:
                continue
            groups.setdefault((pbn.name, channel), []).append((fc, pbn))

        removed = 0
        for (name, channel), curves in groups.items():
            fcurves = [fc for fc, _ in curves]
            all_frames = None
            for fc in fcurves:
                cf = [
                    float(k.co[0]) for k in fc.keyframe_points
                    if self.sf <= float(k.co[0]) <= self.ef
                ]
                all_frames = (
                    cf if all_frames is None
                    else [f for f in all_frames if f in cf]
                )
            if not all_frames or len(all_frames) <= 2:
                self._smooth(fcurves)
                continue

            values = []
            ok = True
            for fc in fcurves:
                lookup = {
                    float(k.co[0]): float(k.co[1])
                    for k in fc.keyframe_points
                }
                try:
                    values.append([lookup[f] for f in all_frames])
                except KeyError:
                    ok = False
                    break
            if not ok:
                self._fail(
                    self.lookup_failures,
                    "{} {} missing key".format(name, channel),
                )
                self._smooth(fcurves)
                continue

            drop = set()
            prev = 0
            for i in range(1, len(all_frames) - 1):
                t0 = all_frames[prev]
                t1 = all_frames[i]
                t2 = all_frames[i + 1]
                if abs(t2 - t0) < EPS:
                    continue
                ratio = (t1 - t0) / (t2 - t0)
                redundant = True
                for comp in values:
                    pred = comp[prev] + (comp[i + 1] - comp[prev]) * ratio
                    if abs(comp[i] - pred) > tol:
                        redundant = False
                        break
                if redundant:
                    drop.add(t1)
                else:
                    prev = i

            if drop:
                for fc in fcurves:
                    kps = fc.keyframe_points
                    idx = [
                        n for n, k in enumerate(kps)
                        if float(k.co[0]) in drop
                    ]
                    for n in reversed(idx):
                        try:
                            kps.remove(kps[n], fast=True)
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
            for k in fc.keyframe_points:
                fr = float(k.co[0])
                if self.sf <= fr <= self.ef:
                    k.interpolation = 'BEZIER'
                    k.handle_left_type = 'AUTO_CLAMPED'
                    k.handle_right_type = 'AUTO_CLAMPED'
            try:
                fc.update()
            except Exception:
                pass

    # ------------------------------------------------------------------------
    # Cycle modifiers
    # ------------------------------------------------------------------------

    def add_cycle_modifiers(self, chains):
        for fc, pbn, channel in self._bones_fcurve_map(chains):
            if channel == 'location':
                if fc.array_index not in self._translate_axes(pbn):
                    continue
            elif channel not in ROTATION_PATHS:
                continue
            if any(m.type == 'CYCLES' for m in fc.modifiers):
                continue
            try:
                m = fc.modifiers.new('CYCLES')
                try:
                    m.name = CYCLE_MODIFIER_NAME
                except (AttributeError, RuntimeError):
                    pass
            except (RuntimeError, ReferenceError):
                pass

    def remove_cycle_modifiers(self, chains):
        for fc, _pbn, _ch in self._bones_fcurve_map(chains):
            for m in list(fc.modifiers):
                if (
                    m.type == 'CYCLES'
                    and getattr(m, "name", "") == CYCLE_MODIFIER_NAME
                ):
                    try:
                        fc.modifiers.remove(m)
                    except (RuntimeError, ReferenceError):
                        pass

    def close_cycle_seam(self, chains):
        """Copy start-frame values onto the end frame for every baked curve."""
        for fc, _pbn, _ch in self._bones_fcurve_map(chains):
            first = last = None
            for k in fc.keyframe_points:
                fr = float(k.co[0])
                if abs(fr - self.sf) < EPS:
                    first = k
                elif abs(fr - self.ef) < EPS:
                    last = k
            if first is None or last is None or first is last:
                continue
            last.co[1] = first.co[1]
            try:
                fc.update()
            except Exception:
                pass


# ============================================================================
# UI properties
# ============================================================================

class PERFECTOVERLAP_PG_props(bpy.types.PropertyGroup):
    start_frame: bpy.props.IntProperty(
        name="Start", default=0, min=-1048574, max=1048574,
    )
    end_frame: bpy.props.IntProperty(
        name="End", default=100, min=-1048574, max=1048574,
    )
    delay: bpy.props.FloatProperty(
        name="Delay", default=3.0, min=1.0, max=10.0,
        description="Lag time. Higher values make the chain trail further",
    )
    recursion: bpy.props.FloatProperty(
        name="Recursion", default=5.0, min=0.0, max=10.0,
        description="Momentum. 0 settles cleanly, 10 keeps swinging",
    )
    strength: bpy.props.FloatProperty(
        name="Strength", default=1.0, min=1.0, max=10.0,
        description="Exaggerate the overlap offset",
    )
    threshold: bpy.props.FloatProperty(
        name="Threshold", default=0.001, min=0.00001, max=0.1,
        step=0.01, precision=4,
        description="Key reduction tolerance",
    )
    animate_translate: bpy.props.BoolProperty(
        name="Translation", default=False,
        description="Also lag unlocked location channels",
    )
    cycle: bpy.props.BoolProperty(
        name="Cycle", default=False,
        description="Settle the loop with pre-roll and close the seam",
    )
    cycle_preroll: bpy.props.IntProperty(
        name="Pre-roll", default=2, min=1, max=20,
        description="Number of hidden passes used to settle cycle momentum",
    )
    debug: bpy.props.BoolProperty(
        name="Debug", default=False,
        description="Print solver details to the system console",
    )


# ============================================================================
# Operator helpers
# ============================================================================

def _blender_ok():
    return bpy.app.version >= REQUIRED_BLENDER


def _armature_ok(context):
    obj = context.active_object
    return (
        obj is not None
        and obj.type == 'ARMATURE'
        and context.mode == 'POSE'
    )


def _action_editable(obj):
    try:
        if not obj.is_editable:
            return False, "Armature is read-only"
    except AttributeError:
        if obj.library is not None and obj.override_library is None:
            return False, "Armature is linked and read-only"

    adt = obj.animation_data
    action = adt.action if adt else None
    if action is not None:
        try:
            if not action.is_editable:
                return False, "Action is read-only"
        except AttributeError:
            if action.library is not None and action.override_library is None:
                return False, "Action is linked and read-only"
    return True, ""


def _apply_props(solver, props):
    solver.configure(
        sf=props.start_frame,
        ef=props.end_frame,
        delay=props.delay,
        recursion=props.recursion,
        strength=props.strength,
        threshold=props.threshold,
        animate_translate=props.animate_translate,
        debug=props.debug,
    )
    return solver


def _format_failures(label, failures, limit=8):
    if not failures:
        return ""
    vals = sorted(str(v) for v in failures)
    if len(vals) > limit:
        return "{}: {} (+{} more)".format(
            label, ", ".join(vals[:limit]), len(vals) - limit,
        )
    return "{}: {}".format(label, ", ".join(vals))


def _report_failures(operator, solver):
    for label, failures in (
        ("Keying", solver.key_failures),
        ("Channels", solver.lookup_failures),
        ("Pose", solver.pose_failures),
    ):
        msg = _format_failures(label, failures)
        if msg:
            operator.report({'WARNING'}, msg)


# ============================================================================
# Operators
# ============================================================================

class PERFECTOVERLAP_OT_calculate(bpy.types.Operator):
    bl_idname = "perfect_overlap.calculate"
    bl_label = "Calculate"
    bl_description = "Bake overlap / follow-through on the selected chain"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        if not _blender_ok():
            self.report(
                {'ERROR'},
                "Blender {}.{}.{} or newer required".format(*REQUIRED_BLENDER),
            )
            return {'CANCELLED'}
        if not _armature_ok(context):
            self.report({'ERROR'}, "Must be in Pose Mode on an armature")
            return {'CANCELLED'}

        obj = context.active_object
        ok, why = _action_editable(obj)
        if not ok:
            self.report({'ERROR'}, why)
            return {'CANCELLED'}

        props = context.scene.perfect_overlap_props
        if props.start_frame >= props.end_frame:
            self.report({'ERROR'}, "Start frame must be less than end frame")
            return {'CANCELLED'}

        scene = context.scene
        original_frame = scene.frame_current
        solver = _apply_props(PerfectOverlapSolver(), props)

        chains = None
        snapshot = None
        touched = False
        result = {'FINISHED'}

        try:
            chains = solver.build_chains()
            if not chains:
                self.report(
                    {'WARNING'},
                    "No chain detected. Select the child controls; each "
                    "chain needs a parent driver.",
                )
                return {'CANCELLED'}

            snapshot = solver.capture_pose(chains)

            # ---- Phase 1: read ----
            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            source = solver.sample_source(chains)
            if not source:
                self.report({'ERROR'}, "Could not read the source animation")
                return {'CANCELLED'}

            # ---- Phase 2: solve (pure math) ----
            results = solver.solve(
                chains,
                source,
                cycle=bool(props.cycle),
                preroll=int(props.cycle_preroll) if props.cycle else 0,
            )
            if not results:
                self.report({'ERROR'}, "Solver produced no output")
                return {'CANCELLED'}

            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = solver.capture_pose(chains)

            # ---- Phase 3: write ----
            touched = True
            solver.delete_keys(chains)
            solver.restore_pose(chains, start_pose)
            solver.write_results(chains, results, start_pose)

            # ---- Post-processing (best effort) ----
            try:
                solver.reduce_keys(chains)
            except Exception as exc:
                traceback.print_exc()
                self.report(
                    {'WARNING'},
                    "Key reduction skipped: {}".format(exc),
                )

            if props.cycle:
                try:
                    solver.close_cycle_seam(chains)
                    solver.add_cycle_modifiers(chains)
                except Exception as exc:
                    traceback.print_exc()
                    self.report(
                        {'WARNING'},
                        "Cycle finalization skipped: {}".format(exc),
                    )
            else:
                try:
                    solver.remove_cycle_modifiers(chains)
                except Exception:
                    pass

            if props.animate_translate and solver.translate_skipped:
                self.report(
                    {'WARNING'},
                    "Translation skipped on {} connected/locked control(s)".format(
                        len(solver.translate_skipped),
                    ),
                )

            _report_failures(self, solver)

            bone_count = sum(1 for _ in solver.iter_bones(chains))
            self.report(
                {'INFO'},
                "Baked overlap on {} bone(s), frames {} to {}".format(
                    bone_count, props.start_frame, props.end_frame,
                ),
            )

        except Exception as exc:
            traceback.print_exc()
            if touched:
                # The animation was already partly rewritten. Return
                # FINISHED so Blender pushes one undo step for it.
                self.report(
                    {'ERROR'},
                    "Partial bake ({}). Press Ctrl+Z to undo.".format(exc),
                )
                result = {'FINISHED'}
            else:
                try:
                    if snapshot and chains:
                        solver.restore_pose(chains, snapshot)
                except Exception:
                    pass
                self.report(
                    {'ERROR'},
                    "Bake failed, animation unchanged: {}".format(exc),
                )
                result = {'CANCELLED'}

        finally:
            try:
                scene.frame_set(original_frame)
                bpy.context.view_layer.update()
            except Exception:
                pass

        return result


class PERFECTOVERLAP_OT_delete(bpy.types.Operator):
    bl_idname = "perfect_overlap.delete"
    bl_label = "Delete Keys"
    bl_description = "Delete Perfect Overlap keys on the selected chain"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        if not _blender_ok():
            self.report(
                {'ERROR'},
                "Blender {}.{}.{} or newer required".format(*REQUIRED_BLENDER),
            )
            return {'CANCELLED'}
        if not _armature_ok(context):
            self.report({'ERROR'}, "Must be in Pose Mode on an armature")
            return {'CANCELLED'}

        obj = context.active_object
        ok, why = _action_editable(obj)
        if not ok:
            self.report({'ERROR'}, why)
            return {'CANCELLED'}

        props = context.scene.perfect_overlap_props
        if props.start_frame >= props.end_frame:
            self.report({'ERROR'}, "Start frame must be less than end frame")
            return {'CANCELLED'}

        scene = context.scene
        original_frame = scene.frame_current
        solver = _apply_props(PerfectOverlapSolver(), props)

        chains = None
        snapshot = None
        touched = False
        result = {'FINISHED'}

        try:
            chains = solver.build_chains()
            if not chains:
                self.report({'WARNING'}, "No chain detected")
                return {'CANCELLED'}

            snapshot = solver.capture_pose(chains)
            scene.frame_set(props.start_frame)
            bpy.context.view_layer.update()
            start_pose = solver.capture_pose(chains)

            touched = True
            solver.delete_keys(chains)
            solver.restore_pose(chains, start_pose)

            _report_failures(self, solver)
            self.report(
                {'INFO'},
                "Deleted overlap keys on {} bone(s)".format(
                    sum(1 for _ in solver.iter_bones(chains)),
                ),
            )

        except Exception as exc:
            traceback.print_exc()
            if touched:
                self.report(
                    {'ERROR'},
                    "Partial delete ({}). Press Ctrl+Z to undo.".format(exc),
                )
                result = {'FINISHED'}
            else:
                try:
                    if snapshot and chains:
                        solver.restore_pose(chains, snapshot)
                except Exception:
                    pass
                self.report(
                    {'ERROR'},
                    "Delete failed, animation unchanged: {}".format(exc),
                )
                result = {'CANCELLED'}

        finally:
            try:
                scene.frame_set(original_frame)
                bpy.context.view_layer.update()
            except Exception:
                pass

        return result


class PERFECTOVERLAP_OT_reset(bpy.types.Operator):
    bl_idname = "perfect_overlap.reset_settings"
    bl_label = "Reset Settings"
    bl_description = "Reset all Perfect Overlap values to defaults"
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
        self.report({'INFO'}, "Settings reset to defaults")
        return {'FINISHED'}


# ============================================================================
# Panel
# ============================================================================

class PERFECTOVERLAP_PT_panel(bpy.types.Panel):
    bl_label = "Perfect Overlap"
    bl_idname = "PERFECTOVERLAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Perfect Overlap"
    bl_context = "posemode"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.perfect_overlap_props

        layout.label(text="Select chain bones (not the driver)")
        layout.separator()

        box = layout.box()
        box.label(text="Frame Range")
        row = box.row(align=True)
        row.prop(props, "start_frame")
        row.prop(props, "end_frame")

        box = layout.box()
        box.label(text="Solver")
        row = box.row(align=True)
        row.prop(props, "delay")
        row.prop(props, "recursion")
        row.prop(props, "strength")
        box.prop(props, "threshold")

        box = layout.box()
        box.label(text="Options")
        box.prop(props, "animate_translate")
        box.prop(props, "cycle")
        if props.cycle:
            box.prop(props, "cycle_preroll")

        box = layout.box()
        box.prop(props, "debug")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.6
        row.operator(
            "perfect_overlap.calculate", icon="KEYTYPE_KEYFRAME_VEC",
        )
        row = layout.row()
        row.operator("perfect_overlap.delete", icon="KEYFRAME")
        row = layout.row()
        row.operator("perfect_overlap.reset_settings", icon="LOOP_BACK")


# ============================================================================
# Registration
# ============================================================================

classes = (
    PERFECTOVERLAP_PG_props,
    PERFECTOVERLAP_OT_calculate,
    PERFECTOVERLAP_OT_delete,
    PERFECTOVERLAP_OT_reset,
    PERFECTOVERLAP_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.perfect_overlap_props = bpy.props.PointerProperty(
        type=PERFECTOVERLAP_PG_props,
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
