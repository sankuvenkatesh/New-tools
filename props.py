"""Scene properties and preferences of the add-on."""
import os

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import AddonPreferences, PropertyGroup

from . import deps, depth, reconstruction, rigify_map, utils


class PC_BoneMapping(PropertyGroup):
    """One "role -> bone" mapping row."""

    role: StringProperty(name="Role")
    label: StringProperty(name="Description")
    bone: StringProperty(name="Bone")


class PC_Settings(PropertyGroup):
    # ---------------- Input ----------------
    image_source: EnumProperty(
        name="Image Source",
        items=[("FILE", "Image File", "Pick an image from disk"),
               ("DATA", "Loaded Image", "Use an image datablock that is already loaded in Blender")],
        default="FILE")
    image_filepath: StringProperty(name="Image", subtype="FILE_PATH", default="")
    image_name: StringProperty(name="Image", default="")
    person_index: IntProperty(name="Person Index", default=0, min=0, max=32,
                              description="Which person to use when the picture contains several (0 = the largest person)")

    # ---------------- Target ----------------
    armature: PointerProperty(
        name="Target Armature", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE")
    bone_map: CollectionProperty(type=PC_BoneMapping)
    bone_map_index: IntProperty(default=0)
    map_ready: BoolProperty(default=False)
    auto_map_on_apply: BoolProperty(
        name="Auto-Match Bone Names", default=True,
        description="On Apply, re-run automatic matching if the bone names are stale or the limb bones "
                    "do not match the Limb Write Mode (IK selected but FK bones mapped); "
                    "manually edited rows will be overwritten")

    # ---------------- Detection parameters ----------------
    score_thr: FloatProperty(name="Keypoint Conf.", default=0.3, min=0.05, max=1.0)
    det_thr: FloatProperty(name="Person Box Conf.", default=0.3, min=0.05, max=1.0)

    # ---------------- Pose solving parameters ----------------
    facing: EnumProperty(
        name="Person Facing",
        items=[(key, reconstruction.FACING_LABELS[key], "")
               for key in reconstruction.FACING_DIRS],
        default="FRONT")
    depth_mode: EnumProperty(
        name="Depth Source",
        items=[(key, reconstruction.DEPTH_MODE_LABELS[key],
                "DEPTH: use the depth model (Depth Anything V2) to solve a real depth for every joint, "
                "front/back direction comes from the pixels; BONE: the old bone-length constraint solve "
                "(front/back direction relies on fixed assumptions); "
                "FLAT: no depth, the whole pose is flattened into the image plane")
               for key in reconstruction.DEPTH_MODES],
        default="DEPTH")
    depth_focal: FloatProperty(
        name="Perspective Strength", default=1.2, min=0.3, max=4.0,
        description="Focal length factor: f = factor x the longer image side. Larger = longer lens (weaker perspective); "
                    "increase it if the depth solve looks too flat, decrease it if it looks exaggerated")
    depth_flip: BoolProperty(
        name="Flip Depth Direction", default=False,
        description="Normally the direction is decided automatically by the fit; when the fit cannot tell front from back, "
                    "a physical criterion (the person is closer than the background) decides the sign, so you do not "
                    "need to tick this. Tick it only if the result is really reversed (for example the person is behind "
                    "glass or a mirror, where the physical criterion does not apply): it flips the whole pose "
                    "front-to-back and disables the physical criterion. The depth preview prints a 'Depth direction' "
                    "line you can check")
    swap_lr: BoolProperty(name="Swap Left/Right", default=False,
                          description="Only tick this if the detector swaps the person's left and right")
    twist: BoolProperty(name="Estimate Bone Twist (roll)", default=True,
                        description="Align the bone axes using the shoulder line / hip line / bend axes, so arms and head do not get twisted")
    twist_blend: FloatProperty(name="Twist Strength", default=1.0, min=0.0, max=1.0)
    scale_hint: FloatProperty(name="Scale Fine-Tune", default=1.0, min=0.5, max=2.0,
                              description="Slightly adjust the 'keypoint pixels / armature metres' ratio (normally no need to change)")
    fingers: BoolProperty(name="Write Fingers", default=False,
                          description="Only enable when the finger keypoints are of good quality")
    hand_align: FloatProperty(
        name="Hand Along Forearm", default=0.5, min=0.0, max=1.0,
        description="Fraction by which the hand (palm + fingers) is pulled toward the forearm direction so that it follows "
                    "the forearm: 0 = no pull (only clamped when the deviation exceeds 30 degrees), "
                    "0.5 = deviation halved (default), 1 = fully along the forearm. Hand keypoints only reach the "
                    "finger bases, so a fist / side-on palm / misplaced fingertips can give a very large deviation "
                    "(palm sticking out sideways from the wrist), which is why the default pulls halfway toward the forearm; "
                    "the hand's shape and length stay the same, only its orientation relative to the forearm changes")
    root_motion: BoolProperty(name="Write Root Translation", default=False,
                              description="Write the person's horizontal position / standing height to the root bone")
    ground_align: BoolProperty(name="Ground Align", default=True)
    limb_mode: EnumProperty(
        name="Limb Write Mode",
        items=[("IK", "IK Controls (Rigify default)",
                "Write the pose straight into the IK controls hand_ik / foot_ik / upper_arm_ik / thigh_ik "
                "(no FK switch, no IK->FK snap); the FK chain is written too, so the pose is not lost when you drag "
                "back to FK. Automatic bone matching also prefers the IK controls for limbs (recommended)"),
               ("FK", "FK Bones",
                "The pose stays on the FK bones (IK_FK slider = 1) and the IK controls are hidden; "
                "automatic bone matching prefers *_fk for limbs")],
        default="IK")
    sync_visibility: BoolProperty(
        name="Sync Control Visibility", default=True,
        description="IK mode shows only the IK controls, FK mode only the FK controls, so that 'visible but does nothing' "
                    "is not mistaken for a bug (you can switch them back by hand in the Bone Collections panel)")
    keyframe: BoolProperty(name="Insert Keyframe", default=False)

    # ---------------- State ----------------
    status: StringProperty(name="Status", default="")
    log: StringProperty(name="Log", default="")
    detected: BoolProperty(default=False)
    preview_name: StringProperty(default="")
    detect_info: StringProperty(default="")
    depth_info: StringProperty(default="")
    depth_direction: StringProperty(default="")
    depth_preview_name: StringProperty(default="")


class PC_Preferences(AddonPreferences):
    bl_idname = __package__

    backend: EnumProperty(
        name="Inference Backend",
        items=[("auto", "Auto (onnxruntime preferred)", ""),
               ("onnxruntime", "onnxruntime", ""),
               ("opencv", "OpenCV DNN", "")],
        default="auto")
    backend_package: EnumProperty(
        name="Package to Install",
        items=[(name, name, "") for name in deps.BACKEND_PACKAGE_IDS],
        default="onnxruntime")
    device: EnumProperty(
        name="Inference Device",
        items=[("auto", "Auto", "Pick automatically from the available providers"),
               ("cpu", "CPU", ""),
               ("dml", "DirectML", ""),
               ("cuda", "CUDA", "")],
        default="auto")
    local_dwpose_root: StringProperty(
        name="Local Model Folder (optional)", subtype="DIR_PATH", default="",
        description="A folder that already contains downloaded DWPose / Easy-DWPose models; the add-on searches it recursively for onnx files")
    local_depth_model: StringProperty(
        name="Depth Model File (optional)", subtype="FILE_PATH", default="",
        description="An existing depth model onnx (Depth Anything / MiDaS etc.). Leave empty to use the one in the add-on model folder")
    depth_view_python: StringProperty(
        name="Preview Window Interpreter (optional)", subtype="FILE_PATH", default="",
        description="A python.exe that can run tkinter (Blender's bundled Python has no tkinter). "
                    "Leave empty to search PATH and the common install locations automatically")
    depth_variant: EnumProperty(
        name="Depth Model Variant", items=depth.variant_items(), default=depth.DEFAULT_VARIANT,
        description="fp32 is the most accurate (about 95 MB), fp16 is in between (about 48 MB), int8 is the smallest and fastest (about 27 MB)")
    auto_install: BoolProperty(name="Auto-Install Missing Dependencies", default=True)
    verbose: BoolProperty(name="Verbose Logging", default=True)

    def draw(self, context):
        layout = self.layout
        status = deps.backend_status()

        box = layout.box()
        box.label(text="Dependency Status", icon="INFO")
        row = box.row()
        row.label(text="onnxruntime: %s" % (status["onnxruntime"] or "not installed"))
        row.label(text="opencv: %s" % (status["opencv"] or "not installed"))
        if status.get("providers"):
            box.label(text="Available providers: %s" % ", ".join(status["providers"]))
        box.label(text="Dependency folder: %s" % utils.libs_dir())
        box.label(text="Model folder: %s" % utils.models_dir())
        box.label(text="Bundled Python: %s" % (utils.python_executable() or "not found"))

        row = layout.row(align=True)
        row.prop(self, "backend")
        row.prop(self, "device")
        row = layout.row(align=True)
        row.prop(self, "backend_package", text="")
        row.operator("pose_capture.install_deps", text="Install / Update Dependencies", icon="IMPORT")
        layout.prop(self, "auto_install")
        layout.prop(self, "verbose")

        box = layout.box()
        box.label(text="Model Source", icon="FILE_FOLDER")
        box.prop(self, "local_dwpose_root")
        row = box.row(align=True)
        row.operator("pose_capture.import_local_models", text="Import from Local Folder")
        row.operator("pose_capture.download_models", text="Download Models Online")

        box = layout.box()
        box.label(text="Depth Model (solves real depth for the joints)", icon="IMAGE_ZDEPTH")
        _path, ready = resolve_depth_model_path()
        row = box.row(align=True)
        row.prop(self, "depth_variant", text="")
        row.label(text="Ready" if ready else "Not downloaded",
                  icon="CHECKMARK" if ready else "ERROR")
        box.prop(self, "local_depth_model")
        row = box.row(align=True)
        row.operator("pose_capture.download_depth_model", text="Download Depth Model", icon="URL")
        row.operator("pose_capture.import_depth_model", text="Import Depth Model", icon="FILE_FOLDER")

        box = layout.box()
        box.label(text="Depth Preview Window (separate Tk window)", icon="IMAGE_ZDEPTH")
        row = box.row(align=True)
        row.operator("pose_capture.open_depth_view", text="Open Preview Window", icon="WINDOW")
        box.prop(self, "depth_view_python")
        box.label(text="Blender's bundled Python has no tkinter; a system Python is needed here", icon="INFO")


# --------------------------------------------------------------------------- helper functions
def get_settings(context=None) -> PC_Settings:
    context = context or bpy.context
    return context.scene.pose_capture


def get_prefs(context=None) -> PC_Preferences:
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    if addon is not None and addon.preferences is not None:
        return addon.preferences
    return context.preferences.addons[__package__].preferences


def ensure_mapping_rows(settings: PC_Settings) -> None:
    """Make sure the mapping table has a row for every role (existing user settings are not overwritten)."""
    existing = {row.role for row in settings.bone_map}
    for role in rigify_map.all_role_ids(include_fingers=True):
        if role in existing:
            continue
        row = settings.bone_map.add()
        row.role = role
        row.label = rigify_map.role_label(role)


def mapping_from_settings(settings: PC_Settings) -> rigify_map.BoneMap:
    bmap = rigify_map.BoneMap()
    for row in settings.bone_map:
        if row.bone:
            bmap.set(row.role, row.bone)
    return bmap


def sync_mapping_to_settings(settings: PC_Settings, bmap: rigify_map.BoneMap) -> None:
    ensure_mapping_rows(settings)
    for row in settings.bone_map:
        row.bone = bmap.get(row.role) or ""
    settings.map_ready = True


def options_from_settings(settings: PC_Settings) -> dict:
    return {
        "score_thr": settings.score_thr,
        "facing": settings.facing,
        "depth_mode": settings.depth_mode,
        "depth_focal": settings.depth_focal,
        "depth_flip": settings.depth_flip,
        "swap_lr": settings.swap_lr,
        "fingers": settings.fingers,
        "hand_align": settings.hand_align,
        "twist": settings.twist,
        "twist_blend": settings.twist_blend,
        "scale_hint": settings.scale_hint,
        "root_motion": settings.root_motion,
        "ground_align": settings.ground_align,
        "keep_depth": True,
        "fk_mode": settings.limb_mode == "FK",
        "limb_mode": settings.limb_mode,
        "sync_visibility": settings.sync_visibility,
        "keyframe": settings.keyframe,
    }


def resolve_model_paths() -> tuple[str, str, bool]:
    """Return (detector model, pose model, whether both are ready)."""
    prefs = get_prefs()
    det, pose = deps.model_paths()

    override = getattr(prefs, "local_dwpose_root", "")
    if override and (not os.path.isfile(det) or not os.path.isfile(pose)):
        found = deps.find_local_models(override)
        if found:
            det, pose = found

    ready = os.path.isfile(det) and os.path.isfile(pose)
    return det, pose, ready


def resolve_depth_model_path() -> tuple[str, bool]:
    """Return (depth model path, whether it is usable).

    Priority: the file specified by hand in the preferences > the matching variant in the add-on model
    folder > a depth*.onnx in the local model folder (re-using one already downloaded, e.g. for ComfyUI).
    """
    prefs = None
    try:
        prefs = get_prefs()
    except Exception:  # noqa: BLE001  (no preferences entry when registered manually from a dev script)
        prefs = None
    override = getattr(prefs, "local_depth_model", "")
    if override:
        path = bpy.path.abspath(override)
        if os.path.isfile(path):
            return path, True

    variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
    path = deps.depth_model_path(variant)
    if os.path.isfile(path):
        return path, True

    root = getattr(prefs, "local_dwpose_root", "")
    found = deps.find_local_depth_model(root) if root else None
    if found:
        return found, True
    return path, False


def resolve_providers():
    prefs = get_prefs()
    mapping = {"cpu": ["CPUExecutionProvider"],
               "dml": ["DmlExecutionProvider"],
               "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    device = getattr(prefs, "device", "auto")
    providers = mapping.get(device)
    if device == "auto" or providers is None:
        return None
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        filtered = [name for name in providers if name in available]
        return filtered or None
    except Exception:  # noqa: BLE001
        return None

