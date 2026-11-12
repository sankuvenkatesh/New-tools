"""Sanku AI Pose Studio (based on Pose Capture) - detect a human skeleton from an image and convert it into a Pose usable by a Rigify armature.

Pipeline: image -> DWPose (ONNX) keypoints -> depth model (Depth Anything V2) solves a real depth for every joint
      -> 2D->3D perspective mapping -> Rigify pose bone rotations
"""
bl_info = {
    "name": "Sanku AI Pose Studio",
    "author": "Sanku Venkatesh",
    "version": (1, 3, 4),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Sanku AI Pose Studio",
    "description": "Image-to-pose generation for Rigify characters using DWPose and depth estimation",
    "category": "Animation",
}

import bpy

from . import operators, props, rigify_limbs, ui, utils

_CLASSES = (
    props.PC_BoneMapping,
    props.PC_Settings,
    props.PC_Preferences,
) + operators.CLASSES + ui.CLASSES


def register():
    utils.ensure_libs_on_path()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pose_capture = bpy.props.PointerProperty(type=props.PC_Settings)
    rigify_limbs.ensure_watcher()
    utils.log("Sanku AI Pose Studio enabled. Dependency folder: %s" % utils.libs_dir())


def unregister():
    rigify_limbs.remove_watcher()
    if hasattr(bpy.types.Scene, "pose_capture"):
        del bpy.types.Scene.pose_capture
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    register()
