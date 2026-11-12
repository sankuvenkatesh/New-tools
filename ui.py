"""3D Viewport sidebar (N panel) user interface."""
from __future__ import annotations

import bpy

from . import deps, props, rigify_limbs, rigify_map


class PC_UL_bone_map(bpy.types.UIList):
    """Role -> bone mapping list; the bone name can be searched directly in each row."""

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _propname, _index):
        settings = bpy.context.scene.pose_capture
        armature = settings.armature
        row = layout.row(align=True)
        row.label(text=rigify_map.role_label(item.role))
        if armature is not None:
            row.prop_search(item, "bone", armature.data, "bones", text="")
            found = bool(item.bone) and item.bone in armature.data.bones
            row.label(text="", icon="CHECKMARK" if found else ("ERROR" if item.bone else "DOT"))
        else:
            row.prop(item, "bone", text="")


class PC_PT_main(bpy.types.Panel):
    bl_label = "Sanku AI Pose Studio"
    bl_idname = "PC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Sanku AI Pose Studio"

    def draw(self, context):
        layout = self.layout
        settings = props.get_settings(context)
        # Loading a new file clears the Python-side handlers, so re-install the watcher while drawing the panel (very cheap)
        rigify_limbs.ensure_watcher()

        # ---------------- Environment status ----------------
        status = deps.backend_status()
        backend_name = None
        if status["onnxruntime"]:
            backend_name = "onnxruntime %s" % status["onnxruntime"]
        elif status["opencv"]:
            backend_name = "opencv %s" % status["opencv"]
        _det, _pose, models_ready = props.resolve_model_paths()
        _depth_path, depth_ready = props.resolve_depth_model_path()

        box = layout.box()
        col = box.column(align=True)
        col.label(text="Dependencies: %s" % (backend_name or "not installed"),
                  icon="CHECKMARK" if backend_name else "ERROR")
        col.label(text="Models: %s" % ("ready" if models_ready else "not downloaded"),
                  icon="CHECKMARK" if models_ready else "ERROR")
        col.label(text="Depth model: %s" % ("ready" if depth_ready else "not downloaded"),
                  icon="CHECKMARK" if depth_ready else "INFO")
        if not (backend_name and models_ready):
            row = col.row(align=True)
            row.operator("pose_capture.install_deps", text="Install Dependencies", icon="IMPORT")
            row.operator("pose_capture.download_models", text="Download Models", icon="URL")

        # ---------------- 1 Input ----------------
        box = layout.box()
        box.label(text="1. Input Image", icon="IMAGE_DATA")
        col = box.column(align=True)
        col.prop(settings, "image_source", expand=True)
        if settings.image_source == "FILE":
            col.prop(settings, "image_filepath", text="")
        else:
            col.prop_search(settings, "image_name", bpy.data, "images", text="")
        col.prop(settings, "person_index")
        row = col.row(align=True)
        row.prop(settings, "det_thr")
        row.prop(settings, "score_thr")
        row = box.row(align=True)
        row.operator("pose_capture.detect", icon="PLAY",
                     text="Detect Pose" if not settings.detected else "Detect Again")
        row.operator("pose_capture.show_preview", text="", icon="IMAGE_REFERENCE")
        row.operator("pose_capture.show_depth", text="", icon="IMAGE_ZDEPTH")
        row.operator("pose_capture.export_json", text="", icon="EXPORT")
        # ---------------- 2 Target armature ----------------
        box = layout.box()
        box.label(text="2. Target Armature", icon="ARMATURE_DATA")
        row = box.row(align=True)
        row.prop(settings, "armature", text="")
        row.operator("pose_capture.use_active_armature", text="", icon="EYEDROPPER")
        row = box.row(align=True)
        row.operator("pose_capture.add_metarig", text="New Metarig")
        row.operator("pose_capture.generate_rig", text="Generate Rig")
        row = box.row(align=True)
        row.operator("pose_capture.auto_map", text="Auto-Match Bones")
        row.operator("pose_capture.reset_pose", text="Clear Pose")
        box.prop(settings, "auto_map_on_apply")
        if settings.armature is not None:
            box.label(text="Bones: %d" % len(settings.armature.data.bones))

        # ---------------- 3 Solving parameters ----------------
        box = layout.box()
        box.label(text="3. Pose Solving", icon="SETTINGS")
        col = box.column(align=True)
        col.prop(settings, "facing")
        col.prop(settings, "depth_mode")
        if settings.depth_mode == "DEPTH":
            col.label(text="Depth model: %s" % ("ready" if depth_ready else "not downloaded"),
                      icon="CHECKMARK" if depth_ready else "ERROR")
            if not depth_ready:
                col.operator("pose_capture.download_depth_model",
                             text="Download Depth Model", icon="URL")
            row = col.row(align=True)
            row.prop(settings, "depth_focal")
            row.prop(settings, "depth_flip", text="Flip Direction")
            if settings.depth_preview_name:
                row = col.row(align=True)
                row.operator("pose_capture.show_depth", text="View Depth Map",
                             icon="IMAGE_REFERENCE")
                row.operator("pose_capture.open_depth_view", text="Depth Preview Window",
                             icon="WINDOW")
        elif settings.depth_mode == "BONE":
            col.label(text="Old method: front/back direction relies on fixed assumptions (elbows back / knees forward)", icon="INFO")
        row = col.row(align=True)
        row.prop(settings, "twist")
        row.prop(settings, "twist_blend", text="Strength")
        col.prop(settings, "swap_lr")
        col.prop(settings, "fingers")
        col.prop(settings, "hand_align")
        col.prop(settings, "scale_hint")
        if settings.depth_info:
            box.label(text=settings.depth_info[:80], icon="INFO")
        if settings.depth_direction:
            mismatch = "★" in settings.depth_direction
            box.label(text=settings.depth_direction[:110],
                      icon="ERROR" if mismatch else "CHECKMARK")

        # ---------------- 4 Write ----------------
        box = layout.box()
        box.label(text="4. Write Pose", icon="POSE_HLT")
        col = box.column(align=True)
        col.prop(settings, "limb_mode")
        if settings.armature is not None and rigify_limbs.limb_specs(settings.armature):
            modes = rigify_limbs.limb_modes(settings.armature)
            text = ", ".join("%s=%s" % (name.split(".")[0], "FK" if value >= 0.5 else "IK")
                             for name, value in sorted(modes.items()))
            col.label(text="Current sliders: %s" % (text or "none"), icon="INFO")
            col.operator("pose_capture.sync_limb_display", icon="FILE_REFRESH",
                         text="Sync Control Display (click after dragging a Rigify slider)")
        col.prop(settings, "sync_visibility")
        col.prop(settings, "root_motion")
        if settings.root_motion:
            col.prop(settings, "ground_align")
        row = col.row(align=True)
        row.prop(settings, "keyframe")
        row.label(text="Current frame %d" % context.scene.frame_current)
        box.operator("pose_capture.apply_pose", icon="CHECKMARK", text="Apply to Armature")


class PC_PT_mapping(bpy.types.Panel):
    bl_label = "Bone Mapping (editable)"
    bl_idname = "PC_PT_mapping"
    bl_parent_id = "PC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = props.get_settings(context)
        if settings.armature is None:
            layout.label(text="Please specify the target armature first", icon="INFO")
            return
        if not len(settings.bone_map):
            layout.label(text="Click 'Auto-Match Bones' above to build the mapping table", icon="INFO")
            return
        layout.template_list("PC_UL_bone_map", "", settings, "bone_map",
                             settings, "bone_map_index", rows=8)


class PC_PT_log(bpy.types.Panel):
    bl_label = "Log"
    bl_idname = "PC_PT_log"
    bl_parent_id = "PC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = props.get_settings(context)
        if settings.status:
            layout.label(text=settings.status, icon="TIME")
        lines = [line for line in (settings.log or "").splitlines() if line.strip()]
        for line in lines[-12:]:
            layout.label(text=line[:90])


CLASSES = (PC_UL_bone_map, PC_PT_main, PC_PT_mapping, PC_PT_log)

