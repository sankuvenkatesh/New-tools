"""Operators: install dependencies, download models, detect pose, map bones, write the pose onto the armature."""
import json
import os
import queue
import subprocess
import sys
import threading
import traceback

import numpy as np

import bpy

from . import (coco, deps, depth, depth_view, dwpose, imops, props, reconstruction,
               retarget, rigify_limbs, rigify_map, utils)

# Session cache for models / detection results (not saved in the .blend; detection must be repeated after reopening the file)
_ESTIMATOR_CACHE = {"key": None, "obj": None}
_DEPTH_CACHE = {"key": None, "obj": None}
_RESULT = {"keypoints": None, "scores": None, "boxes": None, "box": None,
           "image_size": None, "path": None, "rgb": None, "person_index": 0,
           "depth_map": None, "depth_size": None}


# --------------------------------------------------------------------------- shared helpers
def load_source_image(settings) -> np.ndarray:
    if settings.image_source == "DATA":
        image = bpy.data.images.get(settings.image_name)
        if image is None:
            raise ValueError("No loaded image found: %s" % settings.image_name)
        return utils.image_datablock_to_rgb8(image)
    return utils.load_image_rgb8(bpy.path.abspath(settings.image_filepath))


def get_estimator(detector: str, pose: str, backend: str, providers):
    key = (os.path.abspath(detector), os.path.abspath(pose), backend, tuple(providers or ()))
    if _ESTIMATOR_CACHE["key"] != key or _ESTIMATOR_CACHE["obj"] is None:
        _ESTIMATOR_CACHE["obj"] = dwpose.DwPoseEstimator(detector, pose, backend, providers)
        _ESTIMATOR_CACHE["key"] = key
    return _ESTIMATOR_CACHE["obj"]


def get_depth_estimator(path: str, backend: str, providers):
    """The depth model is cached too (loading takes about 1 s, each inference afterwards about 0.3 s)."""
    key = (os.path.abspath(path), backend, tuple(providers or ()))
    if _DEPTH_CACHE["key"] != key or _DEPTH_CACHE["obj"] is None:
        _DEPTH_CACHE["obj"] = depth.DepthEstimator(path, backend, providers)
        _DEPTH_CACHE["key"] = key
    return _DEPTH_CACHE["obj"]


def _write_image(name: str, canvas: np.ndarray) -> str:
    """Write an (H,W,3) uint8 canvas into a Blender image datablock (re-created when the size changes).

    Note (verified with Blender 5.2): assigning colorspace_settings.name on a GENERATED image **after the pixels were written**
    (even to the same 'sRGB') makes Blender rebuild the pixel buffer and the content turns completely black - this was
    the reason why "the depth map / skeleton preview was always pure black" before.
    So the colour space must be set before writing the pixels, and not touched afterwards.
    """
    height, width = canvas.shape[:2]

    image = bpy.data.images.get(name)
    if image is not None and (image.size[0] != width or image.size[1] != height):
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(name, width=width, height=height, alpha=False)

    try:
        if image.colorspace_settings.name != "sRGB":
            image.colorspace_settings.name = "sRGB"
    except Exception:  # noqa: BLE001  some generated images have no colour space settings
        pass

    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = canvas.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    rgba = rgba[::-1]  # Blender stores rows bottom-to-top
    image.pixels.foreach_set(np.ascontiguousarray(rgba).reshape(-1))
    image.update()
    image.use_fake_user = True
    return image.name


def update_preview_image(name: str, rgb: np.ndarray, keypoints, scores, thr: float) -> str:
    return _write_image(name, imops.draw_pose_preview(rgb, keypoints, scores, thr))


def update_depth_preview_image(name: str, rgb: np.ndarray, depth_map,
                               near_is_large: bool, keypoints=None, scores=None,
                               thr: float = 0.3) -> str:
    """False-colour depth (near=blue / far=red) + skeleton overlay, to check the depth direction and quality by eye."""
    canvas = depth.preview_image(rgb, depth_map, near_is_large)
    if keypoints is not None and scores is not None:
        canvas = imops.draw_pose_preview(canvas, keypoints, scores, thr)
    return _write_image(name, canvas)


def depth_orientation(depth_map: np.ndarray, box, image_size) -> bool:
    """Decide whether "a large value" in the depth map means near or far, using the fact that "the person is closer to the camera than the background".

    Returns True for "large value = near" (used as the colour direction of the false-colour preview).
    """
    if box is None or depth_map is None:
        return True
    height, width = depth_map.shape[:2]
    scale_x = width / max(float(image_size[0]), 1e-6)
    scale_y = height / max(float(image_size[1]), 1e-6)
    x0 = int(max(0, float(box[0]) * scale_x))
    x1 = int(min(width, float(box[2]) * scale_x))
    y0 = int(max(0, float(box[1]) * scale_y))
    y1 = int(min(height, float(box[3]) * scale_y))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return True
    inside = float(np.mean(depth_map[y0:y1, x0:x1]))
    mask = np.ones(depth_map.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    if not mask.any():
        return True
    outside = float(np.mean(depth_map[mask]))
    return inside > outside


def depth_direction(depth_map: np.ndarray, box, image_size, recon=None) -> dict:
    """Depth direction check: the **physical criterion** (the person is closer than the background) vs the **fit criterion** (the sign of alpha in the reconstruction).

    The preview colours always use the physical criterion: it depends only on the pixels and does not flip together with "Flip Depth Direction",
    so it works as an independent check of the fit direction - if the two disagree, the 3D pose is reversed front-to-back as a whole.

    Returns ``{"physical", "fit", "agree", "near_is_large", "text", "warning"}``,
    where ``near_is_large`` (True = large value = near) is the direction that should be used to draw the preview.
    """
    physical = depth_orientation(depth_map, box, image_size)
    info = (recon or {}).get("depth") or {}
    rejected = bool(info.get("rejected"))     # the fit was judged untrustworthy and the body text used the bone-length solve
    fit = bool(info["near_is_large"]) if ("near_is_large" in info and not rejected) else None

    def word(flag):
        return "near" if flag else "far"

    if box is None:
        text = "Depth direction: no person box, coloured with the default 'large = near' (near=blue far=red)"
    else:
        text = "Depth direction: large = %s (physical: the person is closer than the background; false colour near=blue far=red)" % word(physical)
    if fit is not None:
        text += "; fit says 'large = %s' %s" % (word(fit), "✓" if fit == physical else "✗")
    elif rejected:
        text += "; the depth fit was judged untrustworthy (fell back to the bone-length solve), direction not checked"

    warning = ""
    if fit is not None and fit != physical:
        warning = ("The depth direction does not match the fit: the physical criterion says 'large = %s', but the reconstruction used "
                   "'large = %s', so the whole pose may be reversed front-to-back - tick 'Flip Direction' and click "
                   "'Apply to Armature' again" % (word(physical), word(fit)))
    return {"physical": bool(physical), "fit": fit, "agree": fit is None or fit == physical,
            "near_is_large": bool(physical), "text": text, "warning": warning}


def recon_summary(recon: dict) -> str:
    """Write "which solver was used, the fit parameters, the fallback hints" as one line, for checking in the panel / log."""
    labels = {"DEPTH": "depth model", "BONE": "bone-length solve", "FLAT": "flattened"}
    mode = recon.get("mode", "?")
    parts = ["Solver: %s" % labels.get(mode, mode)]
    info = recon.get("depth") or {}
    if info:
        parts.append("%s, alpha=%.3f, beta=%.2fm, bone-length residual %.3f/%.3f m"
                     % (info.get("transform", "?"), float(info.get("alpha") or 0.0),
                        float(info.get("beta") or 0.0),
                        float(info.get("bone_error") or 0.0),
                        float(info.get("bone_error_fallback") or 0.0)))
        if info.get("repaired"):
            parts.append("bone-length repair of %d segments" % int(info["repaired"]))
        if info.get("rejected"):
            parts.append("this fit was judged untrustworthy")
    parts.extend(recon.get("notes") or [])
    return "; ".join(parts)


def report_summary(report: dict) -> str:
    parts = [report.get("message", "")]
    if report.get("ik_direct"):
        parts.append("IK controls written directly (%s)" % ", ".join(report["ik_direct"]))
    if report.get("ik_snapped"):
        parts.append("IK controls aligned (%s)" % ", ".join(report["ik_snapped"]))
    if report.get("fk_switched"):
        parts.append("switched %s to FK" % ", ".join(report["fk_switched"]))
    if report.get("visibility"):
        parts.append("synced the display of %d collections" % len(report["visibility"]))
    if report.get("root_offset"):
        parts.append("root offset %s" % (report["root_offset"],))
    if report.get("keyframes"):
        parts.append("inserted %d keyframes" % report["keyframes"])
    warnings = report.get("warnings") or []
    if warnings:
        parts.append("Notes: " + "; ".join(warnings[:3]))
    skipped = report.get("skipped") or []
    if skipped:
        parts.append("skipped %d items (e.g. %s)" % (len(skipped), skipped[0][0]))
    return "; ".join(part for part in parts if part)


# --------------------------------------------------------------------------- modal background task base class
class PC_ModalWorker(bpy.types.Operator):
    bl_options = {"REGISTER", "INTERNAL"}

    _task_queue = None
    _thread = None
    _timer = None
    _cancel_flag = False
    _progress = None

    # -------------------------------------------------- implemented by subclasses
    def worker(self, emit, cancel):
        raise NotImplementedError

    def on_done(self, context, result):
        return {"FINISHED"}

    def on_error(self, context, message):
        self.report({"ERROR"}, message)
        return {"CANCELLED"}

    # -------------------------------------------------- internals
    def _push(self, kind, payload=None):
        # Workers call emit(("log", text)) with ONE tuple argument. The old signature treated that tuple as
        # `kind`, so every log / status / progress message was silently dropped. Accept both call styles.
        if payload is None and isinstance(kind, (tuple, list)) and len(kind) == 2:
            kind, payload = kind
        if self._task_queue is not None:
            self._task_queue.put((kind, payload))

    def _thread_main(self):
        try:
            result = self.worker(self._push, lambda: self._cancel_flag)
            self._push("done", result)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._push("error", "%s: %s" % (type(exc).__name__, exc))

    def start_worker(self, context):
        self._task_queue = queue.Queue()
        self._cancel_flag = False
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def log_line(self, context, text: str) -> None:
        text = str(text)
        print(utils.LOG_PREFIX, text)
        settings = getattr(context.scene, "pose_capture", None)
        if settings is not None:
            settings.log = ((settings.log or "") + "\n" + text).strip()[-4000:]

    def status_text(self, context, text: str) -> None:
        settings = getattr(context.scene, "pose_capture", None)
        if settings is not None:
            settings.status = text
        if context.workspace is not None:
            context.workspace.status_text_set(text)

    def update_progress(self, context, done: int, total: int) -> None:
        window_manager = context.window_manager
        total = max(int(total), 1)
        if self._progress is None or self._progress[1] != total:
            window_manager.progress_begin(0, total)
            self._progress = [0, total]
        window_manager.progress_update(int(min(done, total)))

    def _cleanup(self, context) -> None:
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._progress is not None:
            context.window_manager.progress_end()
            self._progress = None
        if context.workspace is not None:
            context.workspace.status_text_set(None)
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        settings = getattr(context.scene, "pose_capture", None)
        if settings is not None:
            settings.status = ""

    def modal(self, context, event):
        if event.type == "ESC":
            self._cancel_flag = True
            self.status_text(context, "Cancelling...")
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        finished = result = error = None
        while True:
            try:
                kind, payload = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log_line(context, payload)
            elif kind == "status":
                self.status_text(context, payload)
            elif kind == "progress":
                self.update_progress(context, payload[0], payload[1])
            elif kind == "done":
                finished, result = True, payload
            elif kind == "error":
                finished, error = True, payload

        if finished:
            self._cleanup(context)
            if error is not None:
                return self.on_error(context, error)
            return self.on_done(context, result)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        # Called by Blender when the modal operator is aborted from outside (window closed / new file loaded).
        self._cancel_flag = True
        try:
            self._cleanup(context)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- install dependencies
class PC_OT_install_deps(PC_ModalWorker):
    bl_idname = "pose_capture.install_deps"
    bl_label = "Install / Update Inference Dependencies"

    def execute(self, context):
        prefs = props.get_prefs(context)
        self._package = prefs.backend_package
        self.report({"INFO"}, "Started installing %s (may take a few minutes)" % self._package)
        return self.start_worker(context)

    def worker(self, emit, cancel):
        emit(("status", "Installing %s..." % self._package))
        ok = deps.pip_install(self._package, log_cb=lambda line: emit(("log", line)), cancel_cb=cancel)
        return {"ok": ok, "package": self._package}

    def on_done(self, context, result):
        status = deps.backend_status()
        if result.get("ok"):
            self.report({"INFO"}, "Dependencies installed: onnxruntime=%s, opencv=%s"
                        % (status["onnxruntime"], status["opencv"]))
        else:
            self.report({"ERROR"}, "Installation failed, see the log (maybe a network problem or no matching wheel)")
        return {"FINISHED"}


# --------------------------------------------------------------------------- model download / import
class PC_OT_download_models(PC_ModalWorker):
    bl_idname = "pose_capture.download_models"
    bl_label = "Download DWPose Models"

    def execute(self, context):
        prefs = props.get_prefs(context)
        det, pose = deps.model_paths()
        variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
        self._variant = variant
        self._targets = [(dwpose.DETECTOR_URLS, det, dwpose.DETECTOR_FILE),
                         (dwpose.POSE_URLS, pose, dwpose.POSE_FILE)]
        self._depth_target = (depth.model_urls(variant), deps.depth_model_path(variant),
                              depth.model_filename(variant))
        self.report({"INFO"}, "Started downloading models (DWPose about 340 MB + depth model)")
        return self.start_worker(context)

    def worker(self, emit, cancel):
        total_bytes = (deps.DETECTOR_SIZE_MB + deps.POSE_SIZE_MB
                       + int(depth.variant_info(self._variant)["size_mb"])) * 1024 * 1024
        offset = 0
        for urls, dest, name in self._targets:
            if cancel():
                return {"ok": False, "message": "Cancelled"}
            if os.path.isfile(dest):
                emit(("log", "Already exists, skipped: %s" % name))
                offset += os.path.getsize(dest)
                continue
            emit(("status", "Downloading %s..." % name))

            def progress(done, _total, _offset=offset):
                emit(("progress", (min(_offset + done, total_bytes), total_bytes)))

            ok = deps.download_model(urls, dest,
                                     progress_cb=progress,
                                     log_cb=lambda line: emit(("log", line)),
                                     cancel_cb=cancel)
            if not ok:
                return {"ok": False, "message": "Download failed: %s" % name}
            offset += os.path.getsize(dest)

        # Depth model: a failed download is not fatal (the bone-length solve is still available as a fallback)
        urls, dest, name = self._depth_target
        depth_ok = False
        if cancel():
            emit(("log", "Depth model download cancelled"))
        elif os.path.isfile(dest):
            emit(("log", "Already exists, skipped: %s" % name))
            depth_ok = True
        else:
            emit(("status", "Downloading %s..." % name))

            def depth_progress(done, _total, _offset=offset):
                emit(("progress", (min(_offset + done, total_bytes), total_bytes)))

            depth_ok = deps.download_model(urls, dest, progress_cb=depth_progress,
                                           log_cb=lambda line: emit(("log", line)),
                                           cancel_cb=cancel)
            if not depth_ok:
                emit(("log", "Depth model download failed: use 'Import Depth Model' instead, "
                             "or switch the variant to int8 and try again"))
        return {"ok": True, "depth_ok": depth_ok}

    def on_done(self, context, result):
        if not result.get("ok"):
            self.report({"ERROR"}, result.get("message", "Download failed"))
            return {"FINISHED"}
        if result.get("depth_ok"):
            self.report({"INFO"}, "Models are ready: %s" % utils.models_dir())
        else:
            self.report({"WARNING"},
                        "DWPose models are ready; the depth model was not downloaded (you can use the bone-length solve for now)")
        return {"FINISHED"}


class PC_OT_import_local_models(bpy.types.Operator):
    bl_idname = "pose_capture.import_local_models"
    bl_label = "Import Models from Local Folder"

    def execute(self, context):
        prefs = props.get_prefs(context)
        root = bpy.path.abspath(prefs.local_dwpose_root)
        found = deps.find_local_models(root)
        if not found:
            self.report({"ERROR"}, "Not found in %s: %s / %s"
                        % (root or "(empty)", dwpose.DETECTOR_FILE, dwpose.POSE_FILE))
            return {"CANCELLED"}
        det, _pose = deps.copy_into_models_dir(found, log_cb=utils.log)
        # If the same folder has depth*.onnx, import it as well (most ComfyUI users have one)
        depth_source = deps.find_local_depth_model(root)
        if depth_source:
            deps.copy_depth_into_models_dir(
                depth_source, getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT),
                log_cb=utils.log)
        self.report({"INFO"}, "Models imported: %s%s"
                    % (os.path.dirname(det), " (including the depth model)" if depth_source else ""))
        return {"FINISHED"}


class PC_OT_download_depth_model(PC_ModalWorker):
    bl_idname = "pose_capture.download_depth_model"
    bl_label = "Download Depth Model"

    def execute(self, context):
        prefs = props.get_prefs(context)
        self._variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
        self._dest = deps.depth_model_path(self._variant)
        self._urls = depth.model_urls(self._variant)
        size = int(depth.variant_info(self._variant)["size_mb"])
        self.report({"INFO"}, "Started downloading depth model %s (about %d MB)"
                    % (depth.model_filename(self._variant), size))
        return self.start_worker(context)

    def worker(self, emit, cancel):
        if os.path.isfile(self._dest):
            emit(("log", "Already exists, skipped: %s" % self._dest))
            return {"ok": True}
        total = int(depth.variant_info(self._variant)["size_mb"]) * 1024 * 1024
        emit(("status", "Downloading depth model..."))
        ok = deps.download_model(
            self._urls, self._dest,
            progress_cb=lambda done, _total: emit(("progress", (done, total))),
            log_cb=lambda line: emit(("log", line)),
            cancel_cb=cancel)
        return {"ok": ok}

    def on_done(self, context, result):
        if result.get("ok"):
            self.report({"INFO"}, "Depth model is ready: %s" % self._dest)
        else:
            self.report({"ERROR"}, "Depth model download failed (use 'Import Depth Model' instead, "
                                   "or retry with the int8 variant)")
        return {"FINISHED"}


class PC_OT_import_depth_model(bpy.types.Operator):
    """Pick an existing depth model onnx (Depth Anything / MiDaS etc.) and install it into the add-on folder."""

    bl_idname = "pose_capture.import_depth_model"
    bl_label = "Import Depth Model"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.onnx", options={"HIDDEN"})

    def execute(self, context):
        prefs = props.get_prefs(context)
        source = bpy.path.abspath(self.filepath)
        if not os.path.isfile(source):
            self.report({"ERROR"}, "File not found: %s" % source)
            return {"CANCELLED"}
        variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
        target = deps.copy_depth_into_models_dir(source, variant, log_cb=utils.log)
        prefs.local_depth_model = target if os.path.isfile(target) else source
        self.report({"INFO"}, "Depth model is ready: %s" % target)
        return {"FINISHED"}

    def invoke(self, context, event):
        if not self.filepath:
            prefs = props.get_prefs(context)
            self.filepath = getattr(prefs, "local_depth_model", "") or ""
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class PC_OT_show_depth(bpy.types.Operator):
    """Show the false-colour depth map in the Image Editor, to check whether the depth direction is right."""

    bl_idname = "pose_capture.show_depth"
    bl_label = "Show Depth Map"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = props.get_settings(context)
        image = (bpy.data.images.get(settings.depth_preview_name)
                 if settings.depth_preview_name else None)
        if image is None:
            self.report({"ERROR"}, "No depth map yet: set 'Depth Source' to the depth model first and detect the pose")
            return {"CANCELLED"}
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.spaces.active.image = image
                    area.tag_redraw()
                    self.report({"INFO"}, "Depth map shown in the Image Editor")
                    return {"FINISHED"}
        self.report({"INFO"}, "Depth map already exists: %s (select it in the Image Editor)" % image.name)
        return {"FINISHED"}


# --------------------------------------------------------------------------- depth preview window
def _panel_stats(values, canvas) -> dict:
    """Numeric statistics of one image (used to tell whether it is really all black / constant)."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    stats = {"min": -1.0, "max": -1.0, "mean": -1.0, "std": -1.0,
             "percentiles": [], "mean255": None}
    if finite.size:
        stats["min"] = float(finite.min())
        stats["max"] = float(finite.max())
        stats["mean"] = float(finite.mean())
        stats["std"] = float(finite.std())
        stats["percentiles"] = [float(v) for v in np.percentile(finite, [1, 2, 50, 98, 99])]
    if canvas is not None:
        stats["mean255"] = float(np.asarray(canvas, dtype=np.float64).mean())
    return stats


def _normalize_gray(values, near_is_large: bool = True, percentiles: bool = True) -> np.ndarray:
    """Depth -> grayscale: **whiter = farther** (matches "far = red" of the false colour, so the two images can be cross-checked).

    By default stretched over the 2%~98% percentiles, so a few outliers do not crush the whole image to black / white.
    """
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros(array.shape, dtype=np.uint8)
    clean = np.where(np.isfinite(array), array, float(np.median(finite)))
    if percentiles:
        low, high = np.percentile(clean, [2.0, 98.0])
    else:
        low, high = float(clean.min()), float(clean.max())
    span = max(float(high) - float(low), 1e-6)
    norm = np.clip((clean - low) / span, 0.0, 1.0)      # 1 = large value
    if near_is_large:
        norm = 1.0 - norm                               # white = small value = far
    return (norm * 255.0 + 0.5).astype(np.uint8)


def _save_canvas_png(path: str, canvas: np.ndarray) -> None:
    """Save an (H,W,3) uint8 canvas as a PNG (the temporary image datablock is deleted after writing)."""
    height, width = canvas.shape[:2]
    image = bpy.data.images.new("PC_TMP_VIEW", width=width, height=height, alpha=False)
    try:
        if image.colorspace_settings.name != "sRGB":
            image.colorspace_settings.name = "sRGB"
        rgba = np.empty((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = canvas.astype(np.float32) / 255.0
        rgba[:, :, 3] = 1.0
        image.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).reshape(-1))
        image.update()
        image.filepath_raw = path
        image.file_format = "PNG"
        image.save()
    finally:
        try:
            bpy.data.images.remove(image)
        except Exception:  # noqa: BLE001
            pass


def _open_folder(directory: str) -> None:
    try:
        if hasattr(os, "startfile"):
            os.startfile(directory)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", directory])
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", directory])
    except Exception as exc:  # noqa: BLE001
        utils.warn("Cannot open folder %s: %s" % (directory, exc))


def build_depth_view(settings, out_dir: str, keypoints=None, scores=None) -> dict:
    """Write the original image / depth grayscale / depth false colour as PNGs + stats.json into the preview folder and return the manifest."""
    depth_map = _RESULT.get("depth_map")
    rgb = _RESULT.get("rgb")
    if depth_map is None or rgb is None:
        raise ValueError("No depth map yet: set 'Depth Source' to the depth model first and click 'Detect Pose'")

    depth_info = (_RESULT.get("recon") or {}).get("depth") or {}
    direction = depth_direction(depth_map, _RESULT.get("box"), _RESULT.get("image_size"),
                                _RESULT.get("recon"))
    near_is_large = direction["near_is_large"]
    thr = float(settings.score_thr)

    photo = rgb
    if keypoints is not None and scores is not None:
        photo = imops.draw_pose_preview(rgb, keypoints, scores, thr)
    gray = _normalize_gray(depth_map, near_is_large=near_is_large)
    gray_3 = np.repeat(gray[:, :, None], 3, axis=2)   # imops.resize only takes 2D/3D images
    gray_rgb = imops.resize(gray_3, rgb.shape[1], rgb.shape[0])
    color = depth.preview_image(rgb, depth_map, near_is_large)

    os.makedirs(out_dir, exist_ok=True)
    items = (
        ("photo", "Input photo (with detected skeleton)", photo, photo,
         "First confirm that the detected person is the right one"),
        ("depth_gray", "Depth grayscale (whiter = farther)", gray_rgb, depth_map,
         "2%~98% stretch; white = far, black = near, matches the false colour's 'far = red'"),
        ("depth_color", "Depth false colour (near=blue far=red)", color, depth_map,
         "Compare with the photo: near parts such as face / hands should be blue and the far background red"),
    )
    panels = []
    for name, label, canvas, values, caption in items:
        path = os.path.join(out_dir, "%s.png" % name)
        _save_canvas_png(path, canvas)
        panels.append({"label": label, "caption": caption, "path": path,
                       "stats": _panel_stats(values, canvas)})

    notes = []
    finite = np.asarray(depth_map, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        span = float(finite.max()) - float(finite.min())
        if span < 1e-3:
            notes.append("The depth map is almost constant (span=%.6f): the model output is degenerate, check that the model file is correct"
                         % span)
    if float(gray.mean()) < 2.0:
        notes.append("The grayscale image is almost all black: the mapping range is wrong")
    if float(np.asarray(color, dtype=np.float64).mean()) < 2.0:
        notes.append("The false-colour image is almost all black: the depth value distribution is abnormal")
    if direction["warning"]:
        notes.append(direction["warning"])

    manifest = {
        "title": "Sanku AI Pose Studio depth preview (%s)"
                 % os.path.basename(_RESULT.get("path") or "untitled"),
        "source": _RESULT.get("path") or "",
        "image_size": list(_RESULT.get("image_size") or []),
        "depth_size": list(_RESULT.get("depth_size") or []),
        "near_is_large": near_is_large,
        "direction": direction,
        "depth_info": {key: value for key, value in depth_info.items()
                       if isinstance(value, (int, float, str, bool))},
        "panels": panels,
        "notes": notes,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
    return manifest


class PC_OT_open_depth_view(bpy.types.Operator):
    """Open a separate Tk preview window: original / depth grayscale / depth false colour + numeric statistics.

    It is independent of Blender's Image Editor, so "is the all-black preview a data problem or a display problem" can be told at a glance.
    """

    bl_idname = "pose_capture.open_depth_view"
    bl_label = "Depth Preview Window"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = props.get_settings(context)
        if _RESULT.get("depth_map") is None:
            self.report({"ERROR"}, "No depth map yet: set 'Depth Source' to the depth model first and click 'Detect Pose'")
            return {"CANCELLED"}

        out_dir = os.path.join(utils.user_data_dir("depth_view"), "latest")
        try:
            manifest = build_depth_view(settings, out_dir, _RESULT.get("keypoints"),
                                        _RESULT.get("scores"))
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Failed to create the preview images: %s" % exc)
            return {"CANCELLED"}
        utils.log("Depth preview manifest: %s (%s)" % (manifest.get("title"), out_dir))

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_view.py")
        prefs = props.get_prefs(context)
        python = depth_view.find_python(getattr(prefs, "depth_view_python", "") or "")
        if python is None:
            _open_folder(out_dir)
            self.report({"WARNING"},
                        "No Python with tkinter was found (Blender's bundled one has no tkinter). "
                        "The PNGs were saved to %s; you can set an interpreter by hand in the add-on preferences" % out_dir)
            return {"CANCELLED"}

        creation = 0
        for flag in ("DETACHED_PROCESS", "CREATE_NO_WINDOW"):
            creation |= getattr(subprocess, flag, 0)
        try:
            subprocess.Popen([python, script, "--dir", out_dir],
                             creationflags=creation, close_fds=True)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Failed to start the preview window: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Depth preview window opened: %s" % python)
        return {"FINISHED"}


# --------------------------------------------------------------------------- pose detection
class PC_OT_detect_pose(PC_ModalWorker):
    bl_idname = "pose_capture.detect"
    bl_label = "Detect Pose from Image"

    def execute(self, context):
        settings = props.get_settings(context)
        prefs = props.get_prefs(context)

        detector, pose_path, ready = props.resolve_model_paths()
        if not ready:
            self.report({"ERROR"}, "Model files missing (%s / %s), please download or import them in the add-on preferences first"
                        % (dwpose.DETECTOR_FILE, dwpose.POSE_FILE))
            return {"CANCELLED"}

        try:
            rgb = load_source_image(settings)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Failed to read the image: %s" % exc)
            return {"CANCELLED"}

        self._rgb = rgb
        self._paths = (detector, pose_path)
        self._backend = prefs.backend
        self._providers = props.resolve_providers()
        self._det_thr = float(settings.det_thr)
        self._score_thr = float(settings.score_thr)
        self._person = int(settings.person_index)
        # Depth model: only counted when "Depth Source = depth model" and the model is ready
        self._depth_path = None
        if settings.depth_mode == "DEPTH":
            path, ready = props.resolve_depth_model_path()
            if ready:
                self._depth_path = path
            else:
                utils.warn("No depth model, using the bone-length solve as a fallback this time "
                           "(you can download or import a depth model in the add-on preferences)")
        return self.start_worker(context)

    def worker(self, emit, cancel):
        detector, pose_path = self._paths
        emit(("status", "Loading models..."))
        estimator = get_estimator(detector, pose_path, self._backend, self._providers)
        if cancel():
            return {"ok": False, "message": "Cancelled"}
        emit(("status", "Detecting... (about 1~5 s on CPU)"))
        result = estimator.run(self._rgb, det_thr=self._det_thr,
                               person_index=self._person)
        result["ok"] = result.get("keypoints") is not None
        result["depth_map"] = None
        result["depth_size"] = None

        if result["ok"] and self._depth_path:
            if cancel():
                return {"ok": False, "message": "Cancelled"}
            emit(("status", "Estimating depth... (about 1 s on CPU)"))
            try:
                depth_estimator = get_depth_estimator(self._depth_path, self._backend,
                                                      self._providers)
                depth_map, depth_size = depth_estimator.predict(self._rgb)
                result["depth_map"] = depth_map
                result["depth_size"] = depth_size
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                emit(("log", "Depth estimation failed (using the bone-length solve this time): %s" % exc))
        return result

    def on_done(self, context, result):
        settings = props.get_settings(context)
        if not result.get("ok"):
            settings.detected = False
            self.report({"WARNING"}, "No person was detected, try another image or adjust the thresholds")
            return {"FINISHED"}

        _RESULT.update({
            "keypoints": result["keypoints"],
            "scores": result["scores"],
            "boxes": result["boxes"],
            "box": result.get("box"),
            "image_size": (self._rgb.shape[1], self._rgb.shape[0]),
            "rgb": self._rgb,
            "path": settings.image_filepath,
            "person_index": result.get("person_index", 0),
            "depth_map": result.get("depth_map"),
            "depth_size": result.get("depth_size"),
        })
        settings.detected = True
        settings.person_index = int(result.get("person_index", 0))
        info = "Detected %d people, using person #%d" % (len(result["boxes"]),
                                              result.get("person_index", 0) + 1)
        settings.detect_info = info
        utils.log(info, "box =", result.get("box"))

        try:
            settings.preview_name = update_preview_image(
                "PoseCapture_Preview", self._rgb, result["keypoints"],
                result["scores"], self._score_thr)
        except Exception as exc:  # noqa: BLE001
            utils.warn("Failed to create the preview image: %s" % exc)

        depth_map = result.get("depth_map")
        if depth_map is not None:
            direction = depth_direction(depth_map, result.get("box"),
                                        _RESULT["image_size"])
            near_is_large = direction["near_is_large"]
            settings.depth_info = "Depth map %dx%d (%s; false colour: near=blue far=red)" % (
                depth_map.shape[1], depth_map.shape[0],
                "large=near" if near_is_large else "large=far")
            settings.depth_direction = direction["text"]
            try:
                settings.depth_preview_name = update_depth_preview_image(
                    "PoseCapture_Depth", self._rgb, depth_map, near_is_large,
                    result["keypoints"], result["scores"], self._score_thr)
            except Exception as exc:  # noqa: BLE001
                utils.warn("Failed to create the depth preview: %s" % exc)
        else:
            settings.depth_info = ""
            settings.depth_direction = ""
            settings.depth_preview_name = ""

        self.report({"INFO"}, info)
        return {"FINISHED"}


class PC_OT_show_preview(bpy.types.Operator):
    """Show the preview image in the Image Editor."""
    bl_idname = "pose_capture.show_preview"
    bl_label = "Show Detection Preview"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = props.get_settings(context)
        image = bpy.data.images.get(settings.preview_name) if settings.preview_name else None
        if image is None:
            self.report({"ERROR"}, "No preview image yet, please detect the pose first")
            return {"CANCELLED"}
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.spaces.active.image = image
                    area.tag_redraw()
                    self.report({"INFO"}, "Preview shown in the Image Editor")
                    return {"FINISHED"}
        self.report({"INFO"}, "Preview image already exists: %s (select it in the Image Editor)" % image.name)
        return {"FINISHED"}


# --------------------------------------------------------------------------- armature related
def _resolve_armature(settings, context):
    """Prefer the armature specified in the panel, otherwise use the active object (if it is an armature)."""
    arm_obj = settings.armature
    if arm_obj is None and context.object is not None and context.object.type == "ARMATURE":
        arm_obj = context.object
    return arm_obj


class PC_OT_use_active_armature(bpy.types.Operator):
    bl_idname = "pose_capture.use_active_armature"
    bl_label = "Use Selected Armature"

    def execute(self, context):
        settings = props.get_settings(context)
        obj = context.object
        if obj is None or obj.type != "ARMATURE":
            self.report({"ERROR"}, "Please select an armature object first")
            return {"CANCELLED"}
        settings.armature = obj
        bmap = rigify_map.auto_detect(obj, settings.fingers, settings.limb_mode)
        props.sync_mapping_to_settings(settings, bmap)
        self.report({"INFO"}, "Bound %s, %d roles mapped (%d missing)"
                    % (obj.name, len(bmap.roles), len(bmap.missing)))
        return {"FINISHED"}


class PC_OT_auto_map(bpy.types.Operator):
    bl_idname = "pose_capture.auto_map"
    bl_label = "Auto-Match Bones"

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "Please specify the target armature first")
            return {"CANCELLED"}
        bmap = rigify_map.auto_detect(arm_obj, settings.fingers, settings.limb_mode)
        props.sync_mapping_to_settings(settings, bmap)
        utils.log("Bone mapping:", bmap.roles)
        utils.log("Not found:", bmap.missing)
        self.report({"INFO"}, "Matched %d roles, %d missing (you can set them by hand in the dropdowns)"
                    % (len(bmap.roles), len(bmap.missing)))
        return {"FINISHED"}


class PC_OT_add_metarig(bpy.types.Operator):
    bl_idname = "pose_capture.add_metarig"
    bl_label = "Add Rigify Human Metarig"

    def execute(self, context):
        settings = props.get_settings(context)
        try:
            bpy.ops.preferences.addon_enable(module="rigify")
        except Exception:  # noqa: BLE001
            pass
        try:
            bpy.ops.object.armature_human_metarig_add()
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Cannot add the metarig (is Rigify enabled?): %s" % exc)
            return {"CANCELLED"}

        obj = context.object
        settings.armature = obj
        bmap = rigify_map.auto_detect(obj, settings.fingers, settings.limb_mode)
        props.sync_mapping_to_settings(settings, bmap)
        self.report({"INFO"}, "Added %s" % obj.name)
        return {"FINISHED"}


class PC_OT_generate_rig(bpy.types.Operator):
    bl_idname = "pose_capture.generate_rig"
    bl_label = "Generate Rigify Rig"

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "Please specify the metarig first")
            return {"CANCELLED"}

        previous = context.view_layer.objects.active
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj
        try:
            if arm_obj.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
            bpy.ops.pose.rigify_generate()
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Generation failed: %s" % exc)
            if previous is not None:
                context.view_layer.objects.active = previous
            return {"CANCELLED"}

        generated = context.object
        if generated is not None and generated.type == "ARMATURE":
            settings.armature = generated
            bmap = rigify_map.auto_detect(generated, settings.fingers, settings.limb_mode)
            props.sync_mapping_to_settings(settings, bmap)
            self.report({"INFO"}, "Generated %s and updated the mapping" % generated.name)
        return {"FINISHED"}


class PC_OT_reset_pose(bpy.types.Operator):
    bl_idname = "pose_capture.reset_pose"
    bl_label = "Clear Written Pose"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from mathutils import Matrix

        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "Please specify the target armature first")
            return {"CANCELLED"}
        bmap = props.mapping_from_settings(settings)
        count = 0
        for name in set(bmap.used_bones()):
            pbone = arm_obj.pose.bones.get(name)
            if pbone is None:
                continue
            pbone.matrix_basis = Matrix.Identity(4)
            count += 1
        context.view_layer.update()
        self.report({"INFO"}, "Reset %d bones" % count)
        return {"FINISHED"}


class PC_OT_sync_limb_display(bpy.types.Operator):
    """Hide the group of controls that is "not in effect" according to the current IK/FK slider state.

    Click it after dragging the IK-FK slider on Rigify's panel yourself: slide to FK and the IK controls are hidden and FK shown,
    and vice versa - so you do not see controls that cannot be dragged and think the add-on / armature is broken.
    """

    bl_idname = "pose_capture.sync_limb_display"
    bl_label = "Sync Control Display"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "Please specify the target armature first")
            return {"CANCELLED"}
        specs = rigify_limbs.limb_specs(arm_obj)
        if not specs:
            self.report({"ERROR"}, "This is not a Rigify-generated armature: no IK/FK controls found")
            return {"CANCELLED"}
        touched = rigify_limbs.sync_visibility(arm_obj)
        modes = rigify_limbs.limb_modes(arm_obj, specs)
        detail = ", ".join("%s=%s" % (name, "FK" if value >= 0.5 else "IK")
                           for name, value in sorted(modes.items()))
        self.report({"INFO"}, "Synced %d bone collections (%s)" % (len(touched), detail or "none"))
        return {"FINISHED"}


# --------------------------------------------------------------------------- apply pose
class PC_OT_apply_pose(bpy.types.Operator):
    bl_idname = "pose_capture.apply_pose"
    bl_label = "Write Pose to Armature"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "Please specify the target armature first (pick it in the side panel)")
            return {"CANCELLED"}
        if _RESULT.get("keypoints") is None:
            self.report({"ERROR"}, "No detection result yet: click 'Detect Pose' first")
            return {"CANCELLED"}

        props.ensure_mapping_rows(settings)
        bmap = props.mapping_from_settings(settings)
        stale = [name for name in bmap.used_bones() if name not in arm_obj.data.bones]
        # Also re-match when the limb bones do not match the "Limb Write Mode" (IK selected but FK bones mapped):
        # otherwise a pose written to FK bones is invisible in IK mode.
        switched = rigify_limbs.mapped_limb_mode(arm_obj, bmap) not in ("", settings.limb_mode)
        remap = (not bmap.roles or (stale and settings.auto_map_on_apply)
                 or (switched and settings.auto_map_on_apply))
        if remap:
            bmap = rigify_map.auto_detect(arm_obj, settings.fingers, settings.limb_mode)
            props.sync_mapping_to_settings(settings, bmap)
            utils.log("Bone names were matched automatically again")

        metrics = rigify_map.read_metrics(arm_obj, bmap)
        if metrics.get("errors"):
            self.report({"ERROR"}, "Incomplete armature: %s" % "; ".join(metrics["errors"]))
            return {"CANCELLED"}

        options = props.options_from_settings(settings)
        # Depth direction: on "standing frontal" photos, where the body is flat in depth anyway, the bone-length fit cannot tell front from back (in testing the two directions
        # differ in cost by only 0.1%), so the physical criterion (the person is closer than the background, pixels only) pins the sign - otherwise the whole pose
        # comes out reversed front-to-back, which shows up as "the hand is in front in the depth map, but behind the body after applying".
        # If "Flip Depth Direction" is ticked the user's explicit choice wins (manual flip has the highest priority and no physical criterion is used).
        if _RESULT.get("depth_map") is not None and not options.get("depth_flip"):
            options = dict(options, depth_near_large=depth_orientation(
                _RESULT["depth_map"], _RESULT.get("box"), _RESULT.get("image_size")))
        recon = reconstruction.reconstruct(_RESULT["keypoints"], _RESULT["scores"],
                                          metrics, _RESULT["image_size"], options,
                                          depth_map=_RESULT.get("depth_map"),
                                          depth_size=_RESULT.get("depth_size"))
        if not recon.get("ok"):
            self.report({"ERROR"}, "Pose reconstruction failed: %s" % recon.get("message", ""))
            return {"CANCELLED"}
        _RESULT["recon"] = recon

        solved_info = recon_summary(recon)
        utils.log(solved_info)
        for note in recon.get("notes") or []:
            utils.warn(note)

        # Depth direction check: the preview colours always follow the physical criterion (the person is closer than the background), and it is used to check the fit direction in turn
        direction = depth_direction(_RESULT.get("depth_map"), _RESULT.get("box"),
                                    _RESULT.get("image_size"), recon)
        settings.depth_info = solved_info
        settings.depth_direction = direction["text"]
        if direction["warning"]:
            utils.warn(direction["warning"])
            settings.depth_direction += "; ★ " + direction["warning"]

        # Redraw the depth preview with the physical criterion (no longer following the fit direction, so a front/back reversal becomes visible)
        if (_RESULT.get("recon") or {}).get("depth") and _RESULT.get("depth_map") is not None:
            try:
                settings.depth_preview_name = update_depth_preview_image(
                    "PoseCapture_Depth", _RESULT["rgb"], _RESULT["depth_map"],
                    direction["near_is_large"],
                    _RESULT.get("keypoints"), _RESULT.get("scores"),
                    float(settings.score_thr))
            except Exception as exc:  # noqa: BLE001
                utils.warn("Failed to update the depth preview: %s" % exc)

        target_options = dict(options)
        target_options["frame"] = context.scene.frame_current
        report = retarget.apply_pose(arm_obj, bmap, metrics, recon, target_options)
        summary = report_summary(report)
        settings.detect_info = summary
        utils.log(summary)

        if not report.get("ok"):
            self.report({"ERROR"}, summary or "No bone was written")
            return {"CANCELLED"}

        try:
            for obj in context.selected_objects:
                obj.select_set(False)
            arm_obj.select_set(True)
            context.view_layer.objects.active = arm_obj
            if arm_obj.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
        except Exception:  # noqa: BLE001
            pass

        self.report({"INFO"}, summary)
        return {"FINISHED"}


class PC_OT_export_json(bpy.types.Operator):
    bl_idname = "pose_capture.export_json"
    bl_label = "Export Keypoints JSON"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        if _RESULT.get("keypoints") is None:
            self.report({"ERROR"}, "No detection result yet")
            return {"CANCELLED"}
        keypoints = np.asarray(_RESULT["keypoints"], dtype=float)
        scores = np.asarray(_RESULT["scores"], dtype=float)
        payload = {
            "source": _RESULT.get("path") or "",
            "image_size": list(_RESULT.get("image_size") or []),
            "person_index": _RESULT.get("person_index", 0),
            "keypoint_names": coco.KEYPOINT_NAMES,
            "keypoints": [[round(float(v), 3) for v in point] for point in keypoints],
            "scores": [round(float(v), 4) for v in scores],
        }
        with open(self.filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        self.report({"INFO"}, "Exported: %s" % self.filepath)
        return {"FINISHED"}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = os.path.join(utils.user_data_dir(), "keypoints.json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


CLASSES = (
    PC_OT_install_deps,
    PC_OT_download_models,
    PC_OT_download_depth_model,
    PC_OT_import_local_models,
    PC_OT_import_depth_model,
    PC_OT_detect_pose,
    PC_OT_show_preview,
    PC_OT_show_depth,
    PC_OT_open_depth_view,
    PC_OT_use_active_armature,
    PC_OT_auto_map,
    PC_OT_add_metarig,
    PC_OT_generate_rig,
    PC_OT_reset_pose,
    PC_OT_sync_limb_display,
    PC_OT_apply_pose,
    PC_OT_export_json,
)





