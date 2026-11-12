"""Dependency and model management: check/install onnxruntime, download the DWPose models.

pip installs and downloads run in a worker thread and report progress through callbacks
(see the modal worker in operators).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

from . import depth, dwpose, utils

# Selectable inference backend packages
BACKEND_PACKAGE_IDS = [
    "onnxruntime",
    "onnxruntime-directml",
    "onnxruntime-gpu",
    "opencv-python-headless",
]

DETECTOR_SIZE_MB = 207
POSE_SIZE_MB = 128


# --------------------------------------------------------------------------- status
_STATUS_CACHE: dict = {"time": 0.0, "value": None}
_STATUS_TTL = 3.0   # seconds - the panel redraws constantly, so do not re-import onnxruntime / cv2 on every redraw


def backend_status(refresh: bool = False) -> dict:
    cached = _STATUS_CACHE["value"]
    if cached is not None and not refresh and time.monotonic() - _STATUS_CACHE["time"] < _STATUS_TTL:
        return cached
    utils.ensure_libs_on_path()
    status = {"onnxruntime": None, "opencv": None, "providers": []}
    for key, module in (("onnxruntime", "onnxruntime"), ("opencv", "cv2")):
        try:
            mod = __import__(module)
            status[key] = getattr(mod, "__version__", "?")
            if key == "onnxruntime":
                status["providers"] = list(mod.get_available_providers())
        except Exception as exc:  # noqa: BLE001
            status[key] = None
            if key == "onnxruntime":
                status["providers_error"] = str(exc)
    _STATUS_CACHE["time"] = time.monotonic()
    _STATUS_CACHE["value"] = status
    return status


def backend_ready() -> bool:
    status = backend_status()
    return bool(status["onnxruntime"] or status["opencv"])


# --------------------------------------------------------------------------- pip
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _pip_available(interpreter: str) -> bool:
    try:
        proc = subprocess.run([interpreter, "-m", "pip", "--version"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=60, creationflags=_NO_WINDOW)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def ensure_pip(interpreter: str, log_cb=None) -> bool:
    """Blender's bundled Python usually ships WITHOUT pip - bootstrap it with ensurepip when it is missing."""
    if _pip_available(interpreter):
        return True
    if log_cb:
        log_cb("pip is not installed in Blender's Python, bootstrapping it with ensurepip...")
    for extra in ([], ["--user"]):      # second try: per-user install when the Blender folder is not writable
        try:
            proc = subprocess.run([interpreter, "-m", "ensurepip", "--upgrade"] + extra,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                                  encoding="utf-8", errors="replace", timeout=300, creationflags=_NO_WINDOW)
        except Exception as exc:  # noqa: BLE001
            if log_cb:
                log_cb("ensurepip failed to start: %s" % exc)
            continue
        if log_cb and proc.stdout:
            for line in proc.stdout.splitlines():
                if line.strip():
                    log_cb(line.rstrip())
        if proc.returncode == 0 and _pip_available(interpreter):
            return True
    return False


def pip_install(package: str, log_cb=None, cancel_cb=None) -> bool:
    """Install dependencies with Blender's bundled Python into the add-on data folder (Blender itself stays untouched)."""
    utils.ensure_libs_on_path()
    python = utils.python_executable()
    target = utils.libs_dir()

    interpreter = python or sys.executable
    if not ensure_pip(interpreter, log_cb):
        if log_cb:
            log_cb("pip is not available and could not be set up with ensurepip")
        return False
    cmd = [interpreter, "-m", "pip", "install", "--upgrade", "--no-cache-dir",
           "--disable-pip-version-check", "--no-warn-script-location",
           "--target", target, package]

    if log_cb:
        log_cb("$ " + " ".join('"%s"' % c if " " in c else c for c in cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:  # noqa: BLE001
        if log_cb:
            log_cb("Failed to start pip: %s" % exc)
        return False

    assert proc.stdout is not None
    stop = threading.Event()
    if cancel_cb:
        def _watch():          # lets ESC abort a stuck pip run (it used to be impossible to cancel)
            while not stop.wait(0.3):
                if cancel_cb():
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                    return
        threading.Thread(target=_watch, daemon=True).start()
    for line in proc.stdout:
        line = line.rstrip()
        if line and log_cb:
            log_cb(line)
    proc.wait()
    stop.set()
    cancelled = bool(cancel_cb and cancel_cb())
    if cancelled and log_cb:
        log_cb("Installation cancelled")

    if proc.returncode != 0 and log_cb:
        log_cb("pip exit code %d (possibly a network problem, or no matching wheel for this Python version)"
               % proc.returncode)

    import importlib
    importlib.invalidate_caches()
    _STATUS_CACHE["value"] = None      # force a fresh dependency check after installing
    return proc.returncode == 0 and not cancelled


# --------------------------------------------------------------------------- download
def _download_one(url: str, dest: str, progress_cb=None, cancel_cb=None) -> bool:
    tmp = dest + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "pose-capture-blender"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                if cancel_cb and cancel_cb():
                    handle.close()
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    return False
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)
        if total and done != total:
            raise IOError("Incomplete download: received %d of %d bytes" % (done, total))
    except BaseException:
        # never leave a half-written .part file behind, and never accept a truncated model file
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    if os.path.exists(dest):
        os.remove(dest)
    os.replace(tmp, dest)
    return True


def download_model(urls, dest: str, progress_cb=None, log_cb=None, cancel_cb=None) -> bool:
    for index, url in enumerate(urls):
        if log_cb:
            log_cb("Downloading %s" % url)
        try:
            if _download_one(url, dest, progress_cb, cancel_cb):
                return True
        except Exception as exc:  # noqa: BLE001
            if log_cb:
                log_cb("Failed: %s" % exc)
        if cancel_cb and cancel_cb():
            return False
        if index < len(urls) - 1 and log_cb:
            log_cb("Trying the mirror address...")
    return False


def model_paths() -> tuple[str, str]:
    folder = utils.models_dir()
    return (os.path.join(folder, dwpose.DETECTOR_FILE),
            os.path.join(folder, dwpose.POSE_FILE))


def models_present() -> bool:
    det, pose = model_paths()
    return os.path.isfile(det) and os.path.isfile(pose)


# --------------------------------------------------------------------------- depth model
def depth_model_path(variant: str = None) -> str:
    """Path of the depth model file (named per variant, so variants never overwrite each other)."""
    return os.path.join(utils.models_dir(),
                        depth.model_filename(variant or depth.DEFAULT_VARIANT))


def depth_model_present(variant: str = None) -> bool:
    return os.path.isfile(depth_model_path(variant))


def find_local_depth_model(root: str):
    """Recursively look for a depth model onnx in the given folder (re-uses an existing Depth Anything from ComfyUI)."""
    if not root or not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if depth.looks_like_model(name):
                return os.path.join(dirpath, name)
    return None


def copy_depth_into_models_dir(source: str, variant: str = None, log_cb=None) -> str:
    """Copy an external depth model into the add-on model folder; on failure reference the original path directly."""
    dest = depth_model_path(variant)
    if os.path.abspath(source) == os.path.abspath(dest) or not os.path.isfile(source):
        return dest if os.path.isfile(dest) else source
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        if os.path.exists(dest):
            os.remove(dest)
        shutil.copy2(source, dest)
        if log_cb:
            log_cb("Copied depth model: %s" % dest)
        return dest
    except Exception as exc:  # noqa: BLE001
        if log_cb:
            log_cb("Failed to copy the depth model (using the original file directly instead): %s" % exc)
        return source


def find_local_models(root: str):
    """Recursively look for the DWPose model files in the given folder (re-uses an existing ComfyUI / DWPose install)."""
    if not root or not os.path.isdir(root):
        return None
    det = pose = None
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            lower = name.lower()
            if lower == dwpose.DETECTOR_FILE:
                det = det or os.path.join(dirpath, name)
            elif lower == dwpose.POSE_FILE:
                pose = pose or os.path.join(dirpath, name)
        if det and pose:
            break
    if det and pose:
        return det, pose
    return None


def copy_into_models_dir(sources, log_cb=None) -> tuple[str, str]:
    """Copy external models into the add-on model folder; if copying fails, use the original paths directly."""
    det_src, pose_src = sources
    det_dst, pose_dst = model_paths()
    os.makedirs(os.path.dirname(det_dst), exist_ok=True)

    result = []
    for src, dst, kind in ((det_src, det_dst, "detector"), (pose_src, pose_dst, "pose")):
        if os.path.abspath(src) == os.path.abspath(dst) or not os.path.isfile(src):
            result.append(dst if os.path.isfile(dst) else src)
            continue
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)
            if log_cb:
                log_cb("Copied %s model: %s" % (kind, dst))
            result.append(dst)
        except Exception as exc:  # noqa: BLE001
            if log_cb:
                log_cb("Failed to copy %s model (using the original file directly instead): %s" % (kind, exc))
            result.append(src)
    return result[0], result[1]

