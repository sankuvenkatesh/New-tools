# Sanku AI Pose Studio - Blender 5.1 (English build, v1.3.4)

Image-to-Rigify posing add-on based on Pose Capture 1.3.3. Detects a person in a photo (DWPose),
estimates real depth for every joint (Depth Anything V2) and writes the resulting pose onto a Rigify armature.

**Read `HOW_IT_WORKS.txt` for a full explanation of how the add-on works, every setting, troubleshooting and the list of fixes.**

## Install
1. In Blender: Edit > Preferences > Add-ons > Install from Disk (the dropdown arrow in the top-right of the Add-ons tab).
2. Select this ZIP file (do not extract it first).
3. Enable **Sanku AI Pose Studio**.
4. Open 3D View > Sidebar (N) > Sanku AI Pose Studio.

## Workflow
1. Click Install Dependencies and Download Models (needs an internet connection, one time only).
2. Load a reference image and click Detect Pose.
3. Select your Rigify armature (a generated rig is recommended), click Auto-Match Bones.
4. Review the solving settings, then Apply to Armature. Optionally enable Insert Keyframe first.

## Compatibility note
Targets Blender 4.2 - 5.x APIs. The original project reports testing on Blender 5.2.1. This English build was
syntax-checked, structurally verified against the original source and smoke-tested with stubbed Blender modules,
but it has not been run inside a real Blender session in the environment where it was prepared.
Test it on a scene you can afford to lose before production use.

## Scope
This package customizes and packages the Pose Capture core. MediaToPose is not merged into this build: it is a
separate GPL-3.0-or-later project with a Blender 5.2 minimum and a larger provider/runtime architecture.

## Credits and license
Pose Capture source is MIT licensed. See `LICENSE`. Third-party model weights and dependencies are downloaded by the add-on and are not bundled.
