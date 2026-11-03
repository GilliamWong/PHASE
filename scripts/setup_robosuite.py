"""One-time robosuite setup; run after installing requirements.txt.

Writes robosuite's private macro file, and on Windows works around two
robosuite 1.5 issues:
- it loads mujoco.dll from its own package folder, so copy the DLL there from
  the installed mujoco package;
- with GPU rendering on, it forces MUJOCO_GL=egl, which only exists on Linux,
  so turn GPU rendering off (rendering then uses WGL/GLFW).

Usage: python scripts/setup_robosuite.py
"""

import importlib.util
import os
import platform
import shutil


def _package_dir(name: str) -> str:
    # find_spec locates the package without importing it; importing robosuite
    # fails on Windows until the DLL is in place.
    return os.path.dirname(importlib.util.find_spec(name).origin)


def main():
    robosuite_dir = _package_dir("robosuite")
    is_windows = platform.system() == "Windows"

    if is_windows:
        dll = os.path.join(_package_dir("mujoco"), "mujoco.dll")
        shutil.copy(dll, os.path.join(robosuite_dir, "utils"))
        print(f"Copied {dll} into robosuite/utils.")

    macros_path = os.path.join(robosuite_dir, "macros_private.py")
    with open(macros_path, "w", encoding="utf-8") as f:
        f.write("# Written by scripts/setup_robosuite.py.\n")
        f.write(f"MUJOCO_GPU_RENDERING = {not is_windows}\n")
    print(f"Wrote {macros_path}.")


if __name__ == "__main__":
    main()
