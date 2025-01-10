import sys
from cx_Freeze import setup, Executable
sys.setrecursionlimit(10000)

script = "Control.py"

build_options = {
    "packages": ["mediapipe","torch","pyautogui","cv2"],
    "include_files": ["hand_gesture.yaml","models"],
    "excludes":[],
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name = "HandGestureControl",
    version="1.0",
    description="Hand gesture control application",
    options={"build_exe":build_options},
    executables=[Executable(script,base=base)],
)