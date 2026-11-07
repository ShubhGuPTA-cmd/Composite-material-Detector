
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = [
    "numpy",
    "opencv-python",
    "matplotlib",
    "pillow",
    "torch",
    "torchvision",
    "streamlit",
    "reportlab",
    "pytest",
]

lines = [
    "# CompositeGuard AI - installed development versions",
    "# For PyTorch CPU builds, use:",
    "# pip install torch torchvision --index-url "
    "https://download.pytorch.org/whl/cpu",
]

for name in PACKAGES:
    try:
        installed_version = version(name).split("+")[0]
        lines.append(f"{name}=={installed_version}")
    except PackageNotFoundError:
        lines.append(f"# {name} is not installed")

Path("requirements.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))

