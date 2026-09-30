"""Dung chung bo nap anh va ham ve cho cac phan demo Chuong 5."""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy import ndimage as ndi
from PIL import Image, ImageFilter


BASE_DIR = Path(__file__).resolve().parent.parent
_output_setting = os.environ.get("CH5_OUTPUT_DIR")
OUT = Path(_output_setting).expanduser() if _output_setting else BASE_DIR / "ch5_demo_outputs"
if not OUT.is_absolute():
    OUT = BASE_DIR / OUT
OUT.mkdir(parents=True, exist_ok=True)

K_LAP = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "font.family": "DejaVu Sans"})


def savefig(name: str) -> Path:
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def show_gray(ax, image, title, vmin=None, vmax=None):
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")


def show_rgb(ax, image, title):
    ax.imshow(np.clip(image, 0, 1))
    ax.set_title(title)
    ax.axis("off")


def show_signed(ax, image, title, limit=None):
    limit = float(np.max(np.abs(image))) if limit is None else limit
    limit = max(limit, 1e-12)
    ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(title)
    ax.axis("off")
