"""Diem chay chinh: goi lan luot cac phan demo rieng cua Chuong 5."""

from pathlib import Path
import os
import subprocess
import sys

import matplotlib
import numpy
import PIL
import scipy
import skimage


BASE_DIR = Path(__file__).resolve().parent
PARTS = [
    ("Phan 1/4 - Dao ham, gradient, Laplacian va nhieu", "demo_parts/demo_01_dao_ham.py"),
    ("Phan 2/4 - Lam sac net", "demo_parts/demo_02_lam_sac_net.py"),
    ("Phan 3/4 - Phat hien bien", "demo_parts/demo_03_phat_hien_bien.py"),
    ("Phan 4/4 - Gaussian/Laplacian Pyramid va hoa tron", "demo_parts/demo_04_pyramid_blending.py"),
    ("Kiem tra so va tao metrics.json", "demo_parts/demo_05_kiem_tra.py"),
]


def main():
    print("Python:", sys.version.split()[0], flush=True)
    print("NumPy:", numpy.__version__, flush=True)
    print("SciPy:", scipy.__version__, flush=True)
    print("scikit-image:", skimage.__version__, flush=True)
    print("Matplotlib:", matplotlib.__version__, flush=True)
    print("Pillow:", PIL.__version__, flush=True)
    print("Ket qua se duoc luu tai:", BASE_DIR / "ch5_demo_outputs", flush=True)
    print(flush=True)

    environment = os.environ.copy()
    for title, script in PARTS:
        print("=" * 68, flush=True)
        print(title, flush=True)
        print("=" * 68, flush=True)
        completed = subprocess.run([sys.executable, str(BASE_DIR / script)], cwd=BASE_DIR, env=environment)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    print("\nHoan tat 18 hinh minh hoa va cac kiem tra so.", flush=True)


if __name__ == "__main__":
    main()
