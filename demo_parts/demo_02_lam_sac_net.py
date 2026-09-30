"""Phan 2: lam sac net bang Laplacian va Unsharp Masking."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import skimage as ski

from demo_common import OUT, K_LAP, savefig, show_gray, show_signed


def main():
    cam = ski.img_as_float(ski.data.camera())

    # 5. Lam sac net bang Laplacian.
    lap_cam = ndi.convolve(cam, K_LAP, mode="reflect")
    sharp_lap = np.clip(cam - 0.7 * lap_cam, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    show_gray(axes[0], cam, "Original")
    show_signed(axes[1], lap_cam, "Laplacian (signed)")
    show_gray(axes[2], sharp_lap, "Sharpened")
    savefig("05_laplacian_sharpening.png")

    # 6. Unsharp Masking: lay chi tiet tu anh goc tru anh lam mo.
    blur = ndi.gaussian_filter(cam, sigma=3)
    detail = cam - blur
    alphas = [1, 3, 5]
    results = [np.clip(cam + amount * detail, 0, 1) for amount in alphas]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    show_gray(axes[0, 0], cam, "Original")
    show_gray(axes[0, 1], blur, "Blurred (σ=3)")
    show_signed(axes[0, 2], detail, "Detail = original - blurred")
    for ax, image, amount in zip(axes[1], results, alphas):
        show_gray(ax, image, f"Unsharp α={amount}")
    savefig("06_unsharp_masking.png")
    print("Phan lam sac net: da tao 2 hinh trong", OUT)


if __name__ == "__main__":
    main()
