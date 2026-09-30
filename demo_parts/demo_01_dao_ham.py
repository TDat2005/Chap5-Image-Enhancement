"""Phan 1: truc quan dao ham, gradient, Laplacian va anh nhieu."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import skimage as ski

from demo_common import OUT, K_LAP, savefig, show_gray, show_signed


def main():
    # 1. Dao ham tren profile cuong do 1D.
    x = np.arange(0, 120)
    profile = np.zeros_like(x, dtype=float)
    profile[40:80] = 0.9
    profile[80:] = 0.3
    deriv = np.gradient(profile)
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axes[0].plot(x, profile, linewidth=2)
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("1D intensity profile")
    axes[1].plot(x, deriv, linewidth=2)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_ylabel("Derivative")
    axes[1].set_xlabel("Pixel position")
    axes[1].set_title("Derivative peaks at intensity transitions")
    savefig("01_edge_profile.png")

    # 2. Gradient theo hai huong tren checkerboard.
    img = ski.img_as_float(ski.data.checkerboard())
    kx = np.array([[1.0, -1.0]])
    ky = kx.T
    gx = ndi.convolve(img, kx, mode="reflect")
    gy = ndi.convolve(img, ky, mode="reflect")
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    show_gray(axes[0, 0], img, "Original", 0, 1)
    show_signed(axes[0, 1], gx, "Gx (right positive)", 1)
    show_signed(axes[0, 2], gy, "Gy (down positive)", 1)
    show_gray(axes[1, 0], mag, "Magnitude", 0, np.sqrt(2))
    axes[1, 1].imshow(np.ma.masked_where(mag < 1e-12, theta), cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Direction (only where M > 0)")
    axes[1, 1].axis("off")
    show_gray(axes[1, 2], mag > 0.3, "Threshold M > 0.3", 0, 1)
    savefig("02_gradient_checkerboard.png")

    # 3. Laplacian.
    lap = ndi.convolve(img, K_LAP, mode="reflect")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    show_gray(axes[0], img, "Original")
    show_signed(axes[1], lap, "Laplacian (blue < 0 < red)")
    savefig("03_laplacian_checkerboard.png")

    # 4. Nhieu Gaussian va tac dung lam tron truoc khi tinh gradient.
    rng = np.random.default_rng(42)
    noisy = np.clip(img + rng.normal(0, 0.18, img.shape), 0, 1)
    mag_noisy = np.hypot(ndi.convolve(noisy, kx, mode="reflect"), ndi.convolve(noisy, ky, mode="reflect"))
    smoothed = ndi.gaussian_filter(noisy, sigma=1.5)
    mag_smoothed = np.hypot(ndi.convolve(smoothed, kx, mode="reflect"), ndi.convolve(smoothed, ky, mode="reflect"))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, image, title in zip(
        axes.ravel(),
        [img, noisy, mag_noisy, mag_smoothed],
        ["Clean image", "Noisy image", "Gradient of noisy", "Gaussian then gradient"],
    ):
        show_gray(ax, image, title, 0, 1 if "image" in title else np.sqrt(2))
    savefig("04_noise_smoothing.png")
    print("Phan dao ham: da tao 4 hinh trong", OUT)


if __name__ == "__main__":
    main()
