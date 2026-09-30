"""Phan 3: cac bo phat hien bien tu gradient den Canny va LoG."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import skimage as ski

from demo_common import OUT, K_LAP, Image, ImageFilter, savefig, show_gray, show_rgb, show_signed


def non_maximum_suppression(magnitude, direction):
    """NMS 4 huong; x tang sang phai, y tang xuong duoi."""
    angle = np.rad2deg(direction) % 180
    out = np.zeros_like(magnitude)
    center = magnitude[1:-1, 1:-1]
    angle = angle[1:-1, 1:-1]
    pairs = [
        (((angle < 22.5) | (angle >= 157.5)), magnitude[1:-1, 2:], magnitude[1:-1, :-2]),
        (((angle >= 22.5) & (angle < 67.5)), magnitude[2:, 2:], magnitude[:-2, :-2]),
        (((angle >= 67.5) & (angle < 112.5)), magnitude[2:, 1:-1], magnitude[:-2, 1:-1]),
        (((angle >= 112.5) & (angle < 157.5)), magnitude[2:, :-2], magnitude[:-2, 2:]),
    ]
    for sector, neighbor_a, neighbor_b in pairs:
        keep = sector & (center >= neighbor_a) & (center >= neighbor_b)
        out[1:-1, 1:-1][keep] = center[keep]
    return out


def zero_crossings(log_image, threshold_ratio=0.05):
    maximum = ndi.maximum_filter(log_image, size=3, mode="reflect")
    minimum = ndi.minimum_filter(log_image, size=3, mode="reflect")
    contrast = maximum - minimum
    threshold = threshold_ratio * np.max(np.abs(log_image))
    return (maximum > 0) & (minimum < 0) & (contrast > threshold)


def main():
    cam = ski.img_as_float(ski.data.camera())

    # 7. Gradient Sobel va loai bo cuc dai khong cuc bo (NMS).
    sobel_x = ndi.sobel(cam, axis=1, mode="reflect")
    sobel_y = ndi.sobel(cam, axis=0, mode="reflect")
    sobel_mag = np.hypot(sobel_x, sobel_y)
    sobel_mag /= sobel_mag.max() + 1e-12
    cam_small = ski.transform.resize(cam, (256, 256), anti_aliasing=True)
    sx = ndi.sobel(cam_small, axis=1, mode="reflect")
    sy = ndi.sobel(cam_small, axis=0, mode="reflect")
    smag = np.hypot(sx, sy)
    smag /= smag.max() + 1e-12
    nms = non_maximum_suppression(smag, np.arctan2(sy, sx))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    show_gray(axes[0], cam_small, "Original")
    show_gray(axes[1], smag, "Sobel magnitude", 0, 1)
    show_gray(axes[2], nms, "After NMS", 0, 1)
    savefig("07_nms.png")

    # 8. Thanh phan Sobel theo truc.
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 4
    ky = kx.T
    gx_sobel = ndi.correlate(cam, kx, mode="reflect")
    gy_sobel = ndi.correlate(cam, ky, mode="reflect")
    sobel_h = ski.filters.sobel_h(cam)
    sobel_v = ski.filters.sobel_v(cam)
    sobel = ski.filters.sobel(cam)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    show_gray(axes[0, 0], cam, "Original", 0, 1)
    show_signed(axes[0, 1], gx_sobel, "Gx = sobel_v", 1)
    show_signed(axes[1, 0], gy_sobel, "Gy = sobel_h", 1)
    show_gray(axes[1, 1], sobel, "skimage Sobel magnitude", 0, 1)
    savefig("08_sobel.png")

    # 9. So sanh cac bo phat hien bien co ban.
    roberts = ski.filters.roberts(cam)
    prewitt = ski.filters.prewitt(cam)
    scharr = ski.filters.scharr(cam)
    laplace = ndi.convolve(cam, K_LAP, mode="reflect")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    items = [(cam, "Original"), (roberts, "Roberts"), (prewitt, "Prewitt"),
             (sobel, "Sobel"), (scharr, "Scharr"), (np.abs(laplace), "|Laplace|")]
    for ax, (image, title) in zip(axes.ravel(), items):
        show_gray(ax, image / max(float(image.max()), 1e-12), title, 0, 1)
    savefig("09_edge_detectors.png")

    # 10. Anh huong cua sigma trong phat hien Canny.
    canny1 = ski.feature.canny(cam, sigma=1, low_threshold=0.1, high_threshold=0.2)
    canny3 = ski.feature.canny(cam, sigma=3, low_threshold=0.1, high_threshold=0.2)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    show_gray(axes[0], cam, "Original")
    show_gray(axes[1], canny1, "Canny σ=1")
    show_gray(axes[2], canny3, "Canny σ=3")
    savefig("10_canny.png")

    # 11-12. LoG da thang va xap xi DoG.
    coins = ski.img_as_float(ski.data.coins())
    sigmas = [1, 2, 4, 8]
    logs = [sigma * sigma * ndi.gaussian_laplace(coins, sigma=sigma) for sigma in sigmas]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    show_gray(axes[0], coins, "Original")
    log_limit = max(np.max(np.abs(item)) for item in logs)
    for ax, image, sigma in zip(axes[1:], logs, sigmas):
        show_signed(ax, image, f"σ² LoG, σ={sigma}", log_limit)
    savefig("11_log_multiscale.png")

    g1 = ndi.gaussian_filter(coins, sigma=1.0)
    g2 = ndi.gaussian_filter(coins, sigma=1.6)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    show_gray(axes[0], g1, "Gaussian σ=1.0")
    show_gray(axes[1], g2, "Gaussian σ=1.6")
    show_signed(axes[2], g1 - g2, "DoG = G1 - G2")
    savefig("12_dog.png")

    # 13. Marr-Hildreth zero crossing.
    zero_images = [zero_crossings(ndi.gaussian_laplace(coins, sigma=sigma), 0.05) for sigma in [1, 2, 4]]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    show_gray(axes[0], coins, "Original")
    for ax, image, sigma in zip(axes[1:], zero_images, [1, 2, 4]):
        show_gray(ax, image, f"Zero-crossing σ={sigma}")
    savefig("13_zero_crossing.png")

    # 14. Cac bo loc bien va lam net trong Pillow.
    astro = ski.img_as_float(ski.data.astronaut())
    pil = Image.fromarray(ski.util.img_as_ubyte(astro))
    pil_edges = np.asarray(pil.filter(ImageFilter.FIND_EDGES)) / 255.0
    pil_enh = np.asarray(pil.filter(ImageFilter.EDGE_ENHANCE)) / 255.0
    pil_sharp = np.asarray(pil.filter(ImageFilter.SHARPEN)) / 255.0
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, image, title in zip(axes, [astro, pil_edges, pil_enh, pil_sharp],
                                ["Original", "FIND_EDGES", "EDGE_ENHANCE", "SHARPEN"]):
        show_rgb(ax, image, title)
    savefig("14_pillow_filters.png")
    print("Phan phat hien bien: da tao 8 hinh trong", OUT)


if __name__ == "__main__":
    main()
