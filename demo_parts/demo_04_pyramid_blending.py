"""Phan 4: Gaussian Pyramid, Laplacian Pyramid va hoa tron nhieu muc."""

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from demo_common import OUT, savefig, show_gray, show_rgb


def gaussian_pyramid(image, levels=5):
    channel_axis = -1 if image.ndim == 3 else None
    pyramid = [np.asarray(image, dtype=float)]
    for _ in range(levels):
        if min(pyramid[-1].shape[:2]) < 2:
            break
        pyramid.append(ski.transform.pyramid_reduce(
            pyramid[-1], downscale=2, sigma=1, mode="reflect",
            preserve_range=True, channel_axis=channel_axis))
    return pyramid


def expand_to(image, shape):
    channel_axis = -1 if image.ndim == 3 else None
    enlarged = ski.transform.pyramid_expand(
        image, upscale=2, sigma=1, mode="reflect", preserve_range=True,
        channel_axis=channel_axis)
    return ski.transform.resize(
        enlarged, shape, order=1, mode="reflect", preserve_range=True,
        anti_aliasing=False, clip=False)


def laplacian_pyramid(gaussian):
    residuals = [gaussian[i] - expand_to(gaussian[i + 1], gaussian[i].shape)
                 for i in range(len(gaussian) - 1)]
    return residuals + [gaussian[-1].copy()]


def reconstruct(laplacian):
    current = laplacian[-1].copy()
    for index in range(len(laplacian) - 2, -1, -1):
        current = expand_to(current, laplacian[index].shape) + laplacian[index]
    return current


def main():
    # 15. Tao va hien thi Gaussian Pyramid.
    base = ski.img_as_float(ski.data.astronaut())
    base = ski.transform.resize(base, (512, 512), anti_aliasing=True)
    gaussian = gaussian_pyramid(base, 5)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    for index, (ax, image) in enumerate(zip(axes, gaussian)):
        show_rgb(ax, image, f"G{index}: {image.shape[1]}×{image.shape[0]}")
    savefig("15_gaussian_pyramid.png")

    # 16. Laplacian Pyramid giu cac thanh phan du co dau.
    laplacian = laplacian_pyramid(gaussian)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    for index, (ax, image) in enumerate(zip(axes, laplacian)):
        if index < len(laplacian) - 1:
            display = (image - image.min()) / (image.max() - image.min() + 1e-12)
            show_rgb(ax, display, f"L{index} (display)")
        else:
            show_rgb(ax, image, f"Base L{index}")
    savefig("16_laplacian_pyramid.png")

    # 17. Kiem tra tai tao anh tu Laplacian Pyramid.
    rebuilt = reconstruct(laplacian)
    max_error = float(np.max(np.abs(rebuilt - base)))
    print("Pyramid reconstruction max abs error:", max_error)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    show_rgb(axes[0], base, "Original")
    show_rgb(axes[1], rebuilt, "Reconstructed")
    show_gray(axes[2], np.mean(np.abs(rebuilt - base), axis=2), "Absolute error", 0, 1)
    axes[2].set_title(f"Abs error (max={max_error:.2e})")
    savefig("17_reconstruction.png")

    # 18. So sanh ghep cat truc tiep va Pyramid Blending.
    image_a = ski.transform.resize(ski.img_as_float(ski.data.astronaut()), (512, 512), anti_aliasing=True)
    image_b = ski.transform.resize(ski.img_as_float(ski.data.coffee()), (512, 512), anti_aliasing=True)
    mask = np.zeros((512, 512, 3), dtype=float)
    mask[:, :256, :] = 1.0
    gaussian_a = gaussian_pyramid(image_a, 5)
    gaussian_b = gaussian_pyramid(image_b, 5)
    gaussian_mask = gaussian_pyramid(mask, 5)
    laplacian_a = laplacian_pyramid(gaussian_a)
    laplacian_b = laplacian_pyramid(gaussian_b)
    mixed = [m * a + (1 - m) * b for m, a, b in zip(gaussian_mask, laplacian_a, laplacian_b)]
    blended = reconstruct(mixed)
    direct = mask * image_a + (1 - mask) * image_b
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for ax, image, title in zip(axes, [image_a, image_b, mask, direct, blended],
                                ["Image A", "Image B", "Binary mask", "Direct cut", "Pyramid blend"]):
        show_rgb(ax, image, title)
    savefig("18_pyramid_blending.png")
    print("Phan Pyramid: da tao 4 hinh trong", OUT)


if __name__ == "__main__":
    main()
