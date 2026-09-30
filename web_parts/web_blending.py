"""Tab Gaussian/Laplacian Pyramid va hoa tron anh."""

import numpy as np
import streamlit as st
from skimage import data, transform

from .web_common import clamp01, ensure_rgb, png_bytes, sample_image, uploaded_image


def gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    channel_axis = -1 if image.ndim == 3 else None
    pyramid = [np.asarray(image, dtype=float)]
    for _ in range(levels):
        if min(pyramid[-1].shape[:2]) < 4:
            break
        pyramid.append(transform.pyramid_reduce(
            pyramid[-1], downscale=2, sigma=1, mode="reflect",
            preserve_range=True, channel_axis=channel_axis))
    return pyramid


def expand_to(image: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    channel_axis = -1 if image.ndim == 3 else None
    enlarged = transform.pyramid_expand(
        image, upscale=2, sigma=1, mode="reflect", preserve_range=True,
        channel_axis=channel_axis)
    return transform.resize(enlarged, shape, order=1, mode="reflect",
                            preserve_range=True, anti_aliasing=False, clip=False)


def laplacian_pyramid(gaussian: list[np.ndarray]) -> list[np.ndarray]:
    residuals = [gaussian[index] - expand_to(gaussian[index + 1], gaussian[index].shape)
                 for index in range(len(gaussian) - 1)]
    return residuals + [gaussian[-1].copy()]


def reconstruct(laplacian: list[np.ndarray]) -> np.ndarray:
    current = laplacian[-1].copy()
    for index in range(len(laplacian) - 2, -1, -1):
        current = expand_to(current, laplacian[index].shape) + laplacian[index]
    return current


def prepare_blend_image(image: np.ndarray, size: int = 384) -> np.ndarray:
    image = ensure_rgb(image)
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    cropped = image[y0:y0 + side, x0:x0 + side]
    return transform.resize(cropped, (size, size, 3), anti_aliasing=True, preserve_range=True)


def blending_mask(size: int, position: float, softness: float) -> np.ndarray:
    x = np.arange(size, dtype=float)
    center = position * (size - 1)
    if softness <= 0.5:
        line = (x <= center).astype(float)
    else:
        line = 1.0 / (1.0 + np.exp((x - center) / softness))
    return np.repeat(line[None, :, None], size, axis=0).repeat(3, axis=2)


def render_tab() -> None:
    st.subheader("Hòa trộn ảnh nhiều mức")
    st.write("So sánh đường ghép trực tiếp với kết quả dùng Gaussian và Laplacian Pyramid.")
    controls, display = st.columns([0.32, 0.68], gap="large")
    with controls:
        upload_a = st.file_uploader("Ảnh A", type=["png", "jpg", "jpeg", "webp"], key="blend_a")
        upload_b = st.file_uploader("Ảnh B", type=["png", "jpg", "jpeg", "webp"], key="blend_b")
        levels = st.slider("Số mức pyramid", 2, 6, 5)
        position = st.slider("Vị trí đường ghép", 0.15, 0.85, 0.50, 0.01)
        softness = st.slider("Độ mềm của mặt nạ", 0.0, 30.0, 3.0, 0.5)

    image_a = prepare_blend_image(uploaded_image(upload_a, data.astronaut()))
    image_b = prepare_blend_image(uploaded_image(upload_b, data.coffee()))
    mask = blending_mask(image_a.shape[0], position, softness)
    gaussian_a = gaussian_pyramid(image_a, levels)
    gaussian_b = gaussian_pyramid(image_b, levels)
    gaussian_mask = gaussian_pyramid(mask, levels)
    laplacian_a = laplacian_pyramid(gaussian_a)
    laplacian_b = laplacian_pyramid(gaussian_b)
    mixed = [m * a + (1.0 - m) * b for m, a, b in zip(gaussian_mask, laplacian_a, laplacian_b)]
    blended = clamp01(reconstruct(mixed))
    direct = clamp01(mask * image_a + (1.0 - mask) * image_b)

    with display:
        row_a, row_b = st.columns(2, gap="large")
        with row_a:
            st.image(direct, caption="Ghép trực tiếp", width="stretch")
        with row_b:
            st.image(blended, caption="Pyramid Blending", width="stretch")
        preview = st.columns(3)
        preview[0].image(image_a, caption="Ảnh A", width="stretch")
        preview[1].image(mask, caption="Mặt nạ", width="stretch")
        preview[2].image(image_b, caption="Ảnh B", width="stretch")
        st.download_button("Tải ảnh hòa trộn PNG", png_bytes(blended), "chapter5_pyramid_blend.png", "image/png")
