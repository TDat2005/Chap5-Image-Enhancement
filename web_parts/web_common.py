"""Ham dung chung cho giao dien Streamlit Chuong 5."""

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
from skimage import color, data, img_as_float, transform


K_LAPLACE = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def clamp01(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=float), 0.0, 1.0)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = img_as_float(image)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = color.rgba2rgb(image)
    return clamp01(image[..., :3])


def ensure_gray(image: np.ndarray) -> np.ndarray:
    image = img_as_float(image)
    if image.ndim == 3:
        image = color.rgba2rgb(image) if image.shape[2] == 4 else image[..., :3]
        image = color.rgb2gray(image)
    return clamp01(image)


def resize_working(image: np.ndarray, max_side: int = 720) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    shape = (max(1, round(height * scale)), max(1, round(width * scale)))
    if image.ndim == 3:
        shape += (image.shape[2],)
    return transform.resize(image, shape, anti_aliasing=True, preserve_range=True)


def uploaded_image(uploaded, fallback: np.ndarray) -> np.ndarray:
    if uploaded is None:
        return img_as_float(fallback)
    with Image.open(uploaded) as opened:
        return img_as_float(np.asarray(opened.convert("RGB")))


def sample_image(name: str) -> np.ndarray:
    samples = {
        "Ảnh người chụp": data.camera(),
        "Phi hành gia": data.astronaut(),
        "Đồng xu": data.coins(),
        "Tách cà phê": data.coffee(),
    }
    return img_as_float(samples[name])


def png_bytes(image: np.ndarray) -> bytes:
    image = clamp01(image)
    array = (image * 255.0 + 0.5).astype(np.uint8)
    if array.ndim == 2:
        pil_image = Image.fromarray(array, mode="L")
    else:
        pil_image = Image.fromarray(array[..., :3], mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_pair(original: np.ndarray, result: np.ndarray, result_title: str) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.image(original, caption="Ảnh đầu vào", width="stretch", clamp=True)
    with right:
        st.image(result, caption=result_title, width="stretch", clamp=True)
