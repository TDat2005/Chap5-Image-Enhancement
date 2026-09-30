"""Tab lam sac net: Laplacian sharpening va Unsharp Masking."""

import numpy as np
import streamlit as st
from scipy import ndimage as ndi

from .web_common import K_LAPLACE, clamp01, ensure_rgb, png_bytes, render_pair, resize_working, sample_image, uploaded_image


def sharpen_laplacian(image: np.ndarray, amount: float) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim == 2:
        laplace = ndi.convolve(image, K_LAPLACE, mode="reflect")
    else:
        laplace = np.stack(
            [ndi.convolve(image[..., channel], K_LAPLACE, mode="reflect") for channel in range(3)],
            axis=2,
        )
    return clamp01(image - amount * laplace), laplace


def unsharp_mask(image: np.ndarray, sigma: float, amount: float, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    blur_sigma = (sigma, sigma, 0.0) if image.ndim == 3 else sigma
    blurred = ndi.gaussian_filter(image, sigma=blur_sigma, mode="reflect")
    detail = image - blurred
    if threshold > 0:
        detail = np.where(np.abs(detail) >= threshold, detail, 0.0)
    return clamp01(image + amount * detail), detail


def render_tab() -> None:
    st.subheader("Làm sắc nét bằng Laplacian và Unsharp Masking")
    control, display = st.columns([0.32, 0.68], gap="large")
    with control:
        selected = st.selectbox("Ảnh mẫu", ["Ảnh người chụp", "Phi hành gia", "Tách cà phê"], key="sharp_sample")
        uploaded = st.file_uploader("Hoặc tải ảnh", type=["png", "jpg", "jpeg", "webp"], key="sharp_upload")
        method = st.radio("Phương pháp", ["Unsharp Masking", "Laplacian Sharpening"])
        if method == "Unsharp Masking":
            sigma = st.slider("Bán kính làm mờ σ", 0.5, 6.0, 2.0, 0.1)
            amount = st.slider("Mức tăng chi tiết", 0.0, 5.0, 1.5, 0.1)
            threshold = st.slider("Ngưỡng bảo vệ vùng phẳng", 0.0, 0.20, 0.01, 0.005)
        else:
            amount = st.slider("Hệ số Laplacian", 0.0, 2.0, 0.7, 0.05)

    source = resize_working(ensure_rgb(uploaded_image(uploaded, sample_image(selected))))
    if method == "Unsharp Masking":
        result, _detail = unsharp_mask(source, sigma, amount, threshold)
        explanation = "Ảnh mờ được trừ khỏi ảnh gốc để lấy chi tiết, sau đó cộng chi tiết trở lại với hệ số đã chọn."
    else:
        result, _detail = sharpen_laplacian(source, amount)
        explanation = "Laplacian làm nổi bật biến thiên bậc hai. Với quy ước kernel tâm -4, ảnh sắc nét được tính bằng ảnh gốc trừ đáp ứng Laplacian."

    with display:
        render_pair(source, result, method)
        st.info(explanation)
        st.download_button("Tải ảnh kết quả PNG", png_bytes(result), "chapter5_sharpen.png", "image/png")
