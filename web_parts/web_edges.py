"""Tab phat hien bien: cac toan tu gradient, Canny va LoG."""

import numpy as np
import streamlit as st
from scipy import ndimage as ndi
from skimage import feature, filters

from .web_common import K_LAPLACE, ensure_gray, png_bytes, render_pair, resize_working, sample_image, uploaded_image


def normalize_response(image: np.ndarray) -> np.ndarray:
    image = np.abs(np.asarray(image, dtype=float))
    high = float(np.percentile(image, 99.5))
    if high <= 1e-12:
        return np.zeros_like(image)
    return np.clip(image / high, 0.0, 1.0)


def zero_crossings(log_image: np.ndarray, threshold_ratio: float) -> np.ndarray:
    maximum = ndi.maximum_filter(log_image, size=3, mode="reflect")
    minimum = ndi.minimum_filter(log_image, size=3, mode="reflect")
    contrast = maximum - minimum
    threshold = threshold_ratio * float(np.max(np.abs(log_image)))
    return (maximum > 0) & (minimum < 0) & (contrast > threshold)


def edge_result(gray: np.ndarray, method: str, sigma: float, low_threshold: float,
                high_threshold: float, zero_threshold: float) -> tuple[np.ndarray, str]:
    if method == "Sobel":
        return filters.sobel(gray), "Sobel: độ lớn gradient"
    if method == "Prewitt":
        return filters.prewitt(gray), "Prewitt: độ lớn gradient"
    if method == "Roberts":
        return filters.roberts(gray), "Roberts: đáp ứng đường chéo"
    if method == "Scharr":
        return filters.scharr(gray), "Scharr: độ lớn gradient"
    if method == "Laplacian":
        return normalize_response(ndi.convolve(gray, K_LAPLACE, mode="reflect")), "Trị tuyệt đối Laplacian"
    if method == "Canny":
        result = feature.canny(gray, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        return result.astype(float), "Biên Canny"
    log_image = (sigma ** 2) * ndi.gaussian_laplace(gray, sigma=sigma, mode="reflect")
    return zero_crossings(log_image, zero_threshold).astype(float), "Zero crossing của LoG"


def render_tab() -> None:
    st.subheader("So sánh các toán tử phát hiện biên")
    control, display = st.columns([0.32, 0.68], gap="large")
    with control:
        selected = st.selectbox("Ảnh mẫu", ["Ảnh người chụp", "Đồng xu", "Phi hành gia"], key="edge_sample")
        uploaded = st.file_uploader("Hoặc tải ảnh", type=["png", "jpg", "jpeg", "webp"], key="edge_upload")
        method = st.selectbox("Toán tử", ["Sobel", "Prewitt", "Roberts", "Scharr", "Laplacian", "Canny", "LoG zero crossing"])
        sigma = 1.5
        low = 0.08
        high = 0.20
        zero_threshold = 0.05
        if method in {"Canny", "LoG zero crossing"}:
            sigma = st.slider("Thang Gaussian σ", 0.5, 6.0, 1.5, 0.1, key="edge_sigma")
        if method == "Canny":
            low = st.slider("Ngưỡng thấp", 0.0, 0.50, 0.08, 0.01)
            high = st.slider("Ngưỡng cao", max(low + 0.01, 0.02), 1.0, max(0.20, low + 0.01), 0.01)
        if method == "LoG zero crossing":
            zero_threshold = st.slider("Ngưỡng tương phản zero crossing", 0.0, 0.30, 0.05, 0.01)

    source = resize_working(uploaded_image(uploaded, sample_image(selected)))
    gray = ensure_gray(source)
    result, result_title = edge_result(gray, method, sigma, low, high, zero_threshold)
    with display:
        render_pair(gray, result, result_title)
        edge_ratio = float(np.mean(result > 0.5)) * 100.0
        st.metric("Tỷ lệ điểm ảnh biên hoặc đáp ứng mạnh", f"{edge_ratio:.2f}%")
        st.download_button("Tải ảnh biên PNG", png_bytes(result), "chapter5_edges.png", "image/png")
