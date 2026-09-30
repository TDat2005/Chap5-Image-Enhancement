"""Diem vao cua Streamlit; cac tab duoc tach rieng theo tung chu de."""

import streamlit as st

from web_parts.web_blending import render_tab as render_blending_tab
from web_parts.web_edges import render_tab as render_edges_tab
from web_parts.web_sharpening import render_tab as render_sharpening_tab


st.set_page_config(
    page_title="Chapter 5 Image Lab",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Chapter 5 Image Lab")
st.caption("Thử trực tiếp các kỹ thuật tăng cường ảnh bằng đạo hàm trong Chapter 5")

with st.sidebar:
    st.header("Hướng dẫn")
    st.write("Chọn ảnh mẫu hoặc tải ảnh của bạn, sau đó kéo tham số để quan sát kết quả.")
    st.write("Ảnh chỉ được xử lý trong phiên làm việc và không được lưu tự động.")
    st.divider()
    st.subheader("Ứng dụng tương tự")
    st.markdown(
        "- [Photoshop Unsharp Mask](https://helpx.adobe.com/photoshop/desktop/effects-filters/smart-filters/sharpen-images-with-unsharp-mask.html)\n"
        "- [GIMP Edge Detect](https://docs.gimp.org/3.0/en/gimp-filter-edge.html)\n"
        "- [ImageJ Find Edges](https://imagej.net/ij/docs/guide/146-29.html)"
    )
    st.caption("Photoshop, GIMP và ImageJ là ứng dụng cho người dùng. SciPy và scikit-image là thư viện đứng phía sau web này.")

tab_sharpen, tab_edges, tab_blend = st.tabs(
    ["Làm sắc nét", "Phát hiện biên", "Pyramid Blending"]
)

with tab_sharpen:
    render_sharpening_tab()

with tab_edges:
    render_edges_tab()

with tab_blend:
    render_blending_tab()

st.divider()
st.caption("Nhóm 11 · Chapter 5 · Tăng cường ảnh bằng đạo hàm · Bản chạy local")
