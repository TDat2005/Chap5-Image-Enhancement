# Chapter 5 — Image Enhancement Using Derivatives

Mã nguồn minh họa Chương 5 môn Nhập môn Xử lý ảnh của Nhóm 11, lớp 74DCTT25.

## Nội dung

- Đạo hàm ảnh, gradient và Laplacian.
- Làm trơn nhiễu và làm sắc nét bằng Laplacian/Unsharp Mask.
- Phát hiện biên: Sobel, Canny, LoG, DoG và zero-crossing.
- Gaussian Pyramid, Laplacian Pyramid, tái tạo ảnh và hòa trộn nhiều mức.
- Ứng dụng Streamlit để thử các phương pháp trực tiếp trên trình duyệt local.

## Cài đặt

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy từng phần code

```powershell
.\.venv\Scripts\python.exe .\demo_parts\demo_01_dao_ham.py
.\.venv\Scripts\python.exe .\demo_parts\demo_02_lam_sac_net.py
.\.venv\Scripts\python.exe .\demo_parts\demo_03_phat_hien_bien.py
.\.venv\Scripts\python.exe .\demo_parts\demo_04_pyramid_blending.py
.\.venv\Scripts\python.exe .\demo_parts\demo_05_kiem_tra.py
```

Chạy toàn bộ demo theo thứ tự:

```powershell
.\.venv\Scripts\python.exe .\Chapter5_Image_Enhancement_Demo.py
```

Kết quả ảnh và `metrics.json` được tạo trong thư mục `ch5_demo_outputs/`. Thư mục này không được đưa lên Git vì có thể tái tạo bằng code.

## Chạy web local

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\Chapter5_Web_App.py
```

Streamlit sẽ in địa chỉ `localhost` trong terminal. Web chỉ hoạt động khi lệnh trên vẫn đang chạy.

## Cấu trúc

- `demo_parts/`: năm file demo chạy độc lập và phần dùng chung.
- `web_parts/`: các màn hình xử lý ảnh của ứng dụng Streamlit.
- `Chapter5_Image_Enhancement_Demo.py`: chạy tuần tự toàn bộ demo.
- `Chapter5_Web_App.py`: điểm vào của ứng dụng web local.

