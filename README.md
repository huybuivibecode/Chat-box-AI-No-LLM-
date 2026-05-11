# Chatbot FAQ DUE

**Chủ sở hữu**: Bùi Quang Huy

**Liên hệ**: buiquanghuy352k5@gmail.com

## 1. Mục đích

Dự án này là một chatbot FAQ dành cho Đại học Kinh tế (DUE). Ứng dụng trả lời các câu hỏi phổ biến về học vụ, đăng ký học phần, quy định, CTSV và thủ tục bằng cách tìm câu trả lời gần nhất trong dữ liệu FAQ hiện có.

## 2. Cách hoạt động

1. Người dùng nhập câu hỏi tiếng Việt vào giao diện Streamlit.
2. Ứng dụng có thể tiền xử lý câu hỏi bằng `xldl.py`:
   - loại bỏ stopword
   - loại bỏ thẻ HTML
   - mở rộng viết tắt (ví dụ `ĐKHP` → `Đăng ký học phần`)
   - tách từ tiếng Việt
3. Câu hỏi được chuyển thành vector TF-IDF bằng `manual_tfidf.py`.
4. Vector TF-IDF được so sánh với các embeddings đã lưu trong `ChromaDB`.
5. Ứng dụng chọn câu trả lời tốt nhất dựa trên độ tương đồng cosine và hiển thị kèm top-k kết quả liên quan.

> Lưu ý: Đây không phải chatbot tạo ngôn ngữ (LLM). Nó hoạt động theo kiểu tìm kiếm thông tin dựa trên dữ liệu FAQ có sẵn.

## 3. Thành phần chính

- `app.py`
  - Giao diện Streamlit, xử lý người dùng, truy vấn ChromaDB và hiển thị kết quả.
  - Tải model từ `tfidf_model.pkl` và kết nối tới collection `faq_tfidf` trong `chroma_db/`.

- `manual_tfidf.py`
  - Cài đặt TF-IDF thủ công bằng NumPy.
  - Tạo ma trận TF-IDF cho văn bản đầu vào.

- `xldl.py`
  - Hàm tiền xử lý tiếng Việt.
  - Bao gồm loại bỏ stopword, loại thẻ HTML, mở rộng chữ viết tắt và tách từ.

- `tfidf_model.pkl`
  - File lưu model TF-IDF đã huấn luyện.
  - Dùng để chuyển câu hỏi mới thành vector giống với dữ liệu gốc.

- `chroma_db/`
  - Thư mục lưu dữ liệu persistent của ChromaDB.
  - Chứa embeddings và metadata FAQ.

- `data.csv`
  - Dữ liệu gốc ban đầu hoặc tham chiếu dữ liệu FAQ.

- `vietnamese.txt`
  - Tệp chứa dữ liệu ngôn ngữ hỗ trợ tiền xử lý.

- `main.ipynb`
  - Notebook dùng để cập nhật hoặc tái tạo dữ liệu FAQ và embeddings TF-IDF.
  - Có thể dùng để thêm dữ liệu mới hoặc rebuild toàn bộ bộ nhớ tìm kiếm.

- `setup.ipynb`
  - Notebook cài đặt phụ thuộc NLTK.

## 4. Hướng dẫn chạy

1. Cài Python 3.8+.
2. Cài phụ thuộc:

```bash
pip install -r requirements.txt
```

3. Chạy ứng dụng:

```bash
streamlit run app.py
```

4. Mở đường dẫn hiển thị trong terminal (thường là `http://localhost:8501`).

5. Nhập câu hỏi vào ô chat và xem kết quả.

## 5. Các thông số quan trọng

- `Top-k`: số lượng kết quả ChromaDB trả về để hiển thị danh sách kết quả liên quan.
- `Ngưỡng tương đồng`: chỉ hiển thị câu trả lời khi độ tương đồng cosine lớn hơn ngưỡng này.
- Nếu không có câu trả lời đủ tốt, ứng dụng sẽ yêu cầu người dùng mô tả lại câu hỏi.

## 6. Ghi chú quan trọng

- Ứng dụng cần cả hai: `tfidf_model.pkl` và thư mục `chroma_db` để hoạt động đúng.
- Nếu thiếu `xldl.py`, vẫn chạy được nhưng không có bước tiền xử lý nâng cao.
- Nếu thêm dữ liệu mới hoặc thay đổi mạnh nội dung, nên rebuild lại TF-IDF và ChromaDB bằng notebook `main.ipynb`.

## 7. Yêu cầu phụ thuộc

- `streamlit`
- `chromadb`
- `numpy`
- `nltk`
- `beautifulsoup4`
- `pyvi`
- `pandas`

## 8. Đưa lên GitHub

1. Khởi tạo Git trong thư mục dự án:

```bash
git init
```

2. Thêm file vào staging area:

```bash
git add .
```

3. Commit lần đầu:

```bash
git commit -m "Initial commit"
```

4. Tạo repo trên GitHub bằng website hoặc GitHub CLI.

5. Thêm remote và push lên GitHub:

```bash
git remote add origin https://github.com/<username>/<repo>.git
git branch -M main
git push -u origin main
```

## 9. Cách deploy web

### Option 1: Streamlit Community Cloud

- Đăng nhập vào https://streamlit.io/cloud
- Kết nối repo GitHub của bạn
- Chọn branch `main` và file `app.py`
- Streamlit sẽ tự động cài đặt `requirements.txt` và khởi chạy ứng dụng.

### Option 2: Render / Railway / Heroku

- Tạo dịch vụ Web Python mới.
- Chỉ định branch GitHub và repo của bạn.
- Nếu cần, dùng lệnh khởi động:

```bash
streamlit run app.py --server.port $PORT
```

- Đảm bảo repo có `requirements.txt` và `Procfile`.

### Lưu ý

- Nếu `chroma_db/` hoặc `tfidf_model.pkl` chưa nằm trong GitHub và ứng dụng cần chúng để chạy, hãy commit hai mục này nếu kích thước không quá lớn.
- Nếu dữ liệu quá lớn, bạn có thể cấu hình một bước build riêng để tạo `tfidf_model.pkl` và `chroma_db` trên môi trường deploy.
