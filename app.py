import os
import streamlit as st
import chromadb
import pickle
import numpy as np

# ---- Disable chromadb telemetry trước khi import xldl ----
os.environ["ANONYMIZED_TELEMETRY"] = "False"
try:
    import xldl
    def preprocess(text: str) -> str:
        return xldl.main(text)
except Exception:
    def preprocess(text: str) -> str:
        return text  # không có xldl thì giữ nguyên

# ---- Import TF-IDF thủ công của bạn (để pickle load được an toàn) ----
from manual_tfidf import ManualTfidfVectorizer  # quan trọng: để pickle tìm thấy class

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(page_title="Chatbot FAQ DUE", page_icon="🎓", layout="centered")

# ================== CSS TUỲ BIẾN (bong bóng + màu) ==================
st.markdown("""
<style>
/* Nền tổng thể */
.main { background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%); }

/* Tiêu đề đẹp */
h1 { font-weight: 800; letter-spacing: 0.2px; }

/* Khung chat container */
.block-container { padding-top: 1.5rem; padding-bottom: 6rem; }

/* Bong bóng user */
.user-bubble {
  background: #e8f3ff;
  color: #0b3d91;
  padding: 12px 14px; border-radius: 16px;
  border: 1px solid #cfe6ff; display: inline-block;
  max-width: 650px; line-height: 1.5;
  box-shadow: 0 2px 8px rgba(56, 118, 255, 0.06);
}

/* Bong bóng assistant */
.bot-bubble {
  background: #f2f3f5;
  color: #222;
  padding: 12px 14px; border-radius: 16px;
  border: 1px solid #e6e6e6; display: inline-block;
  max-width: 650px; line-height: 1.6;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Hộp “top kết quả” */
.result-card{
  background: #ffffff; border: 1px solid #ebedf0;
  border-radius: 14px; padding: 10px 12px; margin-top: 8px;
}

/* Footer ký tên cố định dưới đáy */
.footer {
  position: fixed; left: 0; right: 0; bottom: 0;
  padding: 10px 12px; text-align: center;
  background: rgba(255,255,255,0.9);
  border-top: 1px solid #eee; font-style: italic; color: #555;
}
</style>
""", unsafe_allow_html=True)

# ================== TẢI MODEL & DB ==================
@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    with open("tfidf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def connect_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection("faq_tfidf")

try:
    tfidf = load_tfidf_model()
    collection = connect_chroma()
except Exception as e:
    st.error("❌ Không thể tải `tfidf_model.pkl` hoặc kết nối `./chroma_db`. Hãy đảm bảo đã build dữ liệu trước.")
    st.exception(e)
    st.stop()

# ================== HÀM XỬ LÝ / TRUY VẤN ==================
SIM_THRESHOLD = 0.10  # 10%

def embed_query(text: str):
    return tfidf.transform([text]).astype(float)[0].tolist()

def search_best_answer(query: str, top_k: int = 3, threshold: float = SIM_THRESHOLD):
    vec = embed_query(query)
    res = collection.query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # Tính similarity = 1 - distance (với cosine)
    triples = []
    for doc, dist, meta in zip(docs, dists, metas):
        sim = 1 - dist
        triples.append({"doc": doc, "meta": meta, "sim": float(sim)})

    if not triples:
        return None, []

    # Lọc theo ngưỡng
    above = [t for t in triples if t["sim"] >= threshold]
    if not above:
        return None, triples  # trả None để dùng câu fallback

    # Tốt nhất
    best = max(above, key=lambda x: x["sim"])
    return best, triples

# ================== GIAO DIỆN ==================
st.title("🎓 Chatbot FAQ – Đại học Kinh tế (DUE)")
st.caption("Hỏi đáp học vụ • Quy định • Đăng ký học phần • CTSV • Thủ tục")

with st.sidebar:
    st.markdown("### ⚙️ Tuỳ chọn")
    k = st.slider("Số kết quả xét (Top-k)", 1, 10, 3)
    thr = st.slider("Ngưỡng tương đồng tối thiểu (%)", 0, 100, int(SIM_THRESHOLD*100))
    SIM_THRESHOLD = thr / 100.0
    st.info("Mẹo: Soạn câu hỏi dễ hiểu, có từ khoá rõ ràng.\n\nVí dụ: *ĐKHP bổ sung khi nào?*")

# Khởi tạo lịch sử
if "history" not in st.session_state:
    st.session_state.history = []

# Hiện lịch sử
for h in st.session_state.history:
    role, content = h["role"], h["content"]
    avatar = "🧑‍🎓" if role == "user" else "🤖"
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div class='{bubble_class}'>{content}</div>", unsafe_allow_html=True)

# Ô chat
user_text = st.chat_input("Nhập câu hỏi của bạn…")

if user_text:
    # Hiển thị câu hỏi
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(f"<div class='user-bubble'>{user_text}</div>", unsafe_allow_html=True)
    st.session_state.history.append({"role": "user", "content": user_text})

    # Tiền xử lý
    processed = preprocess(user_text)

    # Truy vấn
    best, triples = search_best_answer(processed, top_k=k, threshold=SIM_THRESHOLD)

    # Soạn trả lời
    if best is None:
        reply = "Bạn có thể mô tả kỹ hơn về câu hỏi không, mình sẽ sẵn sàng trả lời"
    else:
        # Ưu tiên metadata['answer'] nếu có, ngược lại dùng doc
        meta = best.get("meta") or {}
        ans = meta.get("answer", "").strip() if isinstance(meta, dict) else ""
        if not ans:
            ans = best.get("doc", "")
        reply = f"**{ans}**  \n\n_Độ tương đồng: {best['sim']*100:.1f}%_"

    # Hiển thị bot + Top kết quả trong expander
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"<div class='bot-bubble'>{reply}</div>", unsafe_allow_html=True)

        with st.expander("🔎 Kết quả liên quan (chi tiết)"):
            if not triples:
                st.write("Không có kết quả.")
            else:
                for i, t in enumerate(triples, 1):
                    meta = t.get("meta") or {}
                    preview = meta.get("answer", t["doc"]) if isinstance(meta, dict) else t["doc"]
                    st.markdown(
                        f"<div class='result-card'>"
                        f"<b>#{i}</b> • Similarity: {t['sim']*100:.1f}%<br/>"
                        f"{preview}"
                        f"</div>", unsafe_allow_html=True
                    )

    st.session_state.history.append({"role": "assistant", "content": reply})

# Footer ký tên nhóm
st.markdown("<div class='footer'>— tam thập lục —</div>", unsafe_allow_html=True)
