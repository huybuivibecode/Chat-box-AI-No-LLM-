import os

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---- Thử import xldl; nếu thiếu thì dùng hàm rỗng ----
try:
    import xldl
    def preprocess(text: str) -> str:
        return xldl.main(text)
except Exception:
    def preprocess(text: str) -> str:
        return text

# ---- Import TF-IDF thủ công của bạn ----
from manual_tfidf import ManualTfidfVectorizer

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(page_title="Chatbot FAQ DUE", page_icon="🎓", layout="centered")

# ================== CSS TUỲ BIẾN ==================
st.markdown("""
<style>
.main { background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%); }
h1 { font-weight: 800; letter-spacing: 0.2px; }
.block-container { padding-top: 1.5rem; padding-bottom: 6rem; }
.user-bubble {
  background: #e8f3ff;
  color: #0b3d91;
  padding: 12px 14px; border-radius: 16px;
  border: 1px solid #cfe6ff; display: inline-block;
  max-width: 650px; line-height: 1.5;
  box-shadow: 0 2px 8px rgba(56, 118, 255, 0.06);
}
.bot-bubble {
  background: #f2f3f5;
  color: #222;
  padding: 12px 14px; border-radius: 16px;
  border: 1px solid #e6e6e6; display: inline-block;
  max-width: 650px; line-height: 1.6;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.result-card{
  background: #ffffff; border: 1px solid #ebedf0;
  border-radius: 14px; padding: 10px 12px; margin-top: 8px;
}
.footer {
  position: fixed; left: 0; right: 0; bottom: 0;
  padding: 10px 12px; text-align: center;
  background: rgba(255,255,255,0.9);
  border-top: 1px solid #eee; font-style: italic; color: #555;
}
</style>
""", unsafe_allow_html=True)

# ================== TẢI MODEL & DỮ LIỆU ==================
@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    with open("tfidf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_faq_data():
    try:
        # Thử load từ data.csv
        df = pd.read_csv("data.csv")
        # Giả định có cột 'question' và 'answer'
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.warning("⚠️ data.csv không có cột 'question' hoặc 'answer'")
            return None, None
        
        questions = df['question'].fillna("").astype(str).tolist()
        answers = df['answer'].fillna("").astype(str).tolist()
        return questions, answers
    except Exception as e:
        st.warning(f"⚠️ Không thể load data.csv: {e}")
        return None, None

try:
    tfidf = load_tfidf_model()
    questions, answers = load_faq_data()
    
    if questions is None or answers is None:
        st.error("❌ Không thể tải dữ liệu FAQ. Kiểm tra file data.csv.")
        st.stop()
    
    # Tính embeddings cho tất cả câu hỏi
    faq_embeddings = tfidf.transform(questions).astype(float)
except Exception as e:
    st.error(f"❌ Lỗi khi tải model: {e}")
    st.stop()

# ================== HÀM TRUY VẤN ==================
SIM_THRESHOLD = 0.10

def embed_query(text: str):
    return tfidf.transform([text]).astype(float)

def search_best_answer(query: str, top_k: int = 3, threshold: float = SIM_THRESHOLD):
    query_vec = embed_query(query)
    
    # Tính cosine similarity
    similarities = cosine_similarity(query_vec, faq_embeddings)[0]
    
    # Sắp xếp giảm dần
    sorted_idx = np.argsort(-similarities)
    
    triples = []
    for idx in sorted_idx[:top_k]:
        sim = float(similarities[idx])
        triples.append({
            "idx": idx,
            "question": questions[idx],
            "answer": answers[idx],
            "sim": sim
        })
    
    # Lọc theo ngưỡng
    above = [t for t in triples if t["sim"] >= threshold]
    
    if not above:
        return None, triples
    
    best = above[0]
    return best, triples

# ================== GIAO DIỆN ==================
st.title("🎓 Chatbot FAQ – Đại học Kinh tế (DUE)")
st.caption("Hỏi đáp học vụ • Quy định • Đăng ký học phần • CTSV • Thủ tục")

with st.sidebar:
    st.markdown("### ⚙️ Tuỳ chọn")
    k = st.slider("Số kết quả xét (Top-k)", 1, 10, 3)
    thr = st.slider("Ngưỡng tương đồng tối thiểu (%)", 0, 100, int(SIM_THRESHOLD*100))
    threshold = thr / 100.0
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
    best, triples = search_best_answer(processed, top_k=k, threshold=threshold)

    # Soạn trả lời
    if best is None:
        reply = "Bạn có thể mô tả kỹ hơn về câu hỏi không, mình sẽ sẵn sàng trả lời"
    else:
        ans = best.get("answer", "").strip()
        if not ans:
            ans = best.get("question", "")
        reply = f"**{ans}**  \n\n_Độ tương đồng: {best['sim']*100:.1f}%_"

    # Hiển thị bot + Top kết quả
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"<div class='bot-bubble'>{reply}</div>", unsafe_allow_html=True)

        with st.expander("🔎 Kết quả liên quan (chi tiết)"):
            if not triples:
                st.write("Không có kết quả.")
            else:
                for i, t in enumerate(triples, 1):
                    ans_preview = t.get("answer", t["question"])
                    st.markdown(
                        f"<div class='result-card'>"
                        f"<b>#{i}</b> • Similarity: {t['sim']*100:.1f}%<br/>"
                        f"{ans_preview}"
                        f"</div>", unsafe_allow_html=True
                    )

    st.session_state.history.append({"role": "assistant", "content": reply})

# Footer
st.markdown("<div class='footer'>— tam thập lục —</div>", unsafe_allow_html=True)
