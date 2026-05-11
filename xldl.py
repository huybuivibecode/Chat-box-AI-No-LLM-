import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def stop_word_vi(text):
    stop = stopwords.words('vietnamese.txt')
    text_pre = " ".join(text for text in text.split() if text not in stop)
    return text_pre
from bs4 import BeautifulSoup
def remove_tags(html): 
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script']):
        data.decompose() 
    return ' '.join(soup.stripped_strings)
import re
def only_word1(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()
from pyvi import ViTokenizer
def tach_tu(text):
    text = ViTokenizer.tokenize(text)
    return text
def expand_abbreviations(text: str) -> str:
    """
    Hàm mở rộng các chữ viết tắt phổ biến trong ngữ cảnh sinh viên – DUE.
    Ví dụ:
        "ĐKHP sao rồi?" → "Đăng ký học phần sao rồi?"
        "SV năm nhất cần làm gì?" → "Sinh viên năm nhất cần làm gì?"
    """
    ABBR = {
        "ĐKHP": "Đăng ký học phần",
        "DUE": "Đại học Kinh tế – Đại học Đà Nẵng",
        "UDN": "Đại học Đà Nẵng",
        "SV": "Sinh viên",
        "CVHT": "Cố vấn học tập",
        "CỐ VẤN HT": "Cố vấn học tập",
        "CTSV": "Phòng Công tác sinh viên",
        "PĐT": "Phòng Đào tạo",
        "HP": "Học phí",
        "KTX": "Ký túc xá",
        "DRL": "Điểm rèn luyện",
        "GPA": "Điểm trung bình học tập",
        "HĐTN": "Hoạt động tình nguyện",
        "CLB": "Câu lạc bộ",
        "SV5T": "Sinh viên 5 tốt",
    }

    # Đảm bảo input là chuỗi
    if not isinstance(text, str):
        text = str(text)

    for abbr, full in ABBR.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)

    # Chuẩn hóa lại khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text

#--------------------------------
def main(text):
    preprocessing=stop_word_vi(text)
    preprocessing=remove_tags(preprocessing)
    preprocessing=expand_abbreviations(preprocessing)
    preprocessing=only_word1(preprocessing)
    preprocessing=tach_tu(preprocessing)
    
    return preprocessing


