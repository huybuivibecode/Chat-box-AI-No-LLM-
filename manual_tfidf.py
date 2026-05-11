# manual_tfidf.py
import re, math
import numpy as np
from collections import Counter

def simple_tokenize(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]

class ManualTfidfVectorizer:
    def __init__(self, tokenizer=simple_tokenize):
        self.tokenizer = tokenizer
        self.vocab_ = None
        self.term_index_ = None
        self.idf_ = None

    def fit(self, texts):
        docs_tokens = [self.tokenizer(x or "") for x in texts]
        N = len(docs_tokens)

        vocab_set, df_counts = set(), Counter()
        for tokens in docs_tokens:
            uniq = set(tokens)
            vocab_set |= uniq
            df_counts.update(uniq)

        self.vocab_ = sorted(vocab_set)
        self.term_index_ = {t:i for i,t in enumerate(self.vocab_)}
        V = len(self.vocab_)
        self.idf_ = np.array([math.log(N / df_counts[t]) if df_counts[t] else 0 for t in self.vocab_])
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab_)))
        for i, text in enumerate(texts):
            tokens = self.tokenizer(text or "")
            if not tokens: continue
            counts = Counter(tokens)
            total = len(tokens)
            for term, c in counts.items():
                j = self.term_index_.get(term)
                if j is not None:
                    X[i, j] = (c / total) * self.idf_[j]
        return X

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
