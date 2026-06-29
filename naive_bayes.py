import re
import math

# ==========================================================================
# METADATA EMOSI & BOBOT SKALA SENTIMEN (Skala Nilai 1 s/d 7)
# ==========================================================================
EMOTIONS_METADATA = {
    "Senang": { "value": 1, "sentiment": "Positif", "emoji": "😊" },
    "Percaya": { "value": 2, "sentiment": "Positif", "emoji": "🤝" },
    "Terkejut": { "value": 3, "sentiment": "Positif", "emoji": "😲" },
    "Netral": { "value": 4, "sentiment": "Netral", "emoji": "😐" },
    "Sedih": { "value": 5, "sentiment": "Negatif", "emoji": "😢" },
    "Takut": { "value": 6, "sentiment": "Negatif", "emoji": "😨" },
    "Marah": { "value": 7, "sentiment": "Negatif", "emoji": "😡" }
}

# Daftar stopwords bahasa Indonesia untuk memfilter kata hubung / kata depan umum
STOPWORDS = {
    "dan", "yang", "di", "ke", "dari", "itu", "ini", "untuk", "adalah", "pada", 
    "adapun", "oleh", "bahwa", "ia", "mereka", "kita", "kami", "dia", "anda", 
    "tersebut", "secara", "serta", "juga", "pun", "yaitu", "yakni", "ialah", 
    "merupakan", "sebagai", "bagi", "tentang", "mengenai", "saja"
}

# ==========================================================================
# FUNGSI PREPROCESSING TEKS
# ==========================================================================
def preprocess_text(text):
    """
    Melakukan normalisasi teks (Case Folding, Filtering Karakter, Tokenisasi, Stopword)
    """
    if not text:
        return []
    
    # 1. Case Folding: Mengubah huruf kapital menjadi huruf kecil semuanya
    clean = text.lower()
    
    # 2. Filtering: Hanya mempertahankan huruf kecil a-z dan spasi (simbol, angka, tanda baca dihapus)
    clean = re.sub(r'[^a-z\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    # 3. Tokenize: Memecah kalimat menjadi array kata-kata
    raw_tokens = clean.split()
    
    # 4. Stopword Removal: Memfilter kata hubung yang tidak membawa makna emosi kuat
    tokens = [w for w in raw_tokens if w not in STOPWORDS]
    
    return tokens

def trace_preprocessing_steps(text):
    """
    Fungsi penelusuran untuk merekam detail kata yang disimpan (aktif) vs disaring (discarded)
    """
    if not text:
        return []
    
    clean = text.lower()
    clean = re.sub(r'[^a-z\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    raw_tokens = clean.split()
    
    steps = []
    for tok in raw_tokens:
        is_stop = tok in STOPWORDS
        steps.append({
            "token": tok,
            "status": "discarded" if is_stop else "active",
            "reason": "Stopword" if is_stop else "Kata Kunci"
        })
    return steps

# ==========================================================================
# MODEL NAIVE BAYES CLASSIFIER
# ==========================================================================
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = ["Senang", "Percaya", "Terkejut", "Netral", "Sedih", "Takut", "Marah"]
        self.class_doc_count = {}     # N_C: Jumlah ulasan per kelas emosi
        self.class_word_count = {}    # Total kata yang terdaftar di masing-masing kelas emosi
        self.word_count_per_class = {}# Frekuensi kemunculan tiap kata unik di masing-masing kelas emosi
        self.vocabulary = set()       # V: Kumpulan semua kosakata unik dalam dataset latih
        self.total_docs = 0           # N_total: Total seluruh dokumen data latih

    def train(self, dataset):
        """
        Melatih model: Menghitung parameter Prior P(C) dan Likelihood P(w|C)
        """
        self.class_doc_count = {c: 0 for c in self.classes}
        self.class_word_count = {c: 0 for c in self.classes}
        self.word_count_per_class = {c: {} for c in self.classes}
        self.vocabulary = set()
        self.total_docs = len(dataset)

        # Proses penghitungan dokumen dan kata per kelas emosi
        for item in dataset:
            label = item["label"]
            if label not in self.classes:
                continue
            self.class_doc_count[label] += 1
            
            tokens = preprocess_text(item["text"])
            for t in tokens:
                self.vocabulary.add(t)
                self.word_count_per_class[label][t] = self.word_count_per_class[label].get(t, 0) + 1
                self.class_word_count[label] += 1

    def predict(self, text):
        """
        Melakukan prediksi emosi dan merinci step-by-step perhitungannya secara matematis
        """
        tokens = preprocess_text(text)
        results = []
        v_size = len(self.vocabulary) if len(self.vocabulary) > 0 else 1

        for c in self.classes:
            # 1. RUMUS PROBABILITAS PRIOR: P(C) = N_C / N_total
            doc_count = self.class_doc_count.get(c, 0)
            prior = doc_count / self.total_docs if self.total_docs > 0 else 0
            log_prior = math.log(prior) if prior > 0 else -999.0
            
            log_score = log_prior
            likelihoods = []

            for t in tokens:
                # 2. RUMUS LIKELIHOOD DENGAN LAPLACE SMOOTHING: P(w|C) = (count(w, C) + 1) / (sum count(w', C) + |V|)
                word_freq = self.word_count_per_class.get(c, {}).get(t, 0)
                class_total_words = self.class_word_count.get(c, 0)
                
                # Penerapan Laplace Smoothing (+1 pembilang dan +|V| pembagi)
                prob = (word_freq + 1) / (class_total_words + v_size)
                log_prob = math.log(prob)
                
                # Tambahkan likelihood log ke akumulator Log Posterior
                log_score += log_prob
                
                likelihoods.append({
                    "word": t,
                    "freq": word_freq,
                    "class_total_words": class_total_words,
                    "vocab_size": v_size,
                    "prob": prob,
                    "log_prob": log_prob
                })

            results.append({
                "class": c,
                "prior": prior,
                "log_prior": log_prior,
                "likelihoods": likelihoods,
                "log_score": log_score
            })

        # 3. RUMUS NORMALISASI POSTERIOR (Softmax via Log-Sum-Exp Trick)
        # Menghindari numeric underflow dengan mengurangkan log score dengan nilai log score maksimum
        max_log_score = max(r["log_score"] for r in results)
        sum_exp = 0.0
        for r in results:
            r["exp_score"] = math.exp(r["log_score"] - max_log_score)
            sum_exp += r["exp_score"]

        for r in results:
            r["probability"] = r["exp_score"] / sum_exp if sum_exp > 0 else 0.0

        # Urutkan berdasarkan skor log posterior tertinggi
        results.sort(key=lambda x: x["log_score"], reverse=True)

        # 4. RUMUS SKOR INDEKS EMOSI RATA-RATA (Expected Value dari bobot probabilitas)
        # E[V] = sum( P(C_i | D) * Value(C_i) ) di mana Nilai berkisar 1 (Senang) s.d 7 (Marah)
        expected_val = 0.0
        for r in results:
            expected_val += r["probability"] * EMOTIONS_METADATA[r["class"]]["value"]

        return {
            "prediction": results[0]["class"],
            "tokens": tokens,
            "scores": results,
            "expected_value": expected_val
        }
