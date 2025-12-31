import os
import torch
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import warnings

# Gereksiz uyarıları gizle
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. AYARLAR VE MODEL YÜKLEME (Global)
# ---------------------------------------------------------
print("Modeller yükleniyor... (İlk çalıştırışta zaman alabilir)")

# A) Özetleme Modeli (S-BERT)
# 'all-MiniLM-L6-v2' hem hızlıdır hem de CPU'da rahat çalışır.
summarizer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# B) Soru-Cevap Modeli (BERT-TURK)
# Türkçe SQuAD verisiyle eğitilmiş model.
qa_tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

print("Modeller hazır!\n")


# ---------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------

def read_pdf(file_path):
    """PDF dosyasını okur ve metne çevirir."""
    if not os.path.exists(file_path):
        return None

    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extract = page.extract_text()
        if extract:
            text += extract + " "
    return text


def split_into_sentences(text):
    """Metni cümlelere böler."""
    # Basit bölme. Daha iyisi için 'nltk.sent_tokenize' kullanılabilir.
    sentences = text.replace("\n", " ").split(". ")
    # Çok kısa cümleleri (başlık vs) temizle
    return [s.strip() for s in sentences if len(s.split()) > 4]


def get_extractive_summary(text, num_sentences=5):
    """S-BERT ve K-Means kullanarak metnin özetini çıkarır."""
    sentences = split_into_sentences(text)

    if len(sentences) < num_sentences:
        return " ".join(sentences)

    # Cümleleri vektöre çevir
    embeddings = summarizer_model.encode(sentences)

    # Kümeleme yap (Her küme, metindeki bir ana fikri temsil eder)
    num_clusters = min(num_sentences, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    summary_sentences = []
    for i in range(num_clusters):
        # Merkeze en yakın cümleyi bul
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(embeddings - center, axis=1)
        closest_idx = np.argmin(distances)
        summary_sentences.append(sentences[closest_idx])

    return ".\n- ".join(summary_sentences) + "."


def extract_specific_info(text, question):
    """BERT-TURK ile metnin içinden soruya cevap arar."""
    # Metni paragraflara böl
    paragraphs = text.split("\n\n")
    best_answer = {"score": 0, "answer": "Bulunamadı"}

    # Anahtar kelime filtrelemesi (Hız için)
    keywords = ["yöntem", "metod", "algoritma", "veri", "sonuç", "kaynak"]
    search_context = []

    for p in paragraphs:
        # Paragraf çok kısaysa atla
        if len(p) < 50: continue
        # Soru ile ilgili olabilecek paragrafları seç
        search_context.append(p)

    # BERT'in okuyabileceği boyutta parçalar halinde sor (Sliding Window)
    # Basitlik için ilk 3 uygun paragrafı ve metnin başını tarıyoruz
    candidates = search_context[:5]

    for context in candidates:
        try:
            result = qa_pipeline(question=question, context=context)
            # Eğer skor yüksekse cevabı güncelle
            if result['score'] > best_answer['score']:
                best_answer = result
        except:
            continue

    return best_answer['answer']


# ---------------------------------------------------------
# 3. ANA ÇALIŞTIRMA BLOĞU
# ---------------------------------------------------------
if __name__ == "__main__":
    # A) PDF Dosyasının Adı
    pdf_adi = "makale2.pdf"  # <-- BURAYA PDF ADINI YAZACAKSIN

    print(f"'{pdf_adi}' dosyası okunuyor...")
    full_text = read_pdf(pdf_adi)

    if full_text:
        print(f"Okuma başarılı. Metin uzunluğu: {len(full_text)} karakter.\n")

        print("-" * 40)
        print("OTOMATİK MAKALE ANALİZ RAPORU")
        print("-" * 40)

        # 1. ÖZET ÇIKAR
        print("\n>>> ÖZET (S-BERT ile Çıkarıldı):")
        ozet = get_extractive_summary(full_text, num_sentences=4)
        print(f"- {ozet}")

        # 2. BİLGİ ÇIKAR
        print("\n>>> DETAYLI ANALİZ (BERT-TURK ile Çıkarıldı):")

        sorular = [
            "Bu çalışmada hangi yöntem kullanılmıştır?",
            "Araştırmanın amacı nedir?",
            "Hangi veri seti kullanılmıştır?"
        ]

        for soru in sorular:
            cevap = extract_specific_info(full_text, soru)
            print(f"\nSORU: {soru}")
            print(f"CEVAP: {cevap}")

    else:
        print(f"HATA: '{pdf_adi}' dosyası bulunamadı. Lütfen dosya adını kontrol edin.")