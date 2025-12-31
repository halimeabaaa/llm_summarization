import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import PyPDF2


# 1. PDF OKUMA FONKSİYONU
def read_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text


# 2. METNİ CÜMLELERE BÖLME
def split_into_sentences(text):
    # Basitçe nokta ile ayırıyoruz, daha gelişmişi için NLTK kullanılabilir
    sentences = text.replace("\n", " ").split(". ")
    # Çok kısa cümleleri (başlık vs) eliyoruz
    return [s.strip() for s in sentences if len(s.split()) > 5]


# 3. S-BERT İLE ÖZETLEME (Extractive)
def extractive_summary(text, num_sentences=5):
    # Modeli Yükle (İlk çalışmada indirir, sonra cache'den okur)
    # 'all-MiniLM-L6-v2' çok hızlı ve başarılı bir S-BERT modelidir.
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    sentences = split_into_sentences(text)

    if len(sentences) == 0:
        return "Yeterli metin bulunamadı."

    # Cümleleri Vektöre Çevir (Embedding)
    embeddings = model.encode(sentences)

    # K-Means ile cümleleri grupla (Metindeki ana konu sayısına göre)
    # Küme sayısı = İstenen özet cümle sayısı
    num_clusters = min(num_sentences, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)

    # Her kümenin merkezine en yakın cümleyi bul
    summary_sentences = []
    for i in range(num_clusters):
        # Küme merkezi
        center = kmeans.cluster_centers_[i]

        # Bu kümeye ait cümlelerin vektörleri
        # (Basitlik için tüm cümlelere uzaklığa bakıyoruz)
        distances = np.linalg.norm(embeddings - center, axis=1)

        # En yakın cümlenin indeksi
        closest_idx = np.argmin(distances)
        summary_sentences.append(sentences[closest_idx])

    return ".\n".join(summary_sentences) + "."


# --- TEST ---
pdf_yolu = "ornek_makale.pdf"  # Buraya kendi PDF yolunu yaz
full_text = read_pdf(pdf_yolu)

print("--- DERİN ÖĞRENME TABANLI ÖZET (S-BERT) ---")
print(extractive_summary(full_text, num_sentences=4))