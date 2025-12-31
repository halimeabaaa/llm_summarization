import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline
import PyPDF2
import nltk
import os

# NLTK paketlerini kontrol et ve indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# -----------------------------------------------------------------------------
# 1. MODEL YAPILANDIRMASI
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device} üzerinde çalışılıyor...")

# === DEĞİŞİKLİK BURADA ===
# Bu model (mT5_multilingual_XLSum), 45 dilde (Türkçe dahil) özetleme yapmak için
# BBC verileriyle eğitilmiş, herkese açık ve silinmesi imkansız bir modeldir.
SUM_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

print(f"Özetleme modeli indiriliyor/yükleniyor: {SUM_MODEL_NAME}...")
# Not: Bu model yaklaşık 1GB - 2GB arasıdır. İlk indirme internet hızına göre sürer.
tokenizer_sum = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME).to(device)

# Soru-Cevap Modeli
QA_MODEL_NAME = "savasy/bert-base-turkish-squad"
print(f"Soru-Cevap modeli yükleniyor: {QA_MODEL_NAME}...")
tokenizer_qa = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
model_qa = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME).to(device)

qa_pipeline = pipeline("question-answering", model=model_qa, tokenizer=tokenizer_qa,
                       device=0 if device == "cuda" else -1)

print("Tüm Modeller Hazır!\n")


# -----------------------------------------------------------------------------
# 2. PDF OKUMA VE TEMİZLEME
# -----------------------------------------------------------------------------

def clean_and_read_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    full_text = ""
    # Performans için ilk 10 sayfa + son 3 sayfa (özet ve sonuç genelde buradadır)
    pages_to_read = list(range(min(10, len(reader.pages))))
    if len(reader.pages) > 15:
        pages_to_read += list(range(len(reader.pages) - 3, len(reader.pages)))

    unique_pages = sorted(list(set(pages_to_read)))

    for i in unique_pages:
        text = reader.pages[i].extract_text()
        if text:
            # Satır sonu tirelerini birleştir
            text = text.replace("-\n", "")
            # Gereksiz satır boşluklarını sil
            text = text.replace("\n", " ")
            full_text += text + " "
    return full_text


def semantic_sliding_window(text, max_tokens=512, overlap=2):
    """Metni cümle bütünlüğünü bozmadan parçalara ayırır."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # mT5 tokenizer ile token uzunluğunu ölç
        token_len = len(tokenizer_sum.encode(sentence, add_special_tokens=False))

        if current_length + token_len < max_tokens:
            current_chunk.append(sentence)
            current_length += token_len
        else:
            chunks.append(" ".join(current_chunk))
            # Kayar pencere: Son 'overlap' kadar cümleyi yeni parçaya taşı
            keep_last = current_chunk[-overlap:] if len(current_chunk) > overlap else []
            current_chunk = keep_last + [sentence]
            current_length = sum([len(tokenizer_sum.encode(s, add_special_tokens=False)) for s in current_chunk])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -----------------------------------------------------------------------------
# 3. ÖZETLEME VE SORU CEVAP (CORE)
# -----------------------------------------------------------------------------

def generate_recursive_summary(text):
    """Uzun metinleri parçalayıp özetler."""
    # mT5 modeli için 512 token güvenli sınırdır.
    chunks = semantic_sliding_window(text, max_tokens=500, overlap=2)

    print(f"   -> Metin {len(chunks)} parçaya bölündü. Özetleniyor...")

    partial_summaries = []

    for i, chunk in enumerate(chunks):
        # mT5 özel formatı: metnin başına bir şey eklemeye gerek yok ama
        # dil belirtmek gerekebilir. Bu model otomatik algılar.
        input_ids = tokenizer_sum(
            chunk,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids.to(device)

        summary_ids = model_sum.generate(
            input_ids,
            max_length=100,  # Parça özeti kısa olsun
            min_length=20,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)
        partial_summaries.append(summary)
        print(f"      Parça {i + 1}/{len(chunks)} bitti.")

    combined_text = " ".join(partial_summaries)

    # Final Özet
    print("   -> Parçalar birleştiriliyor ve son özet çıkarılıyor...")
    if len(combined_text.split()) > 100:  # Eğer birleşim hala uzunsa tekrar özetle
        input_ids = tokenizer_sum(
            combined_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids.to(device)

        summary_ids = model_sum.generate(input_ids, max_length=200, num_beams=4)
        final_summary = tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)
        return final_summary
    else:
        return combined_text


def extract_exact_answer(text, question):
    """BERT ile metin içinde cevap arar."""
    # BERT için pencere boyutu
    chunks = semantic_sliding_window(text, max_tokens=350, overlap=5)

    best_score = float('-inf')
    best_answer = "Belirtilmemiş."

    keywords = [w for w in question.lower().split() if len(w) > 3]
    relevant_chunks = [c for c in chunks if any(k in c.lower() for k in keywords)]
    if not relevant_chunks: relevant_chunks = chunks[:5]

    for chunk in relevant_chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)
            # Skor eşiğini biraz düşük tutalım ki cevap kaçmasın
            if result['score'] > best_score and result['score'] > 0.0001:
                best_score = result['score']

                # Cümle tamamlama
                ans_start = result['start']
                ans_end = result['end']

                sent_start = chunk.rfind('.', 0, ans_start) + 1
                sent_end = chunk.find('.', ans_end)
                if sent_end == -1: sent_end = len(chunk)

                # Bazen BERT çok kısa cevap döner (ör: "Ali"), bunu engellemek için tüm cümleyi alıyoruz.
                candidate_sent = chunk[sent_start:sent_end].strip()
                if len(candidate_sent) > 5:  # Çok kısa gürültüleri ele
                    best_answer = candidate_sent
        except:
            continue

    return best_answer


# -----------------------------------------------------------------------------
# 4. ÇALIŞTIRMA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pdf_path = "makale2.pdf"  # PDF DOSYA ADINI KONTROL ET

    if os.path.exists(pdf_path):
        print(f"'{pdf_path}' okunuyor...\n")
        full_text = clean_and_read_pdf(pdf_path)

        if len(full_text) < 100:
            print("HATA: PDF metni çok kısa veya okunamadı.")
        else:
            print("-" * 50)
            print("1. ÖZETLEME (mT5 - BBC Model)")
            print("-" * 50)
            ozet = generate_recursive_summary(full_text)
            print(f"\n>>> ÖZET:\n{ozet}\n")

            print("-" * 50)
            print("2. SORU - CEVAP (BERT Model)")
            print("-" * 50)

            sorular = [
                "Çalışmanın temel amacı nedir?",
                "Hangi yöntemler kullanılmıştır?",
                "Kaynakları nelerdir?",
                "Sonuç ne olmuştur?"
            ]

            for s in sorular:
                c = extract_exact_answer(full_text, s)
                print(f"Soru: {s}")
                print(f"Cevap: {c}\n")
    else:
        print("Dosya bulunamadı.")