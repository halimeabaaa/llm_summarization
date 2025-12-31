from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# 1. QA MODELİNİ YÜKLE (BERT-TURK)
# Bu model "Soru" ve "Metin" verildiğinde, cevabın metnin neresinde olduğunu bulur.
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


# 2. BİLGİ ÇIKARMA FONKSİYONU
def extract_info(text, question):
    # BERT'in token limiti (512) vardır. Metni parçalayıp sormamız gerekir.
    # Basitlik adına metnin ilk 2000 karakterinde veya "Yöntem" geçen kısımlarda arayalım.

    # Strateji: İçinde "yöntem", "metod", "algoritma" geçen paragrafları bulup soruyu onlara sor.
    paragraphs = text.split("\n\n")  # Paragraflara böl
    relevant_context = ""

    keywords = ["yöntem", "metod", "algoritma", "veri", "model"]

    for p in paragraphs:
        if any(word in p.lower() for word in keywords):
            relevant_context += p + " "

    if not relevant_context:
        relevant_context = text[:2000]  # Hiçbir şey bulamazsa başa bak

    try:
        result = qa_pipeline(question=question, context=relevant_context)
        return result['answer']
    except:
        return "Bulunamadı."


# --- TEST ---
print("\n--- BİLGİ ÇIKARIMI (BERT-TURK) ---")

sorular = [
    "Bu çalışmada hangi yöntem kullanılmıştır?",
    "Hangi veri seti kullanılmıştır?",
    "En iyi sonuç nedir?"
]

for soru in sorular:
    cevap = extract_info(full_text, soru)
    print(f"SORU: {soru}")
    print(f"CEVAP: {cevap}\n")