# -*- coding: utf-8 -*-
"""
Türkçe Akademik PDF Analiz Sistemi (Özet + Yöntem + Bulgular + Kaynakça)
- Çoklu PDF desteği (klasör veya liste)
- Section detection (Özet, Giriş, Yöntem, Bulgular, Sonuç, Kaynakça)
- Hiyerarşik özetleme (mT5 XLSum)
- QA (BERT Turkish SQuAD) + TF-IDF chunk retrieval
- Kaynakça parse (deterministik)
"""

import os
import re
import json
import torch
import nltk
import PyPDF2
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering, pipeline
)

# TF-IDF retrieval (opsiyonel ama önerilir)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# NLTK
# ----------------------------
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

ensure_nltk()

TR_ABBR = ["dr", "prof", "doç", "şek", "sek", "tab", "bkz", "vd", "örn", "sn", "mr", "mrs"]

def better_sentence_split(text: str) -> List[str]:
    tmp = text
    for a in TR_ABBR:
        tmp = re.sub(rf"\b{a}\.", f"{a}<DOT>", tmp, flags=re.IGNORECASE)
    sents = nltk.sent_tokenize(tmp)
    return [s.replace("<DOT>", ".") for s in sents]


# ----------------------------
# MODELS
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Cihaz: {device}")

SUM_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
print(f"[INFO] Özetleme modeli yükleniyor: {SUM_MODEL_NAME}")
tokenizer_sum = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME).to(device)

QA_MODEL_NAME = "savasy/bert-base-turkish-squad"
print(f"[INFO] QA modeli yükleniyor: {QA_MODEL_NAME}")
tokenizer_qa = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
model_qa = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME).to(device)

qa_pipeline = pipeline(
    "question-answering",
    model=model_qa,
    tokenizer=tokenizer_qa,
    device=0 if device == "cuda" else -1
)

print("[INFO] Modeller hazır.\n")


# ----------------------------
# PDF READ (layout korunur + header/footer temizliği)
# ----------------------------
def _page_lines(page_text: str) -> List[str]:
    lines = [ln.strip() for ln in (page_text or "").splitlines()]
    return [ln for ln in lines if ln]

def read_pdf_keep_layout(file_path: str) -> str:
    reader = PyPDF2.PdfReader(file_path)
    pages = []
    header_candidates = []
    footer_candidates = []

    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)

        lines = _page_lines(t)
        if len(lines) >= 2:
            header_candidates.extend(lines[:2])
            footer_candidates.extend(lines[-2:])

    from collections import Counter
    hc = Counter(header_candidates)
    fc = Counter(footer_candidates)
    n_pages = max(1, len(pages))

    header_noise = {k for k, v in hc.items() if v / n_pages >= 0.30 and len(k) >= 6}
    footer_noise = {k for k, v in fc.items() if v / n_pages >= 0.30 and len(k) >= 6}

    cleaned_pages = []
    for t in pages:
        lines = (t or "").splitlines()
        new_lines = []
        for ln in lines:
            s = ln.strip()
            if not s:
                new_lines.append("")
                continue
            if s in header_noise or s in footer_noise:
                continue
            new_lines.append(ln)
        cleaned_pages.append("\n".join(new_lines))

    return "\n\n".join(cleaned_pages)


# ----------------------------
# TEXT NORMALIZE (HATA DÜZELTİLDİ: tek normalize_text)
# ----------------------------
def fix_spaced_words(s: str) -> str:
    # "L u c k i n" -> "Luckin", "D erleme" -> "Derleme" gibi
    pattern = r"\b(?:[A-Za-zÇĞİÖŞÜçğıöşü]\s){2,}[A-Za-zÇĞİÖŞÜçğıöşü]\b"
    return re.sub(pattern, lambda m: m.group(0).replace(" ", ""), s)

def normalize_text(text: str) -> str:
    text = text.replace("-\n", "")              # satır sonu tire birleşimi
    text = re.sub(r"[ \t]+", " ", text)         # çoklu boşlukları azalt
    text = re.sub(r"\n{3,}", "\n\n", text)      # çoklu newline azalt
    text = fix_spaced_words(text)               # BOZUK KELİME BİRLEŞTİRME
    return text.strip()


# ----------------------------
# SECTION DETECTION (Türkçe odaklı)  ✅ SÖZLÜK DÜZGÜN KAPATILDI
# ----------------------------
SECTION_HEADINGS = {
    "abstract": [
        r"^\s*(özet|o\s*z\s*e\s*t|abstract)\s*[:\-]?\s*$",
        r"^\s*(öz)\s*[:\-]?\s*$",
    ],
    "introduction": [
        r"^\s*(giriş|g\s*i\s*r\s*i\s*ş|introduction)\s*$",
        r"^\s*\d+\s*[\.\)]\s*(giriş|introduction)\s*$",
    ],
    "methods": [
        r"^\s*(yöntem|yöntemler|metot|metodoloji|yöntembilim)\s*$",
        r"^\s*(materyal\s+ve\s+yöntem|gereç\s+ve\s+yöntem)\s*$",
        r"^\s*(materials\s+and\s+methods|methodology|methods)\s*$",
        r"^\s*\d+\s*[\.\)]\s*(yöntem|yöntemler|metodoloji|methods)\s*$",
    ],
    "results": [
        r"^\s*(bulgular|sonuçlar\s+ve\s+tartışma|tartışma|deneysel\s+sonuçlar)\s*$",
        r"^\s*(results|discussion)\s*$",
        r"^\s*\d+\s*[\.\)]\s*(bulgular|tartışma|results|discussion)\s*$",
    ],
    "conclusion": [
        r"^\s*(sonuç|sonuçlar|genel\s+sonuç|değerlendirme)\s*$",
        r"^\s*(conclusion|conclusions)\s*$",
        r"^\s*\d+\s*[\.\)]\s*(sonuç|sonuçlar|conclusion)\s*$",
    ],
    "references": [
        r"^\s*(kaynakça|kaynaklar|yararlanılan\s+kaynaklar|referanslar)\s*$",
        r"^\s*(references|bibliography)\s*$",
    ],
}

STOP_AFTER_REFS = [
    r"^\s*ek(\s+|\b)|^\s*appendix\b",
    r"^\s*teşekkür\b|^\s*acknowledg",
    r"^\s*yazar\s+katk(ı|i)lar(ı|i)|^\s*author\s+contrib",
]

def find_sections(text: str) -> Dict[str, str]:
    lines = text.splitlines()

    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) > 90:
            continue
        low = s.lower()

        for sec, patterns in SECTION_HEADINGS.items():
            for p in patterns:
                if re.match(p, low, flags=re.IGNORECASE):
                    hits.append((idx, sec))
                    break

    if not hits:
        return {}

    hits = sorted(list(set(hits)), key=lambda x: x[0])

    sections: Dict[str, str] = {}
    for i, (start_idx, sec) in enumerate(hits):
        end_idx = hits[i + 1][0] if i + 1 < len(hits) else len(lines)
        block = "\n".join(lines[start_idx:end_idx]).strip()
        block = re.sub(r"^.*\n", "", block, count=1).strip()  # başlık satırını çıkar
        if len(block) > 200:
            sections[sec] = block

    if "references" in sections:
        ref_lines = sections["references"].splitlines()
        cut_at = None
        for i, ln in enumerate(ref_lines):
            s = ln.strip().lower()
            for sp in STOP_AFTER_REFS:
                if re.match(sp, s, flags=re.IGNORECASE):
                    cut_at = i
                    break
            if cut_at is not None:
                break
        if cut_at is not None and cut_at > 10:
            sections["references"] = "\n".join(ref_lines[:cut_at]).strip()

    return sections


# ----------------------------
# CHUNKING + RETRIEVAL
# ----------------------------
def semantic_sliding_window(text: str, max_tokens=700, overlap=2) -> List[str]:
    sentences = better_sentence_split(text)
    chunks = []
    cur, cur_len = [], 0

    for sent in sentences:
        tok_len = len(tokenizer_sum.encode(sent, add_special_tokens=False))
        if cur_len + tok_len <= max_tokens:
            cur.append(sent)
            cur_len += tok_len
        else:
            if cur:
                chunks.append(" ".join(cur))
            keep = cur[-overlap:] if len(cur) > overlap else cur
            cur = keep + [sent]
            cur_len = sum(len(tokenizer_sum.encode(s, add_special_tokens=False)) for s in cur)

    if cur:
        chunks.append(" ".join(cur))
    return chunks

def retrieve_top_chunks(chunks: List[str], query: str, k=6) -> List[str]:
    if not chunks:
        return []

    if SKLEARN_OK:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
        X = vec.fit_transform(chunks + [query])
        sims = cosine_similarity(X[-1], X[:-1]).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [chunks[i] for i in top_idx]

    q = [w for w in query.lower().split() if len(w) > 3]
    scored = []
    for c in chunks:
        lc = c.lower()
        score = sum(1 for w in q if w in lc)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:k] if s > 0]
    return top if top else chunks[:k]


# ----------------------------
# SUMMARIZATION (hiyerarşik)
# ----------------------------
def summarize_once(text: str, max_new_tokens=320, min_new_tokens=140) -> str:
    inp_text = f"summarize: {text}"

    input_ids = tokenizer_sum(
        inp_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).input_ids.to(device)

    out_ids = model_sum.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        early_stopping=True,
    )
    return tokenizer_sum.decode(out_ids[0], skip_special_tokens=True).strip()

def hierarchical_summary(text: str, chunk_tokens=700) -> str:
    chunks = semantic_sliding_window(text, max_tokens=chunk_tokens, overlap=2)
    partials = []
    for ch in chunks:
        try:
            partials.append(summarize_once(ch, max_new_tokens=260, min_new_tokens=110))
        except Exception:
            partials.append(ch[:1200])

    merged = " ".join(partials).strip()
    if len(merged.split()) > 220:
        try:
            return summarize_once(merged, max_new_tokens=520, min_new_tokens=220)
        except Exception:
            return merged[:3000]
    return merged


# ----------------------------
# QA (retrieve -> QA)
# ----------------------------
def answer_question(full_text: str, question: str) -> str:
    chunks = semantic_sliding_window(full_text, max_tokens=420, overlap=2)
    top = retrieve_top_chunks(chunks, question, k=6)

    best_text = "Belirtilmemiş."
    best_score = -1.0

    for ch in top:
        try:
            r = qa_pipeline(question=question, context=ch)
            if r["score"] > best_score:
                a0, a1 = r["start"], r["end"]
                sent_start = ch.rfind(".", 0, a0) + 1
                sent_end = ch.find(".", a1)
                if sent_end == -1:
                    sent_end = len(ch)
                cand = ch[sent_start:sent_end].strip()
                best_text = cand if len(cand) > 12 else r.get("answer", "Belirtilmemiş.")
                best_score = r["score"]
        except Exception:
            pass

    return best_text


# ----------------------------
# REFERENCES PARSER
# ----------------------------
def extract_references_from_section(ref_section: str, max_items=30) -> List[str]:
    lines = [ln.strip() for ln in ref_section.splitlines() if ln.strip()]
    items = []
    cur = ""

    def is_new_item(line: str) -> bool:
        if re.match(r"^(\[\d+\]|\d+[\.\)])\s+", line):
            return True
        if re.match(r"^[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü\-’' ]{1,40}\(\d{4}\)", line):
            return True
        return False

    for ln in lines:
        if is_new_item(ln):
            if cur:
                items.append(cur.strip())
            cur = ln
        else:
            cur = (cur + " " + ln).strip() if cur else ln

        if len(items) >= max_items:
            break

    if cur and len(items) < max_items:
        items.append(cur.strip())

    items = [it for it in items if len(it) > 25]
    return items[:max_items]


# ----------------------------
# METHODS EXTRACTION (hibrit)
# ----------------------------
METHOD_REGEX = re.compile(
    r"\b("
    r"destek\s+vekt(ö|o)r\s+makin(esi|eleri)|svm|"
    r"k[- ]?en\s+yak(ı|i)n\s+kom(ş|s)u|knn|"
    r"karar\s+a(ğ|g)ac(ı|i)|decision\s+tree|"
    r"rasgele\s+orman|random\s+forest|"
    r"lojistik\s+regresyon|logistic\s+regression|"
    r"do(ğ|g)rusal\s+regresyon|linear\s+regression|"
    r"naive\s+bayes|"
    r"cnn|evri(ş|s)imsel\s+sinir\s+a(ğ|g)(ı|i)|"
    r"lstm|gru|transformer|bert|t5|mt5|"
    r"tf[- ]?idf|word2vec|fasttext|"
    r"pca|k[- ]?means|"
    r")\b",
    flags=re.IGNORECASE
)

def extract_methods(sections: Dict[str, str], full_text: str) -> Tuple[str, List[str]]:
    methods_text = sections.get("methods", "")

    if not methods_text or len(methods_text) < 300:
        chunks = semantic_sliding_window(full_text, max_tokens=650, overlap=2)
        top = retrieve_top_chunks(
            chunks,
            "yöntem metodoloji kullanılan yöntemler algoritma model veri seti deneysel kurulum",
            k=8
        )
        methods_text = "\n".join(top)

    methods_summary = hierarchical_summary(methods_text)

    found = {m.group(0).strip() for m in METHOD_REGEX.finditer(methods_text)}
    found2 = {m.group(0).strip() for m in METHOD_REGEX.finditer(methods_summary)}
    method_list = sorted(set(found | found2), key=lambda x: x.lower())

    return methods_summary, method_list


# ----------------------------
# TITLE HEURISTIC
# ----------------------------
def guess_title(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = []
    for ln in lines[:30]:
        if 8 <= len(ln) <= 120:
            if re.match(r"^\d+[\.\)]\s+", ln):
                continue
            if ln.lower() in ["özet", "giriş", "yöntem", "bulgular", "sonuç", "kaynakça"]:
                continue
            candidates.append(ln)

    if not candidates:
        return "Başlık bulunamadı"

    candidates = sorted(candidates, key=lambda s: (s.endswith("."), abs(len(s) - 70)))
    return candidates[0]


# ----------------------------
# REPORT DATACLASS
# ----------------------------
@dataclass
class PaperReport:
    pdf_path: str
    title: str
    purpose: str
    overall_summary: str
    methods_summary: str
    methods_list: List[str]
    results_summary: str
    conclusion_summary: str
    references: List[str]


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def build_report_for_pdf(pdf_path: str) -> PaperReport:
    raw = read_pdf_keep_layout(pdf_path)
    text = normalize_text(raw)

    title = guess_title(text)
    sections = find_sections(text)

    purpose = answer_question(text, "Çalışmanın temel amacı nedir?")

    base_parts = []
    if "abstract" in sections:
        base_parts.append(sections["abstract"])
    if "results" in sections:
        base_parts.append(sections["results"][:8000])
    if "conclusion" in sections:
        base_parts.append(sections["conclusion"][:6000])

    if base_parts:
        overall_base = "\n\n".join(base_parts)
    else:
        chunks = semantic_sliding_window(text, max_tokens=650, overlap=2)
        top = retrieve_top_chunks(
            chunks,
            "amaç hedef objective yöntem metodoloji bulgular sonuç katkı çıkarım",
            k=10
        )
        overall_base = "\n".join(top)

    overall_summary = hierarchical_summary(overall_base)

    methods_summary, methods_list = extract_methods(sections, text)

    if "results" in sections:
        results_summary = hierarchical_summary(sections["results"][:9000])
    else:
        chunks = semantic_sliding_window(text, max_tokens=650, overlap=2)
        top = retrieve_top_chunks(chunks, "bulgular sonuçlar deneysel sonuç performans karşılaştırma", k=8)
        results_summary = hierarchical_summary("\n".join(top))

    if "conclusion" in sections:
        conclusion_summary = hierarchical_summary(sections["conclusion"][:7000])
    else:
        conclusion_summary = answer_question(text, "Çalışmanın sonucu nedir?")

    references = []
    if "references" in sections:
        references = extract_references_from_section(sections["references"], max_items=30)

    return PaperReport(
        pdf_path=pdf_path,
        title=title,
        purpose=purpose,
        overall_summary=overall_summary,
        methods_summary=methods_summary,
        methods_list=methods_list,
        results_summary=results_summary,
        conclusion_summary=conclusion_summary,
        references=references
    )


def synthesize_across_papers(reports: List[PaperReport]) -> Dict:
    combined = []
    all_methods = []
    all_refs = []

    for r in reports:
        combined.append(
            f"MAKALE: {r.title}\nAMAÇ: {r.purpose}\nÖZET: {r.overall_summary}\nSONUÇ: {r.conclusion_summary}"
        )
        all_methods.extend(r.methods_list)
        all_refs.extend(r.references)

    combined_text = "\n\n".join(combined)
    meta_summary = hierarchical_summary(combined_text[:18000])

    norm = {}
    for m in all_methods:
        key = m.lower().strip()
        if key not in norm:
            norm[key] = m.strip()
    methods_merged = sorted(norm.values(), key=lambda x: x.lower())

    ref_norm = {}
    for ref in all_refs:
        key = re.sub(r"\s+", " ", ref).strip().lower()
        if key not in ref_norm:
            ref_norm[key] = ref.strip()
    refs_merged = list(ref_norm.values())[:50]

    return {
        "meta_summary": meta_summary,
        "common_methods": methods_merged[:40],
        "merged_references_sample": refs_merged
    }


# ----------------------------
# CONFIG + RUN
# ----------------------------
if __name__ == "__main__":
    PDF_FOLDER = ""  # örn: "pdfs"
    PDF_PATHS = [
        "makale2.pdf",
    ]

    pdf_files = []
    if PDF_FOLDER and os.path.isdir(PDF_FOLDER):
        for fn in os.listdir(PDF_FOLDER):
            if fn.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(PDF_FOLDER, fn))
        pdf_files.sort()
    else:
        pdf_files = [p for p in PDF_PATHS if os.path.exists(p)]

    if not pdf_files:
        print("[ERROR] PDF bulunamadı. PDF_FOLDER veya PDF_PATHS kontrol et.")
        raise SystemExit

    reports: List[PaperReport] = []
    for i, pdf in enumerate(pdf_files, 1):
        print(f"\n{'='*80}\n[{i}/{len(pdf_files)}] İşleniyor: {pdf}\n{'='*80}")
        rep = build_report_for_pdf(pdf)
        reports.append(rep)

        print("\n--- BAŞLIK ---")
        print(rep.title)

        print("\n--- AMAÇ ---")
        print(rep.purpose)

        print("\n--- GENEL ÖZET ---")
        print(rep.overall_summary)

        print("\n--- YÖNTEM (ÖZET) ---")
        print(rep.methods_summary)

        print("\n--- YÖNTEM (LİSTE) ---")
        print(", ".join(rep.methods_list) if rep.methods_list else "Yöntem adı yakalanamadı.")

        print("\n--- BULGULAR (ÖZET) ---")
        print(rep.results_summary)

        print("\n--- SONUÇ (ÖZET) ---")
        print(rep.conclusion_summary)

        print("\n--- KAYNAKÇA (İLK 30) ---")
        if rep.references:
            for j, r in enumerate(rep.references, 1):
                print(f"{j}. {r}")
        else:
            print("Kaynakça bölümü bulunamadı / parse edilemedi.")

    if len(reports) > 1:
        print(f"\n{'='*80}\n[HARMANLANMIŞ META RAPOR]\n{'='*80}")
        meta = synthesize_across_papers(reports)
        print("\n--- META ÖZET ---")
        print(meta["meta_summary"])

        print("\n--- ORTAK YÖNTEMLER (İLK 40) ---")
        print(", ".join(meta["common_methods"]) if meta["common_methods"] else "Yöntem bulunamadı.")

    out = {
        "device": device,
        "papers": [asdict(r) for r in reports],
    }
    if len(reports) > 1:
        out["synthesis"] = synthesize_across_papers(reports)

    out_path = "paper_reports.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] JSON çıktı kaydedildi: {out_path}")
