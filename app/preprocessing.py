"""
PREPROCESSING PIPELINE (inference)
Replikasi PERSIS dari preprocessing.py yang dipakai saat training,
supaya teks baru dari user diproses dengan cara yang sama persis
sebelum masuk ke model.
"""
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


def clean_text(text: str) -> str:
    text = str(text).lower()                                  # case folding
    text = re.sub(r"http\S+|www\S+", " ", text)                # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                   # hapus angka & simbol
    text = re.sub(r"\s+", " ", text).strip()                   # rapikan spasi
    return text


def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = stopword_remover.remove(text)                       # stopword removal
    text = stemmer.stem(text)                                  # stemming
    return text
