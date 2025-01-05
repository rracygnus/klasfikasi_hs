import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
import streamlit as st

# Inisialisasi stemmer dan stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))
stop_words.add("rt")  # Tambahkan 'rt' ke daftar stopwords

# Daftar kata penting untuk mempertahankan konteks
important_words = [
    'tidak', 'kurang', 'bukan', 'tak', 'tidak_sopan', 'kurang_ajar'  # Kata negasi
    'namun', 'tetapi', 'meski', 'walaupun',  # Kata penghubung
    'sangat', 'amat', 'paling',  # Kata penguat
    'buruk', 'jelek', 'hina', 'kasar', 'jahat',  # Kata evaluasi negatif
    'dong', 'nih', 'ya', 'aja' , 'jangan' # Kata gaul
]

# Fungsi membersihkan teks
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\brt\b', '', text)  # Hapus retweet
    text = re.sub(r'@\w+', '', text)  # Hapus mention
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'[^\w\s]', '', text)  # Hapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    leet_dict = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '9': 'g'}
    for leet_char, normal_char in leet_dict.items():
        text = text.replace(leet_char, normal_char)
    return text

# Fungsi case folding
def case_folding(text):
    return text.lower() if isinstance(text, str) else ""

# Fungsi tokenisasi dengan mempertahankan konteks kata penting
def tokenization_with_context(text):
    tokens = text.split()
    processed_tokens = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in important_words and i < len(tokens) - 1:
            combined_word = f"{token}_{tokens[i + 1]}"
            processed_tokens.append(combined_word)
            skip_next = True
        else:
            processed_tokens.append(token)
    return processed_tokens

# Fungsi stemming
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# Fungsi menghapus stopwords
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words or word in important_words]

# Pipeline preprocessing
def preprocessing_process(df):
    df['Tweet'] = df['Tweet'].fillna("")  # Isi nilai kosong

    # Proses preprocessing
    df['Cleaned_Text'] = df['Tweet'].apply(clean_text)
    df['Case_Folded'] = df['Cleaned_Text'].apply(case_folding)
    df['Tokenized'] = df['Case_Folded'].apply(tokenization_with_context)
    df['Stemmed'] = df['Tokenized'].apply(stemming)
    df['Without_Stopwords'] = df['Stemmed'].apply(remove_stopwords)
    df['Final_Text'] = df['Without_Stopwords'].apply(lambda x: ' '.join(x))

    # Tampilkan hasil preprocessing di Streamlit
    st.write("Data Setelah Preprocessing:")
    st.write(df[['Cleaned_Text', 'Case_Folded', 'Tokenized', 'Stemmed', 'Without_Stopwords', 'Final_Text']].head())

    # Buat file untuk diunduh
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Hasil Preprocessing", csv, "preprocessed_data.csv", "text/csv")
