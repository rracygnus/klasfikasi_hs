import numpy as np
import fasttext
from tensorflow.keras.models import load_model
import streamlit as st
from fuzzywuzzy import process
from Preprocessing import clean_text, case_folding, stemming, remove_stopwords

# Tambahkan dictionary untuk perbaikan typo dan leet
common_corrections = {
    'gila': 'gila',
    'g1l4': 'gila',
    'g1l@': 'gila',
    'g1l4h': 'gila',
    'g1l0': 'gila',
    'pengen': 'ingin',
    'pingin': 'ingin', 
    'sinting': 'sinting',
    's1nt1ng': 'sinting',
    'sntng': 'sinting',
    's1ntng': 'sinting',
    's1nt1n': 'sinting',
    'sakit jiwa': 'sakit jiwa',
    'sakitj1w@': 'sakit jiwa',
    's@kitj1w@': 'sakit jiwa',
    'sakitjw@': 'sakit jiwa',
    'sj@kit': 'sakit jiwa',
    'bipolar': 'bipolar',
    'b1p0l4r': 'bipolar',
    'bip0lar': 'bipolar',
    'b1pol4r': 'bipolar',
    'b1p0lar': 'bipolar',
    'penyakit jiwa': 'penyakit jiwa',
    'pnyk1tj1w@': 'penyakit jiwa',
    'pnyk1tjiw@': 'penyakit jiwa',
    'penyakitjw@': 'penyakit jiwa',
    'penyakitjiw@': 'penyakit jiwa',
    'kurang ajar': 'kurang ajar',
    'krngajr': 'kurang ajar',
    'kurang@jar': 'kurang ajar',
    'kurang@jr': 'kurang ajar',
    'krngaj@r': 'kurang ajar',
    'kurangajr': 'kurang ajar',
    'mental': 'mental',
    'm3nt4l': 'mental',
    'm3nt@l': 'mental',
    'm3nt@l': 'mental',
    'm3nt4l': 'mental',
    'mntal': 'mental',
    'goblok': 'goblok',
    'g0bl0k': 'goblok',
    'gbl0k': 'goblok',
    'g0blk': 'goblok',
    'bego': 'bego',
    'b3go': 'bego',
    'b3g0': 'bego',
    'b3g0h': 'bego',
    'bgo': 'bego',
    'bdoh' : 'bodoh',
    'bodo':'bodoh',
    'labil': 'labil',
    'l@b1l': 'labil',
    'l4b1l': 'labil',
    'l@b1l': 'labil',
    'l4b1l': 'labil',
    'gila babi': 'gila babi',
    'g1l@b4bi': 'gila babi',
    'g1l4babi': 'gila babi',
    'bacot': 'bacot',
    'b@c0t': 'bacot',
    'b4c0t': 'bacot',
    'otaknya rusak': 'otaknya rusak',
    '0t@knyrus@k': 'otaknya rusak',
    '0t@knyarusa': 'otaknya rusak',
    '0takrusak': 'otaknya rusak',
    'mabok': 'mabok',
    'm4b0k': 'mabok',
    'm4b0k': 'mabok',
    'mab0k': 'mabok',
    'm@b0k': 'mabok',
    'ga': 'tidak',
    'ga': 'jangan',
    'pantes': 'pantas',
    'cocok': 'pantas'
    ''
}

def correct_text(text, fasttext_model):
    """Koreksi typo menggunakan dictionary dan FastText."""
    words = text.split()
    corrected_words = []

    for word in words:
        # Cek dan perbaiki typo menggunakan dictionary perbaikan
        word = common_corrections.get(word, word)
        
        # Koreksi menggunakan FastText jika perlu
        try:
            fasttext_model.get_word_vector(word)  # Cek validitas kata dengan FastText
        except KeyError:
            # Gunakan fuzzy matching untuk kata yang tidak dikenali FastText
            similar_word = process.extractOne(word, common_corrections.keys())
            if similar_word and similar_word[1] >= 80:
                word = common_corrections.get(similar_word[0], word)
        
        corrected_words.append(word)
    
    return ' '.join(corrected_words)

def preprocess_text(text, fasttext_model):
    """Pipeline preprocessing yang sederhana."""
    text = clean_text(text)
    text = case_folding(text)
    text = correct_text(text, fasttext_model)  # Perbaiki typo dan leet speak
    tokens = stemming(text.split())
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)

def embed_text(text, fasttext_model):
    """Menyematkan teks ke dalam vektor menggunakan FastText."""
    vectors = [fasttext_model.get_word_vector(word) for word in text.split()]
    return np.array(vectors) if vectors else np.zeros((1, fasttext_model.get_dimension()))

def classification_process(user_input):
    """Proses klasifikasi teks menggunakan model FastText dan LSTM."""
    try:
        fasttext_model = fasttext.load_model("model/cc.id.300.bin")
        lstm_model = load_model("best_model2.h5")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    preprocessed_text = preprocess_text(user_input, fasttext_model)
    if not preprocessed_text:
        st.error("Preprocessed text is empty.")
        return

    embedding_sequence = embed_text(preprocessed_text, fasttext_model)
    embedding_reshaped = np.expand_dims(embedding_sequence, axis=0)

    try:
        prediction = lstm_model.predict(embedding_reshaped)
        result = 'Hate Speech' if prediction[0][0] >= 0.3 else 'Not Hate Speech'
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return

    # Atur threshold berdasarkan analisis ROC curve atau F1-score
    optimal_threshold = 0.3  # Threshold bisa diatur sesuai evaluasi
    result = 'Hate Speech' if prediction[0][0] >= optimal_threshold else 'Not Hate Speech'

    # Display hasil
    st.write("Preprocessed Text:", preprocessed_text)
    st.write("Prediction Value:", prediction[0][0])
    st.write(f"Hasil Prediksi: {result}")
