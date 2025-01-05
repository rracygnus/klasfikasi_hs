import streamlit as st
from Preprocessing import preprocessing_process
from Modelling import modeling_process
from Classification import classification_process

st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Preprocessing", "Modelling", "Klasifikasi"])

if page == "Preprocessing":
    st.title("Preprocessing Dataset")
    st.write("Unggah file .csv yang akan dipreproses.")

    uploaded_file = st.file_uploader("Upload .csv file", type="csv")
    
    if uploaded_file:
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', sep=';')  # Ganti 'sep' jika diperlukan
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
        else:
            st.write("Kolom pada dataset:", df.columns)

            if df.empty:
                st.error("File yang diunggah kosong. Harap unggah file CSV yang valid.")
            else:
                missing_columns = [col for col in ['Tweet', 'Label'] if col not in df.columns]
                if missing_columns:
                    st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
                    st.info("Pastikan file memiliki kolom 'Tweet' untuk teks dan 'Label' untuk target.")
                else:
                    if st.button("Proses Preprocessing"):
                        preprocessing_process(df)

elif page == "Modelling":
    st.title("LSTM Modeling")
    st.write("Unggah file dataset yang sudah dipreproses untuk memulai proses modeling.")

    uploaded_file = st.file_uploader("Upload Preprocessed Dataset", type=["csv"])
    
    if uploaded_file:
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error membaca file: {e}")
        else:
            required_columns = ['Final_Text', 'Label']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
            else:
                if st.button("Proses Modelling"):
                    modeling_process(df)

elif page == "Klasifikasi":
    st.title("Hate Speech Classification")
    st.write("Masukkan teks yang ingin diklasifikasikan sebagai 'Hate Speech' atau 'Not Hate Speech'.")

    user_input = st.text_area("Input Teks:", "")

    if st.button("Klasifikasi"):
        if user_input:
            classification_process(user_input)
        else:
            st.write("Masukkan teks untuk diklasifikasikan.")
