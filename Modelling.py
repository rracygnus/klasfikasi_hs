import os
import tensorflow as tf
import numpy as np
import pandas as pd
import optuna
import fasttext
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from optuna.samplers import GridSampler
import streamlit as st
import seaborn as sns
import hashlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter
import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Nonaktifkan GPU untuk konsistensi
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.set_visible_devices([], 'GPU')  # Nonaktifkan GPU sepenuhnya

# Konfigurasi TensorFlow deterministik untuk hasil yang stabil
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(SEED)  # Atur seed untuk semua komponen Keras

# Fungsi embedding kalimat
def embed_sentences(fasttext_model, sentences):
    return np.array([fasttext_model.get_sentence_vector(sentence) for sentence in sentences])

# Fungsi menghitung hash
def compute_hash(text):
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

# Fungsi membagi dataset tanpa overlap
def prevent_overlap(X_raw, y, seed=None):
    if len(X_raw) < 10:
        raise ValueError("Dataset terlalu kecil untuk proses split.")

    # Hitung hash unik
    hashes = [compute_hash(text) for text in X_raw]
    hash_to_index = {h: i for i, h in enumerate(hashes)}
    unique_indices = list(hash_to_index.values())

    X_raw = np.array(X_raw)[unique_indices]
    y = np.array(y)[unique_indices]

    # Cek minimal sampel per kelas
    if min(Counter(y).values()) < 2:
        raise ValueError("Setiap kelas harus memiliki setidaknya 2 sampel.")

    # Split 80% train, 20% validation/test
    train_indices, test_indices, y_train, y_test = train_test_split(
        np.arange(len(X_raw)), y, test_size=0.2, stratify=y, random_state=SEED
    )

    return (
        X_raw[train_indices], X_raw[test_indices],
        y_train, y_test
    )

# Fungsi reshaping untuk LSTM
def reshape_for_lstm(embedded_sentences):
    return embedded_sentences.reshape((embedded_sentences.shape[0], 1, embedded_sentences.shape[1]))

# Fungsi plotting metrik
def plot_metrics(history, title="Model Performance"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title(f"{title} - Accuracy")
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title(f"{title} - Loss")
    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# Fungsi plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    # Check for missing classes in predictions
    unique_classes = set(y_true).union(set(y_pred))
    if len(unique_classes) != len(classes):
        st.warning("Ada kelas yang hilang di hasil prediksi. Pastikan model mengenali semua kelas.")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)

# Cache untuk menyimpan hasil berdasarkan parameter
trial_cache = {}

# Fungsi untuk memproses trial
def objective_generic(trial, X_train, y_train, X_val, y_val, classes):
    # Saran nilai hyperparameter
    n_units = trial.suggest_categorical('n_units', [128, 256, 512])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.4, 0.7])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_categorical('epochs', [10, 30, 50])

    # Simpan parameter dalam bentuk hashable tuple
    param_tuple = (n_units, dropout_rate, learning_rate, batch_size, epochs)

    # Jika parameter ditemukan di cache
    if param_tuple in trial_cache:
        st.write(f"Hasil ditemukan di cache untuk parameter: {dict(zip(['n_units', 'dropout_rate', 'learning_rate', 'batch_size', 'epochs'], param_tuple))}")
        cached_result = trial_cache[param_tuple]

        # Tampilkan metrik dan plot dari cache
        st.write(f"### **Metrics (Cached):**")
        st.write(f"- Train Accuracy: {cached_result['train_accuracy']:.4f}")
        st.write(f"- Train Loss: {cached_result['train_loss']:.4f}")
        st.write(f"- Validation Accuracy: {cached_result['val_accuracy']:.4f}")
        st.write(f"- Validation Loss: {cached_result['val_loss']:.4f}")
        st.write(f"- Precision: {cached_result['precision']:.4f}")
        st.write(f"- Recall: {cached_result['recall']:.4f}")
        st.write(f"- F1-Score: {cached_result['f1_score']:.4f}")

        # Tampilkan plot
        plot_metrics(cached_result['history'])
        plot_confusion_matrix(cached_result['y_true'], cached_result['y_pred'], classes)

        return cached_result["val_accuracy"]

    st.write(f"Training dengan parameter baru: {dict(zip(['n_units', 'dropout_rate', 'learning_rate', 'batch_size', 'epochs'], param_tuple))}")

    # Create model
    model = Sequential([
        LSTM(n_units, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'),
        Dropout(dropout_rate),
        Dense(len(classes), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=batch_size, epochs=epochs, verbose=0,
                        shuffle=False, 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    # Evaluate metrics
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)

    # Predict on validation data
    y_val_pred = np.argmax(model.predict(X_val), axis=-1)

    # Calculate metrics for class HS
    target_class_index = list(classes).index("HS")
    precision, recall, f1 = calculate_metrics(y_val, y_val_pred, target_class_index)

    # Periksa apakah parameter baru lebih baik dari yang di-cache
    if (
        param_tuple not in trial_cache or 
        val_accuracy > trial_cache[param_tuple]["val_accuracy"] or
        (val_accuracy == trial_cache[param_tuple]["val_accuracy"] and f1 > trial_cache[param_tuple]["f1_score"])
    ):
        # Simpan result ke dalam cache
        trial_cache[param_tuple] = {
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "y_true": y_val,
            "y_pred": y_val_pred,
            "history": history
        }

    # Display metrics
    st.write(f"### **Metrics:**")
    st.write(f"- Train Accuracy: {train_accuracy:.4f}")
    st.write(f"- Train Loss: {train_loss:.4f}")
    st.write(f"- Validation Accuracy: {val_accuracy:.4f}")
    st.write(f"- Validation Loss: {val_loss:.4f}")
    st.write(f"- Precision: {precision:.4f}")
    st.write(f"- Recall: {recall:.4f}")
    st.write(f"- F1-Score: {f1:.4f}")

    # Display plots
    plot_metrics(history)
    plot_confusion_matrix(y_val, y_val_pred, classes)

    return val_accuracy


def calculate_metrics(y_true, y_pred, target_class_index):
    # Membuat confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Ambil TP, FP, dan FN untuk kelas target (HS)
    tp = cm[target_class_index, target_class_index]  # True Positive
    fp = cm[:, target_class_index].sum() - tp       # False Positive
    fn = cm[target_class_index, :].sum() - tp       # False Negative

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# Pencarian hyperparameter yang diperbaiki
def optimize_param(param_name, param_values, fixed_params, X_train, y_train, X_val, y_val, classes):
    search_space = {param_name: param_values, **fixed_params}
    study = optuna.create_study(direction='maximize', sampler=GridSampler(search_space, seed=SEED))
    study.optimize(lambda trial: objective_generic(trial, X_train, y_train, X_val, y_val, classes), 
                   n_trials=len(param_values))
    best_value = study.best_params[param_name]
    st.write(f"Best {param_name}: {best_value}")
    return best_value

# Proses modeling lengkap
def modeling_process(df):
    try:
        # Path model FastText
        fasttext_model_path = "model/cc.id.300.bin"
        if not os.path.exists(fasttext_model_path):
            st.error(f"FastText model not found at {fasttext_model_path}")
            return

        fasttext_model = fasttext.load_model(fasttext_model_path)

        # Filter minimal sampel per kelas
        min_samples_per_class = 2
        df = df[df['Label'].map(df['Label'].value_counts()) >= min_samples_per_class]
        if df.empty:
            st.error("Dataset terlalu kecil setelah filter minimal sample per kelas.")
            return

        # Encode label dan bagi dataset
        le = LabelEncoder()
        X_raw = df['Final_Text'].values
        y = le.fit_transform(df['Label'].values)

        # Memanggil fungsi prevent_overlap dengan rasio 80-20
        X_train_raw, X_val_raw, y_train, y_val = prevent_overlap(X_raw, y, seed=SEED)

        # Pastikan tidak ada overlap di dataset
        assert len(set(X_train_raw).intersection(set(X_val_raw))) == 0, "Overlap found between train and validation sets!"

        # Proses embedding FastText dan reshaping untuk LSTM
        X_train = reshape_for_lstm(embed_sentences(fasttext_model, X_train_raw))
        X_val = reshape_for_lstm(embed_sentences(fasttext_model, X_val_raw))


        st.write("### Searching Best Hyperparameters")

        # Pencarian hyperparameter bertahap menggunakan Optuna
        best_n_units = optimize_param('n_units', [128, 256, 512], {
            'dropout_rate': [0.1], 'learning_rate': [0.001], 'batch_size': [32], 'epochs': [10]
        }, X_train, y_train, X_val, y_val, le.classes_)

        best_dropout_rate = optimize_param('dropout_rate', [0.1, 0.4, 0.7], {
            'n_units': [best_n_units], 'learning_rate': [0.001], 'batch_size': [32], 'epochs': [10]
        }, X_train, y_train, X_val, y_val, le.classes_)

        best_learning_rate = optimize_param('learning_rate', [0.001, 0.01, 0.1], {
            'n_units': [best_n_units], 'dropout_rate': [best_dropout_rate], 'batch_size': [32], 'epochs': [10]
        }, X_train, y_train, X_val, y_val, le.classes_)

        best_batch_size = optimize_param('batch_size', [32, 64, 128], {
            'n_units': [best_n_units], 'dropout_rate': [best_dropout_rate], 
            'learning_rate': [best_learning_rate], 'epochs': [10]
        }, X_train, y_train, X_val, y_val, le.classes_)

        best_epochs = optimize_param('epochs', [10, 30, 50], {
            'n_units': [best_n_units], 'dropout_rate': [best_dropout_rate], 
            'learning_rate': [best_learning_rate], 'batch_size': [best_batch_size]
        }, X_train, y_train, X_val, y_val, le.classes_)

        # Melatih model dengan hyperparameter terbaik
        st.write("### Training Final Model with Best Parameters")
        final_model = Sequential([
            LSTM(best_n_units, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'),
            Dropout(best_dropout_rate),
            Dense(len(le.classes_), activation='softmax')
        ])

        final_model.compile(optimizer=Adam(learning_rate=best_learning_rate), 
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=best_batch_size, epochs=best_epochs, verbose=1,
            shuffle=False,  # Tambahkan shuffle=False
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )

        # Tombol untuk mengunduh model terbaik
        def download_best_model(model_path):
            try:
                with open(model_path, "rb") as file:
                    st.download_button(
                        label="Download Best Model",
                        data=file,
                        file_name="best_model4.h5",
                        mime="application/octet-stream"
                    )
            except FileNotFoundError: 
                st.error("Model file tidak ditemukan. Pastikan model telah disimpan.")

        # Menyimpan model
        model_path = "best_model4.h5"
        final_model.save(model_path)
        st.success(f"Model telah disimpan di {model_path}")

        # Tambahkan tombol download
        download_best_model(model_path)

        st.success("Modeling process completed.")
    except Exception as e:
        st.error(f"Error: {e}")
