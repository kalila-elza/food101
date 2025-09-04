import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# === Muat Model ===
# Cache model untuk mempercepat aplikasi
@st.cache_resource
def load_food_model():
    model = load_model('mobilenetv2_food101_10classes.h5')
    return model

model = load_food_model()

# === Konfigurasi Aplikasi Streamlit ===
st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Pengenalan Makanan")
st.markdown("Unggah gambar makanan Anda dan biarkan AI kami mengidentifikasinya!")

# Daftar kelas yang sesuai dengan model Anda
# Pastikan urutan ini benar sesuai saat melatih model
class_names = [
    'apple_pie', 'burger', 'donuts', 'french_fries', 'hot_dog',
    'pizza', 'ramen', 'sushi', 'tacos', 'steak'
]

# === Unggah Gambar dari Pengguna ===
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diunggah.', use_column_width=True)
    st.write("")
    st.write("Menganalisis...")
    
    # === Pra-pemrosesan Gambar ===
    # Ubah gambar ke format yang bisa diproses model
    img_tensor = tf.image.resize(img, (224, 224)) # Ukuran input untuk MobileNetV2
    img_tensor = img_tensor / 255.0  # Normalisasi
    img_tensor = np.expand_dims(img_tensor, axis=0) # Tambah dimensi batch

    # === Lakukan Prediksi ===
    prediction = model.predict(img_tensor)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100

    # === Tampilkan Hasil ===
    st.write("---")
    st.subheader("Hasil Prediksi")
    st.write(f"Ini adalah **{predicted_class_name}** dengan keyakinan **{confidence:.2f}%**.")