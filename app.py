import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Food Classifier", page_icon="🍔", layout="wide")

# === Cache Model ===
@st.cache_resource
def load_food_model(model_name):
    if model_name == "MobileNetV2":
        return load_model('mobilenetv2_food101_10classes.h5')
    elif model_name == "Model Lain":
        return load_model('model_lain.h5')  # ganti jika ada model lain
    return None

# === Daftar kelas ===
class_names = ['apple_pie', 'burger', 'donuts', 'french_fries', 'hot_dog',
               'pizza', 'ramen', 'sushi', 'tacos', 'steak']

# === Info makanan ===
food_info = {
    'apple_pie': {"desc": "Pai manis isi apel populer di Amerika.", "origin": "Eropa/AS", "calories": "300 kcal/100g"},
    'burger': {"desc": "Roti isi daging khas Amerika.", "origin": "AS", "calories": "295 kcal/100g"},
    'donuts': {"desc": "Kue manis berbentuk cincin.", "origin": "AS", "calories": "452 kcal/100g"},
    'french_fries': {"desc": "Kentang goreng gurih.", "origin": "Belgia/Prancis", "calories": "312 kcal/100g"},
    'hot_dog': {"desc": "Roti isi sosis.", "origin": "Jerman/AS", "calories": "290 kcal/100g"},
    'pizza': {"desc": "Roti pipih dengan topping.", "origin": "Italia", "calories": "266 kcal/100g"},
    'ramen': {"desc": "Mie kuah Jepang.", "origin": "Jepang", "calories": "436 kcal/100g"},
    'sushi': {"desc": "Nasi Jepang dengan ikan.", "origin": "Jepang", "calories": "130 kcal/100g"},
    'tacos': {"desc": "Tortilla isi daging khas Meksiko.", "origin": "Meksiko", "calories": "226 kcal/100g"},
    'steak': {"desc": "Daging sapi panggang.", "origin": "Eropa/AS", "calories": "271 kcal/100g"}
}

# === Session State untuk history ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Sidebar Navigasi ===
st.sidebar.title("🍴 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔍 Predict", "ℹ About"])

# ===========================
# === Halaman HOME ===
# ===========================
if page == "🏠 Home":
    st.title("🍔 Food Classifier dengan Deep Learning")
    st.image("gambar.jpeg", caption="Berbagai macam makanan")
    st.markdown("""
    Selamat datang di aplikasi **Food Classifier**!  
    Aplikasi ini menggunakan **model deep learning (CNN)** untuk mengenali jenis makanan.  
    
    **Fitur utama:**
    - Upload satu atau banyak gambar
    - Ambil foto langsung dari kamera
    - Visualisasi confidence
    - Info gizi & asal makanan
    - Download hasil prediksi sebagai PDF
    """)

# ===========================
# === Halaman PREDICT ===
# ===========================
elif page == "🔍 Predict":
    st.title("🔍 Prediksi Jenis Makanan")

    col1, col2 = st.columns([2, 1])
    with col1:
        model_option = st.selectbox("Pilih Model:", ["MobileNetV2", "Model Lain"])
    with col2:
        input_option = st.radio("Input Gambar:", ["Upload", "Kamera"])

    model = load_food_model(model_option)

    uploaded_files = []
    if input_option == "Upload":
        uploaded_files = st.file_uploader("Upload gambar (bisa lebih dari 1)", 
                                          type=["jpg", "jpeg", "png"], 
                                          accept_multiple_files=True)
    else:
        camera_image = st.camera_input("Ambil foto")
        if camera_image:
            uploaded_files = [camera_image]

    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption="Gambar yang diunggah", use_column_width=True)

            # Preprocessing
            img_tensor = tf.image.resize(np.array(img), (224, 224))
            img_tensor = img_tensor / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)

            # Prediksi
            prediction = model.predict(img_tensor)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(prediction) * 100

            # Simpan ke history
            st.session_state.history.append((predicted_class_name, confidence))

            with col2:
                st.subheader(f"Hasil: {predicted_class_name}")
                st.write(f"Confidence: **{confidence:.2f}%**")
                info = food_info.get(predicted_class_name, {})
                st.write(f"**Deskripsi**: {info.get('desc', '-')}")
                st.write(f"**Asal**: {info.get('origin', '-')}")
                st.write(f"**Kalori**: {info.get('calories', '-')}")

            # Visualisasi confidence
            st.write("Distribusi Probabilitas:")
            fig, ax = plt.subplots()
            ax.bar(class_names, prediction[0])
            ax.set_ylabel('Probabilitas')
            ax.set_xticklabels(class_names, rotation=45)
            st.pyplot(fig)

    # Download hasil PDF
    if st.session_state.history:
        if st.button("Download Hasil (PDF)"):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            elements = [Paragraph("Hasil Prediksi", styles['Title']), Spacer(1, 20)]
            for i, (food, conf) in enumerate(st.session_state.history):
                elements.append(Paragraph(f"{i+1}. {food} - {conf:.2f}%", styles['Normal']))
            doc.build(elements)
            buffer.seek(0)
            st.download_button("Download PDF", buffer, "hasil_prediksi.pdf", "application/pdf")

# ===========================
# === Halaman ABOUT ===
# ===========================
elif page == "ℹ About":
    st.title("ℹ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk **penugasan Learning Advance Week**.  
    **Hanya kelas berikut yang didukung:**
    """)

    food_classes = [
        'apple_pie', 'eggs_benedict', 'onion_rings', 'baby_back_ribs', 'escargots', 'oysters',
        'baklava', 'falafel', 'pad_thai', 'beef_carpaccio', 'filet_mignon', 'paella',
        'beef_tartare', 'fish_and_chips', 'pancakes', 'beet_salad', 'foie_gras', 'panna_cotta',
        'beignets', 'french_fries', 'peking_duck', 'bibimbap', 'french_onion_soup', 'pho',
        'bread_pudding', 'french_toast', 'pizza', 'breakfast_burrito', 'fried_calamari', 'pork_chop',
        'bruschetta', 'fried_rice', 'poutine', 'caesar_salad', 'frozen_yogurt', 'prime_rib',
        'cannoli', 'garlic_bread', 'pulled_pork_sandwich', 'caprese_salad', 'gnocchi', 'ramen',
        'carrot_cake', 'greek_salad', 'ravioli', 'ceviche', 'grilled_cheese_sandwich', 'red_velvet_cake',
        'cheesecake', 'grilled_salmon', 'risotto', 'cheese_plate', 'guacamole', 'samosa',
        'chicken_curry', 'gyoza', 'sashimi', 'chicken_quesadilla', 'hamburger', 'scallops',
        'chicken_wings', 'hot_and_sour_soup', 'seaweed_salad', 'chocolate_cake', 'hot_dog',
        'shrimp_and_grits', 'chocolate_mousse', 'huevos_rancheros', 'spaghetti_bolognese', 'churros',
        'hummus', 'spaghetti_carbonara', 'clam_chowder', 'ice_cream', 'spring_rolls', 'club_sandwich',
        'lasagna', 'steak', 'crab_cakes', 'lobster_bisque', 'strawberry_shortcake', 'creme_brulee',
        'lobster_roll_sandwich', 'sushi', 'croque_madame', 'macaroni_and_cheese', 'tacos', 'cup_cakes',
        'macarons', 'takoyaki', 'deviled_eggs', 'miso_soup', 'tiramisu', 'donuts', 'mussels',
        'tuna_tartare', 'dumplings', 'nachos', 'waffles', 'edamame', 'omelette'
    ]

    search = st.text_input("Cari makanan:", key="about_search")

    def normalize(text):
        return text.lower().replace("_", " ").strip()

    normalized_search = search.lower().replace("_", " ").strip()
    filtered_classes = [f for f in food_classes if normalized_search in normalize(f)] if search else food_classes

    if filtered_classes:
        st.write(f"Menampilkan {len(filtered_classes)} hasil:")
        cols = st.columns(3)
        for idx, food in enumerate(filtered_classes):
            with cols[idx % 3]:
                st.markdown(f"- **{food}**")
    else:
        st.warning("Tidak ditemukan.")
