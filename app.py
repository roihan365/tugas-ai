# Note: perlu run perintah "pip install streamlit tensorflow pillow" terlebih dahulu
# untuk run aplikasi, gunakan perintah: streamlit run app.py


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("CNN Kelompok 3")

# Load model
@st.cache_resource 
def load_model():
    # model = tf.keras.models.load_model("model.h5")
    model = tf.keras.models.load_model("final_cnn_tuned.h5")
    return model

model = load_model()

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    # Tombol prediksi
    if st.button("Prediksi"):
        st.write("Sedang memproses...")

        # Preprocessing gambar agar sesuai dengan input model
        img = image.resize((32, 32))           # ukuran input model
        img_array = np.array(img) / 255.0      # normalisasi 0–1
        img_array = np.expand_dims(img_array, axis=0)  # tambah dimensi batch

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.success(f"Hasil Prediksi: **{predicted_class}**")
        st.info(f"Tingkat akurasi: {confidence:.2f}")

