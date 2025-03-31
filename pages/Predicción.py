import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Cargar el modelo desde el archivo
with open("Streamlit/pages/svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Título de la página
st.set_page_config(page_title="Clasificador de Dígitos Manuscritos", page_icon="🖊️", layout="centered")
st.title("🖊️ Clasificador de Dígitos Manuscritos")
st.write(
    """
    Dibuja un número del 0 al 9 en el recuadro a continuación y presiona 'Predict' para ver la predicción del modelo SVM. 
    También puedes cargar una imagen manuscrita para hacer la predicción.
    """
)

# Títulos con estilo
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Canvas de dibujo
st.subheader("🎨 Dibuja un número:")
drawing_canvas = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="gray",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Función de preprocesamiento
def preprocess_image(image):
    # Convertir a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    # Redimensionar a 8x8
    image_resized = cv2.resize(image_gray, (8, 8), interpolation=cv2.INTER_AREA)

    # Escalar valores entre 0 y 16
    image_scaled = image_resized / 255.0 * 16

    # Aplanar la imagen
    image_flattened = image_scaled.flatten().reshape(1, -1)

    # Aplicar el escalado
    image_scaled = scaler.transform(image_flattened)

    return image_scaled

# Función de predicción
def predict(image):
    return clf.predict(image)[0]

# Predicción desde el canvas
if st.button("Predecir 🎯"):
    if drawing_canvas.image_data is not None:
        img_array = np.array(drawing_canvas.image_data, dtype=np.uint8)
        img_processed = preprocess_image(img_array)
        prediction = predict(img_processed)
        st.subheader("Predicción")
        st.write(f"El modelo predice que el número es: **{prediction}**")

# Subir imagen
st.subheader("📸 O sube una imagen:")
archivo_subido = st.file_uploader("Sube una imagen manuscrita (JPG o PNG)", type=["jpg", "png"])

if archivo_subido is not None:
    image = Image.open(archivo_subido).convert("RGBA")
    st.image(image, caption='Imagen subida', use_container_width=True)
    image_np = np.array(image)
    img_processed = preprocess_image(image_np)
    prediction = predict(img_processed)
    st.subheader(f"✅ El modelo predice que el número es: **{prediction}**")

# Información adicional
st.write("Esta app usa OpenCV para procesar imágenes y Scikit-learn para predecir dígitos manuscritos.")
st.markdown("""
    ---
    Made with ❤️ by Rubén Sancho Jiménez(https://www.instagram.com/ruben_sanchoj/) | [GitHub Repository](https://github.com/RubenSanchoJimenez)
""", unsafe_allow_html=True)
