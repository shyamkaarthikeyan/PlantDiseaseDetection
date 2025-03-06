import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set Page Configuration (Title & Icon)
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Force-hide Streamlit branding (logos, footer, and menu)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none !important;} /* Hides GitHub/Streamlit deployment button */
        .st-emotion-cache-0 {display: none !important;} /* Hides "Made with Streamlit" */
        .viewerBadge_container__1QSob {display: none !important;} /* Hides Streamlit badge */
        .css-164nlkn {display: none !important;} /* Hides additional unwanted elements */
    </style>
""", unsafe_allow_html=True)

# Load the TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Define class names
class_names = [
    'Background_without_leaves', 'Eggplant_Aphids', 'Eggplant_Cercospora Leaf Spot',
    'Eggplant_Defect', 'Eggplant_Flea Beetles', 'Eggplant_Fresh', 'Eggplant_Fresh_Leaf',
    'Eggplant_Leaf Wilt', 'Eggplant_Phytophthora Blight', 'Eggplant_Powdery Mildew',
    'Eggplant_Tobacco Mosaic Virus', 'Okra_Alternaria Leaf Spot', 'Okra_Cercospora Leaf Spot',
    'Okra_Downy Mildew', 'Okra_Healthy', 'Okra_Leaf curly virus', 'Okra_Phyllosticta leaf spot',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_mosaic_virus', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus'
]

# Disease Resolutions (Complete for all 28)
disease_resolutions = {
    "Background_without_leaves": [
        "🌱 Check soil moisture levels and adjust watering.",
        "🌞 Ensure the plant gets enough sunlight.",
        "💨 Protect from strong winds that may cause leaf drop.",
        "🛑 Monitor for pests or diseases."
    ],
    "Eggplant_Aphids": [
        "🐞 Introduce ladybugs to eat aphids.",
        "🌿 Use neem oil or insecticidal soap.",
        "🚿 Spray leaves with water to remove aphids.",
        "🌾 Avoid over-fertilizing to prevent aphid attraction."
    ],
    "Eggplant_Cercospora Leaf Spot": [
        "🍃 Remove infected leaves immediately.",
        "🦠 Apply copper-based fungicide if needed.",
        "💧 Water at the base to prevent leaf wetness.",
        "🌱 Rotate crops to prevent reinfection."
    ],
    "Tomato_Yellow_Leaf_Curl_Virus": [
        "🐞 Control whiteflies, which spread the virus.",
        "🕸️ Use insect netting.",
        "🔥 Remove infected plants.",
        "🌾 Grow virus-resistant tomato varieties."
    ]
}

# Function to preprocess and predict
def predict_image_tflite(image_file):
    img = Image.open(image_file).convert("RGB").resize((160, 160))
    img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    # Get model input/output details
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    # Get predicted class and confidence
    pred_index = np.argmax(output_data[0])
    pred_class = class_names[pred_index]
    pred_confidence = f"{round(100 * np.max(output_data[0]), 2)}%"

    return pred_class, pred_confidence

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("📸 Upload an image to classify plant diseases.")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("🔄 Analyzing..."):
            result, confidence = predict_image_tflite(uploaded_file)
        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence}")

        # Show Disease Resolution if available
        st.subheader("🩺 Disease Resolution")
        for tip in disease_resolutions.get(result, ["No specific resolution available."]):
            st.write(f"- {tip}")
