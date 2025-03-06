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

# Hide Streamlit branding (footer & GitHub logo)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stButton>button {
            width: 100%;
            font-size: 18px;
            padding: 10px;
        }
        .stImage img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }
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
    "Eggplant_Defect": [
        "🛑 Check for nutrient imbalances.",
        "💧 Maintain a consistent watering schedule.",
        "🌞 Ensure adequate sunlight exposure.",
        "🍂 Remove affected plant parts."
    ],
    "Eggplant_Flea Beetles": [
        "🔥 Clear plant debris to remove beetle eggs.",
        "🪴 Apply diatomaceous earth around plants.",
        "🐞 Introduce beneficial insects like nematodes.",
        "🌿 Use floating row covers to protect young plants."
    ],
    "Eggplant_Fresh": [
        "✅ The plant is healthy.",
        "🌱 Maintain regular watering and feeding.",
        "☀️ Ensure proper sun exposure.",
        "🍆 Harvest at the right time."
    ],
    "Eggplant_Leaf Wilt": [
        "💧 Ensure proper watering without overwatering.",
        "🌞 Provide good sun exposure.",
        "🦠 Check for fungal infections and treat accordingly.",
        "🐞 Control pests that may cause wilting."
    ],
    "Eggplant_Phytophthora Blight": [
        "🛑 Remove infected plants to prevent spreading.",
        "💧 Improve drainage to avoid waterlogging.",
        "🌾 Rotate crops each season.",
        "🦠 Apply fungicides like metalaxyl."
    ],
    "Eggplant_Powdery Mildew": [
        "💨 Increase airflow around plants.",
        "🦠 Use sulfur-based fungicides.",
        "🌞 Expose plants to more sunlight.",
        "🚫 Avoid overcrowding plants."
    ],
    "Eggplant_Tobacco Mosaic Virus": [
        "🛑 Remove infected plants immediately.",
        "👐 Disinfect tools after use.",
        "🐞 Control insect vectors like aphids.",
        "🦠 Grow virus-resistant varieties."
    ],
    "Okra_Alternaria Leaf Spot": [
        "🍃 Remove affected leaves.",
        "🦠 Apply fungicide if the infection spreads.",
        "💨 Space plants for better airflow.",
        "🌱 Rotate crops to reduce recurrence."
    ],
    "Okra_Cercospora Leaf Spot": [
        "🌿 Prune leaves for better ventilation.",
        "💧 Avoid wetting leaves while watering.",
        "🦠 Use copper-based fungicides if needed.",
        "🔥 Destroy infected plant debris."
    ],
    "Okra_Downy Mildew": [
        "🌞 Increase sunlight exposure.",
        "🦠 Apply organic fungicides if needed.",
        "💨 Improve air circulation.",
        "💧 Water early in the morning."
    ],
    "Okra_Healthy": [
        "✅ Your okra plant is in great condition!",
        "🌱 Maintain watering and fertilization.",
        "☀️ Ensure sufficient sun exposure.",
        "🍽️ Harvest regularly for better yield."
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

# Session state to manage UI flow
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file:
    image = Image.open(st.session_state.uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("🔄 Analyzing..."):
            result, confidence = predict_image_tflite(st.session_state.uploaded_file)
        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence}")

        # Show Disease Resolution if available
        st.subheader("🩺 Disease Resolution")
        for tip in disease_resolutions.get(result, ["No specific resolution available."]):
            st.write(f"- {tip}")

    # Button to go back and upload another image
    if st.button("🔄 Try Another Image"):
        st.session_state.uploaded_file = None
        st.rerun()
