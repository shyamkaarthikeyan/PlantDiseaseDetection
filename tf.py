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

# Custom Styling for Mobile & PC
st.markdown("""
    <style>
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
        .stSpinner {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Load the TFLite model (Optimized for performance)
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

# Disease Resolutions
disease_resolutions = {
    "Eggplant_Aphids": [
        "🌱 Spray neem oil or insecticidal soap to control aphids.",
        "🐞 Encourage natural predators like ladybugs.",
        "❌ Avoid excessive nitrogen fertilizers that attract aphids.",
        "🪴 Use reflective mulches to repel aphids."
    ],
    "Eggplant_Cercospora Leaf Spot": [
        "🌿 Remove infected leaves to prevent further spread.",
        "🧴 Apply copper-based fungicides for control.",
        "💨 Ensure proper spacing between plants for airflow.",
        "🚫 Avoid overhead watering to reduce moisture."
    ],
    "Tomato_Bacterial_spot": [
        "🌾 Use disease-free seeds and resistant varieties.",
        "🔬 Apply copper-based bactericides to slow spread.",
        "🤲 Avoid working with wet plants to prevent bacterial spread.",
        "🗑️ Remove and destroy infected plant debris."
    ],
    "Tomato_Early_blight": [
        "♻️ Rotate crops yearly to prevent fungal build-up.",
        "🛡️ Apply fungicides such as chlorothalonil or copper sprays.",
        "🌞 Ensure plants receive adequate sunlight and airflow.",
        "🚮 Remove and dispose of infected leaves immediately."
    ],
    "Tomato_Yellow_Leaf_Curl_Virus": [
        "🐛 Control whiteflies as they spread the virus.",
        "🍅 Use resistant tomato varieties when available.",
        "🕸️ Cover young plants with insect netting.",
        "🔥 Remove and destroy infected plants to prevent spread."
    ],
    "Okra_Downy Mildew": [
        "🌬️ Improve air circulation by proper spacing.",
        "🛑 Apply fungicides like mancozeb if needed.",
        "💦 Avoid overhead watering to reduce humidity.",
        "🌱 Use resistant varieties if available."
    ],
    "Okra_Leaf curly virus": [
        "🚮 Remove and destroy infected plants immediately.",
        "🐞 Control aphid populations as they spread the virus.",
        "🪴 Use reflective mulch to deter aphids.",
        "🚫 Avoid planting near virus-infected crops."
    ]
}

# Function to preprocess and predict
def predict_image_tflite(image_file):
    img = Image.open(image_file)
    img = img.convert("RGB")
    img = img.resize((160, 160))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get model input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class and confidence
    pred_index = np.argmax(output_data[0])  # Get highest probability class index
    pred_class = class_names[pred_index]  # Get actual class name
    pred_confidence = f"{round(100 * np.max(output_data[0]), 2)}%"

    return pred_class, pred_confidence

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("📸 Upload an image to classify plant diseases.")

# Session state to manage UI flow
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

if not st.session_state.image_uploaded:
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.session_state.image_uploaded = True
        st.session_state.uploaded_file = uploaded_file
        st.experimental_rerun()  # Refresh UI to show prediction button

else:
    image = Image.open(st.session_state.uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("🔄 Analyzing..."):
            result, confidence = predict_image_tflite(st.session_state.uploaded_file)
        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence}")

        # Show Disease Resolution if available
        if result in disease_resolutions:
            st.subheader("🩺 Disease Resolution")
            for tip in disease_resolutions[result]:
                st.write(f"- {tip}")

    # Button to go back and upload another image
    if st.button("🔄 Try Another Image"):
        st.session_state.image_uploaded = False
        st.session_state.uploaded_file = None
        st.experimental_rerun()  # Refresh UI to allow a new upload

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🌱 Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
