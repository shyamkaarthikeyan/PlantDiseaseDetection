import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

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

# Define extended disease resolutions
disease_resolutions = {
    "Eggplant_Aphids": "Use neem oil or insecticidal soap. Introduce ladybugs as natural predators. Avoid excessive nitrogen fertilizers. Wash leaves with water. Remove heavily infested leaves. Ensure healthy soil to boost plant immunity.",
    "Eggplant_Cercospora Leaf Spot": "Remove affected leaves. Apply copper-based fungicides. Avoid overhead watering. Improve air circulation. Use resistant varieties. Maintain crop rotation to prevent reinfection.",
    "Eggplant_Defect": "Monitor for pests or nutrient deficiencies. Ensure regular watering and balanced fertilization. Check for physical damage. Avoid handling plants roughly. Prune dead leaves regularly. Use organic compost for better growth.",
    "Eggplant_Flea Beetles": "Apply diatomaceous earth or neem oil. Keep garden weed-free. Use row covers for young plants. Rotate crops yearly. Introduce natural predators. Use sticky traps to monitor flea beetle activity.",
    "Eggplant_Fresh": "No disease detected. Keep monitoring for early symptoms. Ensure optimal sunlight exposure. Water consistently but avoid overwatering. Apply organic mulch for soil health. Regularly inspect leaves for changes.",
    "Eggplant_Fresh_Leaf": "Healthy leaf. Maintain good plant care and monitoring. Fertilize with balanced nutrients. Prune overcrowded areas for better airflow. Inspect for hidden pests. Keep soil well-drained and aerated.",
    "Eggplant_Leaf Wilt": "Ensure proper watering. Check for soil-borne diseases. Rotate crops yearly. Improve soil drainage. Use mulch to retain moisture. Apply compost tea to enhance plant strength.",
    "Eggplant_Phytophthora Blight": "Improve soil drainage. Remove infected plants. Use resistant varieties. Apply fungicides early. Avoid water stagnation. Plant in well-ventilated areas.",
    "Eggplant_Powdery Mildew": "Apply sulfur or potassium bicarbonate sprays. Improve air circulation. Reduce plant crowding. Water plants in the morning. Remove severely infected leaves. Use milk spray as a natural remedy.",
    "Eggplant_Tobacco Mosaic Virus": "Remove infected plants. Control aphids and insects. Avoid handling infected plants. Disinfect tools regularly. Use virus-resistant varieties. Rotate crops to prevent soil contamination.",
    "Okra_Alternaria Leaf Spot": "Apply fungicides. Remove infected leaves. Avoid overhead watering. Rotate crops. Improve soil quality. Ensure good spacing between plants.",
    "Okra_Cercospora Leaf Spot": "Improve air circulation. Use copper-based fungicides. Space plants properly. Apply organic fertilizers. Remove dead leaves. Keep foliage dry.",
    "Okra_Downy Mildew": "Apply copper-based fungicides. Avoid wet leaves. Space plants for airflow. Remove infected plants immediately. Improve soil drainage. Avoid planting in shaded areas.",
    "Okra_Healthy": "No disease detected. Keep monitoring regularly. Maintain consistent watering. Fertilize as needed. Check leaves for early signs of infection. Prune if necessary.",
    "Okra_Leaf curly virus": "Remove and destroy infected plants. Control whiteflies. Use virus-resistant varieties. Disinfect tools. Avoid working in fields when plants are wet. Rotate crops every season.",
    "Okra_Phyllosticta leaf spot": "Apply fungicides. Remove affected leaves. Ensure proper plant spacing. Reduce humidity around plants. Avoid excessive nitrogen fertilizers. Improve air circulation.",
    "Tomato_Bacterial_spot": "Apply copper-based sprays. Avoid working in wet fields. Remove infected plants. Space plants properly. Keep foliage dry. Use disease-free seeds.",
    "Tomato_Early_blight": "Remove affected leaves. Apply fungicides. Rotate crops yearly. Use organic mulch. Avoid excessive nitrogen fertilizers. Space plants to improve airflow.",
    "Tomato_healthy": "No disease detected. Keep an eye for early symptoms. Provide adequate nutrients. Water consistently. Prune regularly. Keep soil healthy with compost.",
    "Tomato_Late_blight": "Destroy infected plants. Apply fungicides. Avoid wet leaves. Rotate crops yearly. Improve drainage. Use resistant tomato varieties.",
    "Tomato_Leaf_Mold": "Ensure good ventilation. Apply fungicides. Avoid high humidity. Water at soil level. Remove affected leaves. Keep greenhouse doors open for airflow.",
    "Tomato_mosaic_virus": "Remove infected plants. Disinfect tools. Control insect vectors. Avoid smoking near plants. Use virus-resistant seeds. Keep weeds away from crops.",
    "Tomato_Septoria_leaf_spot": "Apply fungicides. Remove infected leaves. Improve airflow. Use drip irrigation. Space plants well. Rotate crops yearly.",
    "Tomato_Spider_mites Two-spotted_spider_mite": "Spray with neem oil or insecticidal soap. Increase humidity. Introduce predatory mites. Keep plants well-watered. Remove heavily infested leaves. Avoid dusty conditions.",
    "Tomato_Target_Spot": "Use resistant varieties. Apply fungicides early. Remove infected leaves. Space plants properly. Improve ventilation. Avoid overhead watering.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies. Remove infected plants. Use resistant varieties. Avoid planting near infected crops. Disinfect tools. Keep fields weed-free."
}

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("📸 Upload an image to classify plant diseases.")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            result, confidence = predict_image_tflite(uploaded_file)
        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence}")
        st.warning(f"🩺 Resolution: {disease_resolutions[result]}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🌱 Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
