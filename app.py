import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import google.generativeai as genai
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import re

# Gemini AI
GENAI_API_KEY = "YOUR_KEY"  
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

model = tf.keras.models.load_model(r"D:\Coding 2.0\AI Projects\Skin Disease Detection\skin_disease_model_5000.h5")

# disease labels
label_map = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions"
}

# Function to get structured medical insights from Gemini AI
def get_medical_insights(disease):
    prompt = f"""
    Provide a structured medical overview for {disease}. 
    Format the response clearly under these sections:
    
    **Causes:** (explain the main causes of this disease)  
    **Symptoms:** (describe the common symptoms)  
    **Treatment:** (explain possible treatments)  

    Keep each section concise and informative.
    """
    
    response = gemini_model.generate_content(prompt)
    if response and hasattr(response, "text"):
        text = response.text

        # Extract sections using regex
        causes_match = re.search(r"\*\*Causes:\*\*\s*(.+?)(?=\*\*Symptoms:|\Z)", text, re.DOTALL)
        symptoms_match = re.search(r"\*\*Symptoms:\*\*\s*(.+?)(?=\*\*Treatment:|\Z)", text, re.DOTALL)
        treatment_match = re.search(r"\*\*Treatment:\*\*\s*(.+)", text, re.DOTALL)

        causes = causes_match.group(1).strip() if causes_match else "Not available."
        symptoms = symptoms_match.group(1).strip() if symptoms_match else "Not available."
        treatment = treatment_match.group(1).strip() if treatment_match else "Not available."

        return causes, symptoms, treatment
    else:
        return "Not available.", "Not available.", "Not available."

# Function to process image & predict disease
def process_and_predict(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize for model
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    disease_name = label_map[class_index]
    confidence = round(100 * np.max(prediction), 2)

    return disease_name, confidence

# Streamlit UI
st.title("AI-Powered Skin Disease Detection")
st.write("Upload an image or use your camera to detect skin diseases with AI-powered medical insights.")

# Image input options
option = st.radio("Choose an image source:", ("📸 Live Camera", "🖼 Upload Image"))

if option == "📸 Live Camera":
    camera_image = st.camera_input("Take a photo")

    if camera_image:
        image = Image.open(camera_image)
        disease, confidence = process_and_predict(image)
        st.image(image, caption="Captured Image", use_container_width=True)
        st.success(f"🩺 **Detected Disease:** {disease}")
        st.info(f"📊 **Confidence:** {confidence}%")

        # Get structured medical insights
        st.subheader("📖 Medical Insights")
        causes, symptoms, treatment = get_medical_insights(disease)

        st.markdown(f"### Causes\n{causes}")
        st.markdown(f"### Symptoms\n{symptoms}")
        st.markdown(f"### Treatment\n{treatment}")

elif option == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        disease, confidence = process_and_predict(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.success(f"🩺 **Detected Disease:** {disease}")
        st.info(f"📊 **Confidence:** {confidence}%")

        # Get structured medical insights
        st.subheader("📖 Medical Insights")
        causes, symptoms, treatment = get_medical_insights(disease)

        st.markdown(f"### Causes\n{causes}")
        st.markdown(f"### Symptoms\n{symptoms}")
        st.markdown(f"### Treatment\n{treatment}")
