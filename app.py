import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

# Load your pre-trained model
model = keras.models.load_model('Blind_DetectionV1.keras')

# Class labels mapping
CLASS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Streamlit app configuration
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")

# App header
st.title("🩺 Diabetic Retinopathy Detection")
st.markdown("""
Upload an eye fundus image to predict the severity level of diabetic retinopathy.
The model can classify images into 5 categories:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR
""")

# File upload section
uploaded_file = st.file_uploader("Choose an eye fundus image...", 
                               type=["jpg", "jpeg", "png"])

# Prediction and display section
if uploaded_file is not None:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Convert BGR to RGB for display
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    
    # Make prediction
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_image, 
                caption="Uploaded Fundus Image", 
                width=300)
        
    with col2:
        st.subheader("📊 Prediction Results")
        st.markdown(f"""
        **Predicted Class:** {CLASS_LABELS[predicted_class]}  
        **Confidence:** {confidence:.2%}  
        **Class ID:** {predicted_class}
        """)
        
        # Show probability distribution
        st.bar_chart({
            "Class Probabilities": prediction[0]
        }, use_container_width=True)

st.markdown("---")
st.info("⚠️ Note: This is a diagnostic support tool and should not replace professional medical advice.")