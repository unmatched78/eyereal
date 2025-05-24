# from flask import Flask, render_template, request, jsonify
# import cv2
# import numpy as np
# from tensorflow import keras

# app = Flask(__name__)

# # Load model
# model = keras.models.load_model('Blind_DetectionV1.keras')

# CLASS_LABELS = {
#     0: "No DR",
#     1: "Mild",
#     2: "Moderate",
#     3: "Severe",
#     4: "Proliferative DR"
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     # Read and preprocess image
#     img_bytes = file.read()
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     # Preprocessing
#     resized = cv2.resize(img, (224, 224))
#     normalized = resized / 255.0
#     input_img = np.expand_dims(normalized, axis=0)

#     # Prediction
#     prediction = model.predict(input_img)
#     predicted_class = int(np.argmax(prediction))
#     confidence = float(prediction[0][predicted_class])
    
#     return jsonify({
#         'class': CLASS_LABELS[predicted_class],
#         'confidence': confidence,
#         'class_id': predicted_class,
#         'probabilities': prediction[0].tolist()
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite  # Lightweight runtime

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path='retinopathy.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read and preprocess image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocessing (must match training)
        resized = cv2.resize(img, (224, 224))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0).astype(np.float32)

        # Set input and run inference
        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()
        
        # Get predictions
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_class])

        return jsonify({
            'class': CLASS_LABELS[predicted_class],
            'confidence': confidence,
            'class_id': predicted_class,
            'probabilities': prediction[0].tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)