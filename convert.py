import tensorflow as tf

# Verify TF installation
print("TensorFlow version:", tf.__version__)
print("Keras available?", hasattr(tf, "keras"))

# Load model
model = tf.keras.models.load_model('Blind_DetectionV1.keras')

# Convert to TFLite WITHOUT tflite-runtime
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Converted model size: {len(tflite_model)/1024:.1f} KB")