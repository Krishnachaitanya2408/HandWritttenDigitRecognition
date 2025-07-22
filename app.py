import tensorflow as tf
import numpy as np
from PIL import Image
import gradio as gr

# -----------------------
# Load your model once
# -----------------------
model = tf.keras.models.load_model('digits_recognition_cnn1.h5')

# -----------------------
# Preprocess function
# -----------------------
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(img)
    # Invert colors if needed
    img_array = 255 - img_array
    # Normalize to 0-1
    img_array = img_array / 255.0
    # Add batch & channel dims
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# -----------------------
# Prediction function
# -----------------------
def predict_digit(image):
    img_prepared = preprocess_image(image)
    prediction = model.predict(img_prepared)
    predicted_label = np.argmax(prediction)
    return f"Predicted Digit: {predicted_label}"

# -----------------------
# Gradio Interface
# -----------------------
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Upload a digit image"),
    outputs="text",
    title="Handwritten Digit Recognizer",
    description="Draw or upload a digit (0-9) and the model will predict it."
)

# -----------------------
# Launch for local debug
# -----------------------
if __name__ == "__main__":
    iface.launch()
