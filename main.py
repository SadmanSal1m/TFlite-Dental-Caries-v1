import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Load and preprocess images
data_dir = 'C:/Users/sadma/Downloads/archive(2)/Data caries/Data caries/caries orignal data set/done'
augmented_data_dir = 'C:/Users/sadma/Downloads/archive(2)/Data caries/Data caries/caries augmented data set/preview'

def load_images_from_folder(folder):
    images = []
    labels = []
    target_size = (224, 224)  # Specify the target size for resizing

    for filename in os.listdir(folder):
        img = tf.keras.preprocessing.image.load_img(os.path.join(folder, filename), target_size=target_size)
        if img is not None:
            # Convert the image to a NumPy array
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            # Normalize pixel values to [0, 1]
            img_array /= 255.0

            images.append(img_array)

            # Extract label information from the filename
            label = 1 if "caries" in filename else 0
            labels.append(label)

    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.float32)


# Load and preprocess your data
X_original, y_original = load_images_from_folder(data_dir)
X_augmented, y_augmented = load_images_from_folder(augmented_data_dir)

# Combine original and augmented data if needed
X_data = tf.concat([X_original, X_augmented], axis=0)
y_data = tf.concat([y_original, y_augmented], axis=0)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_data, y_data, epochs=10, validation_split=0.2)

# Convert the Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('caries_model.tflite', 'wb') as f:
    f.write(tflite_model)

import tensorflow as tf
import os

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_tensor_index = interpreter.get_input_details()[0]['index']
output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

# Function to preprocess an image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Function to predict from a folder of images
def predict_from_folder(folder):
    image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []
    detected_images = []
    for image_file in image_files:
        input_data = preprocess_image(image_file)
        interpreter.set_tensor(input_tensor_index, input_data)
        interpreter.invoke()
        prediction = output()[0][0]
        predictions.append(prediction)
        detected_images.append(os.path.basename(image_file) if prediction >= 0.5 else None)
    return detected_images, predictions

# Directory containing new images for prediction
new_images_folder = 'C:/Users/sadma/Downloads/archive(2)/Data caries/Data caries/caries augmented data set/preview'

# Make predictions on new images
detected_images, predictions = predict_from_folder(new_images_folder)

# Print predictions
for i, (image_name, prediction) in enumerate(zip(detected_images, predictions)):
    if image_name is not None:
        print(f"Image {i + 1} - {image_name}: {'Caries' if prediction >= 0.5 else 'No Caries'} (Confidence: {prediction:.2f})")

# Calculate accuracy
num_correct_predictions = sum(1 for prediction in predictions if prediction >= 0.5)
accuracy = num_correct_predictions / len(predictions) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
