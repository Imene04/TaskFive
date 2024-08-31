import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
from glob import glob

# Step 1: Load and Preprocess Data
# Assuming you have a directory structure like: dataset/train/<food_name>/*.jpg

data_dir = 'dataset/train'
categories = os.listdir(data_dir)

# Image preprocessing
img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of the data for validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 2: Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer for classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Step 4: Save the Model
model.save('food_recognition_model.h5')

# Step 5: Estimating Calories
# Let's assume you have a dictionary mapping food categories to average calories
# For simplicity, let's create a mock dictionary:
calorie_dict = {
    "apple": 52,
    "banana": 89,
    "burger": 295,
    "pizza": 266,
    # ... add all categories with average calorie values
}

def estimate_calories(food_category):
    return calorie_dict.get(food_category, "Unknown")

# Step 6: Prediction and Calorie Estimation
def predict_and_estimate(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array)
    predicted_class = categories[np.argmax(prediction)]
    
    estimated_calories = estimate_calories(predicted_class)
    
    return predicted_class, estimated_calories

# Example usage
image_path = 'path/to/your/image.jpg'
predicted_class, estimated_calories = predict_and_estimate(image_path)
print(f"Predicted Food Item: {predicted_class}")
print(f"Estimated Calories: {estimated_calories} kcal")
