import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions
predictions = model.predict(x_test)

# Visualize some predictions
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(f"Predicted: {predictions[i].argmax()}")
plt.show()




# Load your image
image_path = 'data/my.png'  # Replace with your image path
img = Image.open(image_path).convert('L')  # Convert to grayscale

# Resize to 28x28 pixels
img = img.resize((28, 28))

# Invert colors if necessary (MNIST is white on black)
img = np.invert(img)

# Normalize pixel values
img = np.array(img) / 255.0

# Reshape to match model input
img = img.reshape(1, 28, 28)

# Predict the digit
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f'Predicted Digit: {predicted_digit}')