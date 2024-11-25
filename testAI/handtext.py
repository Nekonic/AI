from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

# Visualize the processed image
plt.imshow(img[0], cmap='gray')
plt.show()

