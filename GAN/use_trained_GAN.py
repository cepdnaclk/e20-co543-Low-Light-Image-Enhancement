from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the generator model
generator = load_model('generator_model.h5')

# Load the discriminator model
discriminator = load_model('discriminator_model.h5')

# Optionally, load the entire GAN model (generator + discriminator together)
gan_model = load_model('gan_model.h5')



def enhance_image(image_path, generator):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Generate the enhanced image
    enhanced_image = generator.predict(image)[0]  # Remove batch dimension
    enhanced_image = (enhanced_image * 255).astype(np.uint8)  # Convert to 8-bit format
    
    # Display the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(image_path)[:, :, ::-1])  # Original image
    plt.title("Original Low-Light Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image)
    plt.title("Enhanced Image")
    plt.axis("off")
    
    plt.show()

    return enhanced_image
# TODO:
# Test the generator on a new low-light image
enhanced_img = enhance_image("low_light_sample.jpg", generator)
cv2.imwrite("enhanced_output.jpg", enhanced_img)  # Save the output
