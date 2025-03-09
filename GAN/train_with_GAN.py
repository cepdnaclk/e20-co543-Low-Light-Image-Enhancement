import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Define the Generator model
def create_generator(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    
    # Residual block
    residual = x
    x = layers.Conv2D(256, (3, 3), strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, residual])  # Add the residual connection
    
    # Decoder (Expanding Path)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    
    # Output Layer
    outputs = layers.Conv2DTranspose(3, (3, 3), padding='same', activation='sigmoid')(x)
    
    # Create Generator Model
    generator = models.Model(inputs, outputs)
    
    return generator


# Define the Discriminator model
def create_discriminator(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Conv Layers (with LeakyReLU activation)
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)  # Dropout for regularization
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)  # Dropout for regularization
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)  # Dropout for regularization
    
    # Flatten and Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Real or fake
    
    # Create Discriminator Model
    discriminator = models.Model(inputs, outputs)
    
    return discriminator


# Define the combined GAN model (generator + discriminator)
def create_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator during generator training

    z = layers.Input(shape=(256, 256, 3))  # Input image for the generator
    fake_image = generator(z)  # Generate fake image from generator
    validity = discriminator(fake_image)  # Check if the discriminator thinks it's real or fake

    gan = models.Model(z, validity)
    gan.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return gan


# Loss function for training
loss_fn = tf.keras.losses.BinaryCrossentropy()

def train_gan(generator, discriminator, gan, epochs, batch_size, low_light_images, normal_light_images):
    for epoch in range(epochs):
        # Select a batch of random low-light images
        idx = np.random.randint(0, low_light_images.shape[0], batch_size)
        low_imgs = low_light_images[idx]
        real_imgs = normal_light_images[idx]  # Corresponding normal-light images

        # Generate fake enhanced images from the generator
        fake_imgs = generator.predict(low_imgs)

        # Labels for training the discriminator
        real_labels = np.ones((batch_size, 1))  # Real images -> 1
        fake_labels = np.zeros((batch_size, 1))  # Fake images -> 0

        # Train the Discriminator: real vs. fake
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the Generator (GAN): fool the discriminator
        g_loss = gan.train_on_batch(low_imgs, real_labels)  # Wants D to predict real (1)

        # Print training progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss} | G Loss: {g_loss}")



# Function to save the generated images
def save_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 256, 256, 3))
    generated_images = generator.predict(noise)
    
    # Rescale images from [0, 1] to [0, 255]
    generated_images = 0.5 * generated_images + 0.5
    
    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"generated_images_epoch_{epoch}.png")
    plt.close()

# TODO:
# Path to dataset
LOW_LIGHT_PATH = "LOL_dataset/our485/low/"
NORMAL_LIGHT_PATH = "LOL_dataset/our485/high/"

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 256, 256

def load_images(low_light_path, normal_light_path):
    low_light_images = []
    normal_light_images = []
    
    file_names = os.listdir(low_light_path)  # Get all image filenames

    for file in file_names:
        # Read the low-light image
        low_img = cv2.imread(os.path.join(low_light_path, file))
        low_img = cv2.resize(low_img, (IMG_WIDTH, IMG_HEIGHT))
        low_img = low_img / 255.0  # Normalize

        # Read the corresponding normal-light image
        normal_img = cv2.imread(os.path.join(normal_light_path, file))
        normal_img = cv2.resize(normal_img, (IMG_WIDTH, IMG_HEIGHT))
        normal_img = normal_img / 255.0  # Normalize
        
        low_light_images.append(low_img)
        normal_light_images.append(normal_img)

    return np.array(low_light_images), np.array(normal_light_images)

# Load dataset
low_light_images, normal_light_images = load_images(LOW_LIGHT_PATH, NORMAL_LIGHT_PATH)

# Create the generator, discriminator, and GAN
generator = create_generator()
discriminator = create_discriminator()
gan = create_gan(generator, discriminator)

# Compile the discriminator and GAN
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the GAN
epochs = 100
batch_size = 32
train_gan(generator, discriminator, gan, epochs, batch_size, low_light_images,normal_light_images)

# Save the generator model
generator.save('generator_model.h5')

# Save the discriminator model
discriminator.save('discriminator_model.h5')

# Optionally, save the entire GAN model (generator + discriminator together)
gan.save('gan_model.h5')

