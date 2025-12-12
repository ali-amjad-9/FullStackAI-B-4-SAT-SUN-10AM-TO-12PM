"""
DCGAN (Deep Convolutional GAN) implementation using TensorFlow 2 / Keras.

Features:
- Generator and Discriminator implemented as Keras models
- Training loop using tf.GradientTape
- Saves generated sample images during training
- Uses MNIST by default (grayscale 28x28). Easy to switch to CIFAR-10.

Usage:
- Requires TensorFlow 2.x installed: `pip install tensorflow`
- Run the script: `python dcgan_tensorflow_image_generation.py`
- Check `generated_images/` for sample outputs and `models/` for saved weights.

Notes:
- This implementation is educational and easy to extend (e.g., change dataset, image size,
  add label conditioning, use Wasserstein loss, etc.).
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# -------------------- Configuration --------------------
BUFFER_SIZE = 60000
BATCH_SIZE = 256
IMG_ROWS = 28
IMG_COLS = 28
IMG_CHANNELS = 1  # 1 for MNIST, 3 for CIFAR-10
IMG_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
Z_DIM = 100  # Dimensionality of the noise vector
EPOCHS = 50
SAVE_EVERY = 5  # Save generated images every N epochs
GENERATOR_LR = 2e-4
DISCRIMINATOR_LR = 2e-4
BETA_1 = 0.5
OUTPUT_DIR = "generated_images"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Data Preparation --------------------
def load_mnist():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 127.5 - 1.0  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    return x_train

# Prepare tf.data.Dataset
x_train = load_mnist()
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# -------------------- Model Definitions --------------------
# Generator: noise -> image
def build_generator(z_dim=Z_DIM):
    model = keras.Sequential(name="generator")
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(z_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))  # 7x7x256
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(IMG_CHANNELS, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    # Output shape: 28x28x1
    return model

# Discriminator: image -> real/fake
def build_discriminator(img_shape=IMG_SHAPE):
    model = keras.Sequential(name="discriminator")
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # Logit output
    return model

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()

# -------------------- Losses & Optimizers --------------------
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss: real vs fake
def discriminator_loss(real_output, fake_output):
    real_labels = tf.ones_like(real_output)
    fake_labels = tf.zeros_like(fake_output)
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss: try to fool discriminator
def generator_loss(fake_output):
    fool_labels = tf.ones_like(fake_output)  # want discriminator to output 'real' for fakes
    return cross_entropy(fool_labels, fake_output)

generator_optimizer = keras.optimizers.Adam(learning_rate=GENERATOR_LR, beta_1=BETA_1)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LR, beta_1=BETA_1)

# -------------------- Checkpointing --------------------
checkpoint_prefix = os.path.join(MODEL_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# -------------------- Utility: Save Images --------------------
fixed_seed = tf.random.normal([16, Z_DIM])  # For consistent samples during training

def save_generated_images(model, epoch, test_input=fixed_seed):
    predictions = model(test_input, training=False)
    # Rescale from [-1,1] to [0,1]
    predictions = (predictions + 1.0) / 2.0

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        img = predictions[i, :, :, 0] if IMG_CHANNELS == 1 else predictions[i]
        plt.imshow(img, cmap='gray' if IMG_CHANNELS==1 else None)
        plt.axis('off')

    plt.suptitle(f'Epoch {epoch}')
    filename = os.path.join(OUTPUT_DIR, f'generated_epoch_{epoch:03d}.png')
    plt.savefig(filename)
    plt.close(fig)

# -------------------- Training Step --------------------
@tf.function
def train_step(images):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, Z_DIM])

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train generator
    noise = tf.random.normal([batch_size, Z_DIM])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss

# -------------------- Training Loop --------------------
def train(dataset, epochs):
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        start = time.time()
        gen_loss_avg = keras.metrics.Mean()
        disc_loss_avg = keras.metrics.Mean()

        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
            gen_loss_avg.update_state(g_loss)
            disc_loss_avg.update_state(d_loss)

        # Save samples and checkpoint periodically
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            save_generated_images(generator, epoch)
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch}, Gen Loss: {gen_loss_avg.result():.4f}, Disc Loss: {disc_loss_avg.result():.4f}, Time: {time.time()-start:.2f}s')

    # Save final models
    generator.save(os.path.join(MODEL_DIR, 'generator_final.h5'))
    discriminator.save(os.path.join(MODEL_DIR, 'discriminator_final.h5'))

# -------------------- Main --------------------
if __name__ == '__main__':
    # For reproducibility (optional)
    tf.random.set_seed(42)
    np.random.seed(42)

    train(dataset, EPOCHS)

    # Generate final samples
    save_generated_images(generator, 'final')

    print('Training complete. Generated images in', OUTPUT_DIR)
