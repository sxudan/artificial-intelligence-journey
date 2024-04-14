import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_generator(input_shape=(256, 256, 3)):
    # Encoder (downsampling)
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(512, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    
    # Bottleneck (no downsampling)
    x = Conv2D(512, (4, 4), padding='same', activation='relu')(x)
    
    # Decoder (upsampling)
    x = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    outputs = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)  # Output image
    
    return Model(inputs, outputs, name='Generator')

def build_discriminator(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(512, (4, 4), padding='same', activation='relu')(x)
    outputs = Conv2D(1, (4, 4), padding='same', activation='sigmoid')(x)  # Output a probability
    
    return Model(inputs, outputs, name='Discriminator')

# Create generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator (we won't compile the generator as it will be trained via the combined model)
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Display model architectures
generator.summary()
discriminator.summary()