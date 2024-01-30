# Pooling Layers: A pooling layer downsamples the feature maps to reduce dimensionality and increase translation invariance. This helps to aggregate local features across space.

from tensorflow import keras

# Define a max pooling layer with a window size of 2x2
pooling_layer = keras.layers.MaxPooling2D(pool_size=(2, 2))
