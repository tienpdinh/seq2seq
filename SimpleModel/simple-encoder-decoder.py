from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

# Hyperparams here
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

# Data generators


