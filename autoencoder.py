from __future__ import absolute_import
from __future__ import print_function

from utils.data_generators import generate_copy_task, one_hot_encode
from keras.layers import Activation, dot, concatenate, Input, LSTM, Dense, TimeDistributed
from keras.models import Model
from keras.optimizers import SGD, Adam
import numpy as np

lr = 1e-3 # Learning rate
batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000
length = 15
max_val = 10

# Data generating
X, y = generate_copy_task(length, num_samples, max_val)
encoder_input_data = one_hot_encode(np.array(X), max_val)
decoder_input_data = one_hot_encode(np.array(y), max_val)
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:,:-1,:] = decoder_input_data[:,1:,:]

# Print data shapes
print('Encoder input data shape: {}'.format(encoder_input_data.shape))
print('Decoder input data shape: {}'.format(decoder_input_data.shape))
print('Decoder target data shape: {}'.format(decoder_target_data.shape))

# ENCODER - DECODER
encoder_inputs = Input(shape=(None, max_val))
encoder = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, max_val))
decoder = LSTM(latent_dim, return_sequences=True)
decoder_outputs = decoder(decoder_inputs, initial_state=encoder_states)

# Attention mechanism with dot based scoring
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder_outputs], axes=[2, 1])
decoder_combined_context = concatenate([context, decoder_outputs])

output = TimeDistributed(Dense(max_val, activation='softmax'))(decoder_combined_context)

model = Model([encoder_inputs, decoder_inputs], output)

model.summary()

optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
