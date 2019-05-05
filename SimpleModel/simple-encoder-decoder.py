from __future__ import absolute_import
from __future__ import print_function

from data_generators import generate_copy_task, generate_single_task
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from random import randint

# Hyperparams here
batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
length = 8 # The length of each sequence
max_val = 5 # Maximum value in the sequence

# Data generators
X, y = generate_copy_task(length, num_samples, max_val)
encoder_input_data = np.array(X)
decoder_input_data = np.array(y)
decoder_target_data = np.array([((i[1:]).copy() + [[0]]) for i in y])

# Print data shapes
print('Encoder input data shape: {}'.format(encoder_input_data.shape))
print('Decoder input data shape: {}'.format(decoder_input_data.shape))
print('Decoder target data shape: {}'.format(decoder_target_data.shape))

# Model definitions
# ENCODER
encoder_inputs = Input(shape=(None, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# DECODER
decoder_inputs = Input(shape=(None, 1))
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(1, activation='relu')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

model.compile(optimizer='rmsprop', loss='mse')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

model.save('s2s.h5')

# Inference model
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_state_inputs,
    [decoder_outputs] + decoder_states)

# Inference
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, 1))
    stop_condition = False
    decoded_seq = ''
    for _ in range(length - 1):
        outputs, h, c = decoder_model.predict(
            [target_seq] + states_value)
        decoded_seq += str(int(round(outputs[0, 0, 0]))) + ' '
        target_seq[0, 0, 0] = outputs[0, 0, 0]
        states_value = [h, c]
    return decoded_seq

inf_input = np.array([[[randint(0, max_val)] for _ in range(length)]])
pred = decode_sequence(inf_input)
input_str = ''
inf_input_t = inf_input[:, 1:, :].tolist()[0]
for i in inf_input_t:
    input_str += str(i[0]) + ' '
print(input_str)
print(pred)
