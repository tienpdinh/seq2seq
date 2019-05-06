from __future__ import absolute_import
from __future__ import print_function

from utils.data_generators import generate_copy_task, one_hot_encode
from keras.layers import Activation, dot, concatenate, Input, LSTM, Dense, TimeDistributed
from keras.models import Model
from keras.optimizers import SGD, Adam
import numpy as np

lr = 1e-2 # Learning rate
batch_size = 64
epochs = 15
latent_dim = 256
num_samples = 10000
length = 30
max_val = 30

# Data generating
# Reverse sequence generating works but the order is a little bit off
# since the inference code is designed for forward prediction.
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
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

# Attention mechanism with dot based scoring
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax', name='attention')(attention)

context = dot([attention, encoder_outputs], axes=[2, 1])
decoder_combined_context = concatenate([context, decoder_outputs])

output = TimeDistributed(Dense(max_val, activation='softmax'))(decoder_combined_context)

model = Model([encoder_inputs, decoder_inputs], output)

model.summary()

optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Inference model
def think(num_samples=5):
    global length, max_val, model
    encoder_test_data, _ = generate_copy_task(length, num_samples, max_val)
    encoder_test_data = np.array(encoder_test_data)
    encoder_test_data = one_hot_encode(encoder_test_data, max_val)
    for sample in range(num_samples):
        cur_sample = encoder_test_data[sample]
        dummy_decoder_input = np.zeros_like(cur_sample)
        dummy_decoder_input[0] = cur_sample[0]
        print('Query seq:', end=' ')
        for i in range(1, length):
            print(cur_sample[i].argmax(), end=' ')
        print()
        print('Predi seq:', end=' ')
        for i in range(1, length):
            output = model.predict([cur_sample[np.newaxis,:], dummy_decoder_input[np.newaxis,:]])
            #print(output.shape)
            dummy_decoder_input[i] = output[:,i-1]
            output = output.argmax(axis=2)
            print(output[:,i-1][0], end=' ')
        print()

think()
