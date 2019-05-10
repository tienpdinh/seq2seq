from __future__ import absolute_import
from __future__ import print_function

from utils.data_generators import generate_copy_task, one_hot_encode
from keras.layers import Activation, dot, concatenate, Input, LSTM, Dense, TimeDistributed
from keras.models import Model
from keras.optimizers import SGD, Adam
import numpy as np

class EncoderDecoderCopy:

    def __init__(self, latent_dim=256, num_samples=10000, length=20, max_val=30, reverse=False, reverse_half=False):
        X, y = generate_copy_task(length, num_samples, max_val, reverse, reverse_half)
        self.encoder_input_data = one_hot_encode(X, max_val)
        self.decoder_input_data = one_hot_encode(y, max_val)
        self.decoder_target_data = np.zeros_like(self.decoder_input_data)
        self.decoder_target_data[:,:-1,:] = self.decoder_input_data[:,1:,:]

        self.latent_dim = latent_dim
        self.max_val = max_val
        self.length = length
        self.model = self._init_model()

    def _init_model(self):
        encoder_inputs = Input(shape=(None, self.max_val))
        encoder = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        decoder_inputs = Input(shape=(None, self.max_val))
        decoder = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

        # Attention mechanism with dot based scoring
        attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)

        context = dot([attention, encoder_outputs], axes=[2, 1])
        decoder_combined_context = concatenate([context, decoder_outputs])

        output = TimeDistributed(Dense(self.max_val, activation='softmax'))(decoder_combined_context)
        
        model = Model([encoder_inputs, decoder_inputs], output)
        attention_layer = model.get_layer('attention')
        model_attention = Model(model.inputs, model.outputs + [attention_layer.output])
        return model

    def get_attention_model(self):
        attention_layer = self.model.get_layer('attention')
        model = Model(inputs=self.model.inputs,
                      outputs=self.model.outputs + [attention_layer.output])
        return model

    def train(self, lr=1e-2, batch_size=64, epochs=10, verbose=1):
        optimizer = Adam(lr)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        history = self.model.fit([self.encoder_input_data, self.decoder_input_data],
                                 self.decoder_target_data,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 verbose=verbose)
        return history

    def think(self, num_samples=5):
        Xs, ys = [], []
        encoder_test_data, _ = generate_copy_task(self.length, num_samples, self.max_val)
        encoder_test_data = one_hot_encode(encoder_test_data, self.max_val)
        for sample in range(num_samples):
            X, y = [], []
            cur_sample = encoder_test_data[sample]
            dummy_decoder_input = np.zeros_like(cur_sample)
            dummy_decoder_input[0] = cur_sample[0]
            for i in range(1, self.length):
                X.append(cur_sample[i].argmax())
            for i in range(1, self.length):
                output = self.model.predict([cur_sample[np.newaxis,:], dummy_decoder_input[np.newaxis,:]])
                dummy_decoder_input[i] = output[:,i-1]
                output = output.argmax(axis=2)
                y.append(output[:,i-1][0])
            Xs.append(X)
            ys.append(y)
        return (Xs, ys) if num_samples != 1 else (Xs[0], ys[0])
