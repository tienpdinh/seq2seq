from utils.data_generators import generate_single_task, one_hot_encode
from keras.layers import Activation, Input, LSTM, Dense, TimeDistributed, dot, concatenate
from keras.models import Model
from keras.optimizers import SGD, Adam
import numpy as np

class AutoEncoderSingle:

    def __init__(self, latent_dim, num_samples, length, max_val, return_index=0):
        X, y = generate_single_task(length, num_samples, max_val, return_index)
        self.encoder_input_data = one_hot_encode(X, max_val)
        self.decoder_target_data = one_hot_encode(y, max_val)
        self.decoder_input_data = np.zeros_like(self.decoder_target_data)

        self.latent_dim = latent_dim
        self.max_val = max_val
        self.length = length
        self.model = self._init_model()

    def _init_model(self):
        encoder_inputs = Input(shape=(None, self.max_val))
        encoder = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(1, self.max_val))
        decoder = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

        # Attention mechanism
        attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)

        context = dot([attention, encoder_outputs], axes=[2, 1])
        decoder_combined_context = concatenate([context, decoder_outputs])

        output = TimeDistributed(Dense(self.max_val, activation='softmax'))(decoder_combined_context)

        model = Model([encoder_inputs, decoder_inputs], output)
        return model

    def train(self, lr=1e-2, batch_size=64, epochs=10, verbose=1):
        optimizer = Adam(lr)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit([self.encoder_input_data, self.decoder_input_data],
                                 self.decoder_target_data,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 verbose=verbose)
        return history

    def think(self, num_samples=5):
        Xs, ys = [], []
        encoder_test_data, _ = generate_single_task(self.length, num_samples, self.max_val)
        encoder_test_data = one_hot_encode(encoder_test_data, self.max_val)
        for sample in range(num_samples):
            X, y = [], []
            cur_sample = encoder_test_data[sample]
            dummy_decoder_input = np.zeros((1, self.max_val))
            for i in range(self.length):
                X.append(cur_sample[i].argmax())
            output = self.model.predict([cur_sample[np.newaxis,:], dummy_decoder_input[np.newaxis,:]])
            output = output.argmax(axis=2)
            y.append(output[:,0][0])
            Xs.append(X)
            ys.append(y)
        return (Xs, ys) if num_samples != 1 else (Xs[0], ys[0])
