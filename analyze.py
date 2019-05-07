from simpleencoderdecoder import EncoderDecoderSimple
from autoencoder import EncoderDecoderCopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

MAX_LENGTH = 50
MIN_LENGTH = 5

attend_acc, simple_acc = [], []

print('Sit back, and relax, it will take a while')
for length in tqdm(range(MIN_LENGTH, MAX_LENGTH+1)):
    attend_model = EncoderDecoderCopy(length=length)
    simple_model = EncoderDecoderSimple(length=length)
    attend_history = attend_model.train(epochs=1, verbose=0)
    simple_history = simple_model.train(epochs=1, verbose=0)
    attend_acc.append(max(attend_history.history['val_acc']))
    simple_acc.append(max(simple_history.history['val_acc']))

plt.clf()
plt.figure(figsize=(20,12))
plt.plot(range(MIN_LENGTH, MAX_LENGTH+1), attend_acc)
plt.plot(range(MIN_LENGTH, MAX_LENGTH+1), simple_acc)
plt.title('Acc v. Seq Length')
plt.ylabel('accuracy')
plt.xlabel('seq length')
plt.legend(['with attention', 'without attention'], loc='upper left')
plt.savefig('acc_vs_length.png')
