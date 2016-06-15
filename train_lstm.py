from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.callbacks import Callback
import numpy as np
import random
import sys
import string
import heapq
import os
import time

#saving a list of losses over each batch during training
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))
history = LossHistory()

chars = string.letters + string.digits + ' .,?^\''
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 31

train_path = sys.argv[1]
test_path = sys.argv[2]

def read_data(path):
#   path = sys.argv[1]
    lines = []
    for line in open(path):
        if line == '\n': continue
        heapq.heappush(lines, (random.random(), line))

    print('length of lines: ', len(lines))

    prefix = '^'
    text = ''.join([prefix + line.strip() for _, line in lines])
    all = string.maketrans('', '')
    rem = all.translate(all, chars)
    text = text.translate(None, rem)
    print('corpus length:', len(text))

    width = (len(text) - maxlen) // step
    X = np.zeros((width, maxlen, len(chars)), dtype=np.bool)
    Y = np.zeros((width, maxlen, len(chars)), dtype=np.bool)
    
    print('width = ', width)

    for j in xrange(width):
        for t in xrange(maxlen):
            X[j, t, char_indices[text[step*j+t]]] = 1
            Y[j, t, char_indices[text[step*j+t+1]]] = 1

    return X, Y

# build the model: 2 stacked LSTM

tic = time.time()

print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.5))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributedDense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print('compile completed in',time.time()-tic)

if os.path.exists('model.weights'):
    print('Loading existing weights')
    model.load_weights('model.weights')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

hist_nll = [1200]

# train the model, output generated text after each iteration
for iteration in range(1, 500):
    print('Iteration', iteration)
    
    X, Y = read_data(train_path)
    X_val,Y_val = read_data(test_path)
    model.fit(X, Y, batch_size=128, nb_epoch=1, validation_data=(X,Y), callbacks=[history])
    print('the nll of valset is :', history.losses)
    print('the history: ', hist_nll)
    if hist_nll[iteration-1] > history.losses[0]:
        print('better result: ', history.losses[0])
        model.save_weights('model.weights', overwrite=True)
        hist_nll.append(history.losses[0])
    else:
        print('worse result: ', history.losses[0])
        print('stop...')
        break
'''
    for diversity in []: # [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        sentence = '^'
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(sentence)

        for iteration in range(100):
            x = np.zeros((1, len(sentence), len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0][-1]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence += next_char
            if len(sentence) == 50:
                # Cut off to make predictions faster
                sentence = sentence[-50:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    X, Y = read_data()
    history =  model.fit(X, Y, batch_size=128, nb_epoch=1)
    model.save_weights('model.weights', overwrite=True)
'''
