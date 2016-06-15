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

chars = string.letters + string.digits + ' .,?^\''
print('total chars:', len(chars))
maxlen = 40
step = 31
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


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

iteration = 1
while True:
    print('Iteration', iteration)
    wait = raw_input("please press enter")
    for diversity in [0.2,0.5,1.0,1.2]:
        print()
        print('----- diversity:', diversity)

        sentence = '^'
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(sentence)

        for iteration in range(200):
            x = np.zeros((1, len(sentence), len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0][-1]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence += next_char
            if len(sentence) == 100:
                # Cut off to make predictions faster
                sentence = sentence[-50:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
