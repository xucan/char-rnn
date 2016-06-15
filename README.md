# char-rnn
Character-Lever Language Models
We will train RNN character-level language models. That is, we'll give the RNN a huge chunk of text and ask it to model the probability distribution of the
next character in the sequence given a sequence of previous characters. This will then allow us to generate new text one character at a time.

DATASET:  MovieTriples (Training_Shuffled_Dataset.txt)
Dictionary: 
    26 lowercase letters  and ", .  ? ^ ' " . total 32 chars
    maxlen = 40, step = 31. that means a window which length is 40, step is 31 will from the left to right of text. chars in a window is a sample.
Input:.      
    X = 1663557 * 40 * 32(one-hot)
    Y = 1663557 * 40 * 32
             
Model: 
    LSTM(512)
    Dropout(0.5)
    LSTM(512)
    Dropout(0.5)
    TimeDistributedDense(32)
    softmax

Objectiv:
    categorical crossentropy

Training result:
    epoch: 20
    training set: 1.2104
    val set: 1.1666

Generate:
    less the diversity, the generate is more equal argmax(p), the generate sentence will be the a selection of the training set
    bigger the diversity, the generate is more random
