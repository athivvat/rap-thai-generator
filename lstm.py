import deepcut
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def get_words():
    words = []
    for file in glob.glob("//dataset/*.txt"):
        with open(file) as f:
            lines = f.read().splitlines()

        for line in lines:
            tokenz = deepcut.tokenize(line)
            [words.append(i) for i in tokenz if i.strip()]

    return words


def prepare_sequences(words, n_vocab):
    sequence_length = 8

    words_dict = dict((word, number) for number, word in enumerate(sorted(set(words))))

    network_input = []
    network_output = []

    for i in range(0, len(words) - sequence_length, 1):
        sequence_in = words[i:i + sequence_length]
        sequence_out = words[i + sequence_length]
        network_input.append([words_dict[word] for word in sequence_in])
        network_output.append(words_dict[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(normalized_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(normalized_input.shape[1], normalized_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model


def train(model, normalized_input, network_output):
    filepath = '//training_weights/thai-music-generator-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]

    model.fit(normalized_input, network_output, epochs=100, batch_size=64, callbacks=callbacks_list)


if __name__ == "__main__":
    words = get_words()
    n_vocab = len(set(words))

    # prepare network sequences
    network_input, network_output = prepare_sequences(words, n_vocab)

    # create LSTM model
    model = create_network(network_input, n_vocab)
    model.summary()

    # train network
    train(model, network_input, network_output)


