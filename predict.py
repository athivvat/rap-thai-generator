import deepcut
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from keras.utils import np_utils


def get_words():
    words = []
    for file in glob.glob("./dataset/*.txt"):
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
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return normalized_input, network_output


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


def predict(words, seed_words):
    n_vocab = len(set(words))

    int_to_word = dict((number, word) for number, word in enumerate(sorted(set(words))))
    words_to_int = dict((word, number) for number, word in enumerate(sorted(set(words))))

    normalized_input, network_output = prepare_sequences(words, n_vocab)

    pattern = []

    for i in deepcut.tokenize(seed_words):
        result = words_to_int[i]
        pattern.append(result)

    print(pattern)

    prediction_output = []

    model = create_network(normalized_input, n_vocab)
    model.load_weights('/Users/80094/Workspace/labs/thai-rap-lyrics-generator/training_weights/thai-music-generator-99-0.1968.hdf5')

    for word_index in range(120):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_word[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def validate(vocabulary, seed_words):
    seed_words = deepcut.tokenize(seed_words)
    valid = True
    for w in seed_words:
        print(w, end="")
        if w in vocabulary:
            print(" ✓ in vocabulary")
        else:
            print(" ✗ NOT in vocabulary")
            valid = False
    return valid


if __name__ == "__main__":
    words = get_words()
    seed_words = input("Seed word: ")

    if validate(words, seed_words):
        generate = predict(words, seed_words)
        print(generate)
    else:
        print("error")

