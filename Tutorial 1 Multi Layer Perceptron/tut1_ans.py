import numpy as np
import string
import pandas as pd
import nltk
import keras

from sklearn import random_projection
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras import metrics

stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------
def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens


def get_bagofwords(data, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param vocab_dict: a dict from words to indices, type: dict
    return a dense word matrix,
    '''
    data_matrix = np.zeros((len(data), len(vocab_dict)), dtype=float)
    for i, doc in enumerate(data):
        for word in doc:
            word_idx = vocab_dict.get(word, -1)
            if word_idx != -1:
                data_matrix[i, word_idx] += 1
    return data_matrix


def read_data(file_name, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    data_matrix = get_bagofwords(df['words'], vocab_dict)

    return df['id'], df['label']-1, data_matrix, vocab
# ----------------- End of Helper Functions-----------------


def load_data():
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv")

    # Load testing data
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", vocab)
    test_data_label = pd.read_csv("data/answer.csv")['label'] - 1

    print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(train_id_list))
    print("Test Set Size:", len(test_id_list))

    K = max(train_data_label)+1  # labels begin with 0

    # Data random projection
    rand_proj_transformer = random_projection.GaussianRandomProjection(n_components=2000)
    # YOUR CODE HERE
    train_data_matrix = rand_proj_transformer.fit_transform(train_data_matrix)
    test_data_matrix = rand_proj_transformer.transform(test_data_matrix)
    print("Training Set Shape:", train_data_matrix.shape)
    print("Testing Set Shape:", test_data_matrix.shape)

    # Converts a class vector to binary class matrix.
    # https://keras.io/utils/#to_categorical
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    test_data_label = keras.utils.to_categorical(test_data_label, num_classes=K)
    return train_data_matrix, train_data_label, test_data_matrix, test_data_label


if __name__ == '__main__':
    train_data_matrix, train_data_label, test_data_matrix, test_data_label = load_data()

    # Data shape
    N, V = train_data_matrix.shape
    K = train_data_label.shape[1]

    # Hyperparameters
    input_size = V
    hidden_size = 100
    output_size = K
    batch_size = 100
    dropout_rate = 0.5
    learning_rate = 0.1
    total_epoch = 10

    # New model
    model = Sequential()

    # first hidder layer with softmax activation and dropout
    model.add(Dense(hidden_size, activation='softmax', input_dim=input_size))
    model.add(Dropout(dropout_rate))

    # second hidden layer with softmax activation and dropout
    # YOUR CODE HERE
    model.add(Dense(hidden_size, activation='softmax'))
    model.add(Dropout(dropout_rate))

    # output layer
    # YOUR CODE HERE
    model.add(Dense(K, activation='softmax'))

    # SGD optimizer with momentum
    # YOUR CODE HERE
    optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # training
    model.fit(train_data_matrix, train_data_label, epochs=total_epoch, batch_size=batch_size)
    # testing
    train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
    test_score = model.evaluate(test_data_matrix, test_data_label, batch_size=batch_size)

    print('Training Loss: {}\n Training Accuracy: {}\n'
          'Testng Loss: {}\n Testing accuracy: {}'.format(
              train_score[0], train_score[1],
              test_score[0], test_score[1]))
