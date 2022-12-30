import pandas as pd
import numpy as np
from Neural_Networks.Multi_layer_perceptron import deep_perceptron
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':

    df_train = pd.read_csv('../Data/FakeNews/train.csv')
    X_train = np.array(df_train['text'])
    y_train = np.array(df_train['target'])
    cv = CountVectorizer()
    cv.fit(X_train)
    X_train = cv.transform(X_train).toarray()
    model = deep_perceptron(X_train, y_train, learning_rate=.1, epochs=200, train_size=.8, random_state=0,
                            nb_neurons_by_layer=[32, 1])
    weights, bias = model.fit()
