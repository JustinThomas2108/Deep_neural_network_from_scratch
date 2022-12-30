from Neural_Networks.Artificial_neuron import artificial_neuron
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    df = pd.read_csv('../Data/Diabetes/diabetes.csv')
    X = StandardScaler().fit_transform(df.drop('Outcome', axis=1))
    y = np.array(df.loc[:, 'Outcome'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = artificial_neuron(X_train, y_train, X_test, y_test, learning_rate=.1, epochs=1000)
    weights, bias = model.fit()
    model.evaluation(weights, bias)