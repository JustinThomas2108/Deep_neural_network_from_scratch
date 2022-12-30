from Neural_Networks.Multi_layer_perceptron import deep_perceptron
import numpy as np


if __name__ == '__main__':
    X_train = np.load('../Data/Cat_vs_dog/X_train.npy').reshape((800, 64 * 64)) / 255
    X_valid = np.load('../Data/Cat_vs_dog/X_valid.npy').reshape((200, 64 * 64)) / 255
    X_test = np.load('../Data/Cat_vs_dog/X_test.npy').reshape((200, 64 * 64)) / 255
    y_train = np.load('../Data/Cat_vs_dog/y_train.npy').reshape((800,))
    y_valid = np.load('../Data/Cat_vs_dog/y_valid.npy').reshape((200,))
    y_test = np.load('../Data/Cat_vs_dog/y_test.npy').reshape((200,))

    print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)

    model = deep_perceptron(X_train, y_train, X_test, y_test, learning_rate=.1, epochs=400, train_size=.8,
                            random_state=0, nb_neurons_by_layer=[32, 64, 1])
    parameters = model.fit()
    model.evaluation(parameters, step="test")