import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_circles
from tqdm import tqdm
import matplotlib.pyplot as plt

class deep_perceptron:

    def __init__(self, X_train, y_train, X_test=None, y_test=None, learning_rate=.1, epochs=100, train_size=.8,
                 random_state=0, nb_neurons_by_layer=[32, 1]):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = X_train.shape[0]
        self.n_layers = len(nb_neurons_by_layer)
        self.n = [nb_neurons for nb_neurons in nb_neurons_by_layer]
        self.n.insert(0, X_train.shape[1])
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train,
                                                                                  train_size=train_size,
                                                                                  random_state=random_state)
        self.X_train = np.transpose(self.X_train)
        self.X_valid = np.transpose(self.X_valid)
        if (X_test is not None) and (y_test is not None):
            self.X_test = np.transpose(X_test)
            self.y_test = np.array(y_test)

    def initialization(self):
        W = [[np.random.randn(self.n[i + 1], self.n[i]), np.random.rand(self.n[i + 1], 1)]
             for i in range(self.n_layers)]
        parameters = {}
        i = 1
        for w, b in W:
            parameters['w' + str(i)] = w
            parameters['b' + str(i)] = b
            i += 1
        return parameters

    def forward_propagation(self, parameters, step="train"):
        if step == "train":
            A = self.X_train
        elif step == "test":
            A = self.X_test
        elif step == "valid":
            A = self.X_valid
        else:
            A = 0
            print("error in step variable")
        activations = {}
        for layer in range(1, self.n_layers + 1):
            Z = np.dot(parameters['w' + str(layer)], A) + parameters['b' + str(layer)]
            A = self.sigmoid(Z)
            activations['A' + str(layer)] = A
        return activations


    def backward_propagation(self, activations, parameters):
        activations['A0'] = self.X_train
        gradients = {}
        dZ = activations['A' + str(self.n_layers)] - self.y_train
        for layer in reversed(range(1, self.n_layers + 1)):
            gradients['dw' + str(layer)] = (1 / self.m) * dZ.dot(activations['A' + str(layer - 1)].T)
            gradients['db' + str(layer)] = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)
            dZ = np.dot(parameters['w' + str(layer)].T, dZ) * activations['A' + str(layer - 1)] * (1 - activations['A' + str(layer - 1)])
        """
        gradients['dw' + str(self.n_layers)] = (1 / self.m) * dZ.dot(activations["A" + str(self.n_layers - 1)].T)
        gradients['db' + str(self.n_layers)] = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)
        for layer in range(1, self.n_layers):
            index = self.n_layers - layer
            dZ = np.dot(parameters['w' + str(index + 1)].T, dZ) * activations['A' + str(index)] * (1 - activations['A' + str(index)])
            gradients['dw' + str(index)] = (1 / self.m) * dZ.dot(activations['A' + str(index - 1)].T)
            gradients['db' + str(index)] = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)
        """
        return gradients


    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def predict(self, activations):
        return 1 * (activations['A' + str(self.n_layers)] >= .5)

    def neg_log_loss(self, A):
        nll = (-1 / len(self.y_train)) * np.sum(self.y_train * np.log(A) + (1 - self.y_train) * np.log(1 - A))
        return nll

    def update(self, parameters, jacobian):
        for key, item in parameters.items():
            parameters[key] -= self.learning_rate * jacobian['d' + key]
        return parameters

    def fit(self, plot_learning_curves=True):
        parameters = self.initialization()
        best_parameters = parameters.copy()
        best_accuracy = 0
        loss_train = []
        loss_valid = []
        accuracy_train = []
        accuracy_valid = []
        for epoch in tqdm(range(self.epochs)):
            activations = self.forward_propagation(parameters, step="train")
            gradients = self.backward_propagation(activations, parameters)
            parameters = self.update(parameters, gradients)
            if epoch % 10 == 0:
                loss_train.append(log_loss(self.y_train, activations['A' + str(self.n_layers)].T))
                y_pred_train = self.predict(activations)
                accuracy_train.append(accuracy_score(self.y_train, y_pred_train.T))
                activations_valid = self.forward_propagation(parameters, step="valid")
                loss_valid.append(log_loss(self.y_valid, activations_valid['A' + str(self.n_layers)].T))
                y_pred_valid = self.predict(activations_valid)
                accuracy_valid.append(accuracy_score(self.y_valid, y_pred_valid.T))
                if accuracy_valid[-1] > best_accuracy:
                    best_accuracy = accuracy_valid[-1]
                    best_parameters = parameters.copy()

        if plot_learning_curves:
            N = [10 * i for i in range(len(loss_train))]
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.plot(N, loss_train, c='red', label='train')
            plt.plot(N, loss_valid, c='blue', label='valid')
            plt.legend()
            plt.title('Loss')

            plt.subplot(1, 2, 2)
            plt.plot(N, accuracy_train, c='red', label="train")
            plt.plot(N, accuracy_valid, c='blue', label="valid")
            plt.legend()
            plt.title('Accuracy')
            plt.show()

        return best_parameters

    def evaluation(self, parameters, step="test"):
        activations = self.forward_propagation(parameters, step=step)
        y_pred = self.predict(activations)
        if step == "test":
            print(classification_report(self.y_test, y_pred.T))
            print(accuracy_score(self.y_test, y_pred.T))
        elif step == "train":
            print(classification_report(self.y_train, y_pred.T))
        return 0


if __name__ == '__main__':
    X, y = make_circles(n_samples=1000, noise=.1, factor=.3, random_state=0)

    #X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    model = deep_perceptron(X_train, y_train, X_test, y_test, learning_rate=.05, epochs=4000,
                            nb_neurons_by_layer=[8, 1])
    parameters = model.fit()
    model.evaluation(parameters, step="test")

    index_true = np.where(y == 1)[0]
    index_false = np.where(y == 0)[0]
    plt.figure()
    plt.scatter(X[index_true,0], X[index_true,1], c='yellow')
    plt.scatter(X[index_false,0], X[index_false, 1], c='blue')
    plt.show()

