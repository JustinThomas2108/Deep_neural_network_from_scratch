import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class artificial_neuron:

    def __init__(self, X_train, y_train, X_test, y_test, learning_rate=.1, epochs=100,
                 train_size=.8, random_state=0):
        self.X_train = X_train
        self.y_train = np.array(y_train).reshape((len(y_train), 1))
        self.X_test = X_test
        self.y_test = np.array(y_test).reshape((len(y_test), 1))
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train,
                                                                                  train_size=train_size,
                                                                                  random_state=random_state)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def initialization(self):
        w = np.random.randn(self.X_train.shape[1], 1)
        b = np.random.rand(1)
        return (w, b)

    def model(self, w, b, step="train"):
        if step == "train":
            X = self.X_train
        elif step == "test":
            X = self.X_test
        elif step == "valid":
            X = self.X_valid
        else:
            X = 0
            print("error in step variable")
        Z = np.matmul(X, w) + b
        A = self.sigmoid(Z)
        return A

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def predict(self, A):
        return 1 * (A >= .5)

    def neg_log_loss(self, A):
        nll = (-1 / len(self.y_train)) * np.sum(self.y_train * np.log(A) + (1 - self.y_train) * np.log(1 - A))
        return nll

    def gradients(self, A):
        dw = (1 / len(self.y_train)) * np.matmul(self.X_train.T, A - self.y_train)
        db = (1 / len(self.y_train)) * np.sum(A - self.y_train)
        return (dw, db)

    def update(self, w, b, dw, db):
        w -= self.learning_rate * dw
        b -= self.learning_rate * db
        return (w, b)

    def fit(self, history = True, plot_learning_curves = True):
        w, b = self.initialization()
        loss_train = []
        loss_valid = []
        accuracy_train = []
        accuracy_valid = []
        for epoch in range(self.epochs):
            A_train = self.model(w, b)
            y_pred_train = self.predict(A_train)
            A_valid = self.model(w, b, step="valid")
            y_pred_valid = self.predict(A_valid)
            loss_train.append(log_loss(self.y_train, A_train))
            loss_valid.append(log_loss(self.y_valid, A_valid))
            accuracy_train.append(accuracy_score(self.y_train, y_pred_train))
            accuracy_valid.append(accuracy_score(self.y_valid, y_pred_valid))
            if history:
                print("loss : ", loss_train[-1], ", train_accuracy : ", accuracy_train[-1], ", valid_accuracy : ", accuracy_valid[-1], ", epoch : ", epoch)
            dw, db = self.gradients(A_train)
            w, b = self.update(w, b, dw, db)

        if plot_learning_curves:
            plt.figure()
            plt.plot(loss_train, c='red')
            plt.plot(loss_valid, c='blue')
            plt.legend(["train", "valid"])
            plt.title('loss')

            plt.figure()
            plt.plot(accuracy_train, c='red')
            plt.plot(accuracy_valid, c='blue')
            plt.legend(["train", "valid"])
            plt.title('accuracy')

            plt.show()

        return (w, b)

    def evaluation(self, w, b, step="test"):
        A = self.model(w, b, step=step)
        y_pred = self.predict(A)
        if step == "test":
            print(classification_report(self.y_test, y_pred))
        elif step == "train":
            print(classification_report(self.y_train, y_pred))
        return 0


if __name__ == '__main__':
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = artificial_neuron(X_train, y_train, X_test, y_test, learning_rate=.1, epochs=500)
    weights, bias = model.fit()
    model.evaluation(weights, bias)
    index_true = np.where(y == 1)[0]
    index_false = np.where(y == 0)[0]
    x0 = np.linspace(-2, 2, 100)
    x1 = - weights[0]/weights[1] * x0 - bias
    plt.figure()
    plt.scatter(X[index_true,0], X[index_true,1], c='blue')
    plt.scatter(X[index_false, 0], X[index_false, 1], c='green')
    plt.plot(x0, x1, c='red')
    plt.show()


