import numpy as np


def my_train_test_split(x, y, test_size, random_state):
    if len(x) != len(y):
        raise ValueError("X Y не соответствуют по длине")

    total = len(x)
    total_test = int(total * test_size)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(total)

    test_ind = indices[:total_test]
    train_ind = indices[total_test:]

    x_train = x.iloc[train_ind]
    x_test = x.iloc[test_ind]

    y_train = y.iloc[train_ind]
    y_test = y.iloc[test_ind]

    return x_train, x_test, y_train, y_test

def my_scaler_fit(nums):
    mean = np.mean(nums, axis=0)
    std = np.std(nums, axis=0)

    if np.any(std == 0):
        raise ValueError("Есть признаки с нулевым стандартным отклонением")

    return mean, std


def my_scaler_transform(nums, mean, std):
    return (nums - mean) / std

class MyLinearRegression():
    def __init__(self, lr = 1e-3, epochs = 1000):
        self.lr = lr
        self.epochs = epochs        

        self.weights = None
        self.bias = 0
        self.loss_history = []

    def fit(self, x, y):

        x = np.asarray(x, dtype= float)
        y = np.asarray(y, dtype= float)

        n_samples, n_features = x.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _epoch in range(self.epochs):

            y_pred = x @ self.weights + self.bias

            error = y_pred - y

            dw = (2 / n_samples) * (x.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            y_pred = x @ self.weights + self.bias

            error = y_pred - y

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, x):

        x = np.asarray(x, dtype=float)

        return x @ self.weights + self.bias
