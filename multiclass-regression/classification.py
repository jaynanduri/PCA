from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class SoftMaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, t0=5, t1=50, num_classes=10, random_state=42, tol=1e-5, epochs=100, batch_size=50,
                 reg_constant=0.1):
        """
        Initialise SoftmaxClassifier object with the params.
        :param t0: used to compute learning rate
        :param t1: used to computed learning rate
        :param num_classes: total number of unique classes in y
        :param random_state: set the random state to generate consistent results
        :param tol: used for stopping iteration when w_old and w_new don't change more than tol
        :param epochs: maximum number of iterations for SGD
        :param reg_constant: value used to control the update of W
        :param batch_size: number of values used to update W in SGD
        """
        self.reg_constant = reg_constant
        np.random.seed(random_state)
        self.t0 = t0
        self.t1 = t1
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.tol = tol
        self.epochs = epochs
        self.W = None  # shape = (M x C) - M is number of features, C = Number of classes
        self.Z = None  # H_theta(X) = - X . W
        self.P = None  # P = Exp(Z_i)/sigma(k=0 to C) Exp(Z_ik)
        self.N = None  # number of data points

    def softmax(self):
        """
        Given Z, compute softmax to make it a probability.
        :param Z: Linear combination of inputs
        :return:
        """
        # for each row in Z
        self.P = np.exp(self.Z - np.max(self.Z, axis=1, keepdims=True))  # Prevent overflow
        self.P / np.sum(self.P, axis=1, keepdims=True)

    def predict(self, x):
        """
        Predict the class for a given set of points.
        :param x: data that is not used to train the model
        :return: class with the highest probability
        """
        P = self.predict_prob(x)
        return np.argmax(P, axis=1)

    def predict_prob(self, x):
        """
        Predicts prob of X belonging to class k.
        :param x: data point
        :return: return probability of x being in each class 1-C
        """
        Z = x.dot(self.W)
        P = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Prevent overflow
        return P / np.sum(P, axis=1, keepdims=True)

    def loss(self, X, one_hot_y):
        """
        Compute Cross Entropy loss for multiclass classification.
        :param one_hot_y: one hot encoded output
        :param X: input data points
        :return: neg cross entropy loss
        """
        P = self.predict_prob(X)
        return -np.mean(np.sum(one_hot_y * np.log(P + 1e-7), axis=1))

    def stochastic_gradient_descent(self, X_m, one_hot_y):
        """
        Perform a step to update weights using a batch random indices of x at a time
        :param X_m: number of data point used for each batch
        :param one_hot_y: one hot encoded output
        :return: gradients of delta W_k
        """
        self.Z = X_m.dot(self.W)
        P = np.exp(self.Z - np.max(self.Z, axis=1, keepdims=True))  # Prevent overflow
        P = P / np.sum(P, axis=1, keepdims=True)
        gradient = (1 / self.batch_size) * X_m.T.dot(one_hot_y - P) + 2 * self.reg_constant * self.W
        return gradient

    def learning_schedule(self, t):
        """
        Dynamically change lr to update gradients. Initially we take large steps and as the number of epochs increase,
        we slowly decrease lr and at the end steps are very small.
        :param t:
        :return:
        """
        return self.t0 / (t + self.t1)

    def compute_accuracy(self, X_m, Y_m):
        """
        Given X and Y, compute the accuracy of model predictions
        :param X_m: input data
        :param Y_m: target values
        :return: accuracy
        """
        y_pred = self.predict(X_m)
        return accuracy_score(np.argmax(Y_m, axis=1), y_pred)

    def train(self, X, y):
        """
            Training iteratively to maximise likelihood of predictions, we update the weights for the loss to converge to
            global minima.
            :param X: input data
            :param y: target values
            :return:
            """
        self.N, n_features = X.shape
        self.W = np.random.randn(n_features, self.num_classes) / np.sqrt(n_features)
        one_hot_y = OneHotEncoder(categories='auto', sparse=False).fit_transform(y.reshape(-1, 1))

        for epoch in tqdm(range(self.epochs)):
            for i in range(0, self.N, self.batch_size):
                random_indices = np.random.choice(self.N, self.batch_size, replace=False)
                X_batch = X[random_indices]
                y_batch = one_hot_y[random_indices]
                gradients = self.stochastic_gradient_descent(X_batch, y_batch)
                lr = self.learning_schedule(epoch * self.N + i)
                self.W -= lr * gradients

                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    loss = self.loss(X_batch, y_batch)
                    acc = self.compute_accuracy(X_batch, y_batch)
                    print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
