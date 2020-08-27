# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 01:21:39 2020

@author: Sakshi
"""
import numpy as np

class LinearRegression():
    def __init__(self, X, y, alpha=.1, iterations=1500):

        self.alpha = alpha
        self.iterations = iterations
        self.m = len(y)
        self.n_features = np.size(X, 1)
        self.X = X
        self.y = y[:, np.newaxis]
        self.theta = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None

    def fit(self):
        (self.X, self.theta) = self.normalize_data()
        self.gradient_descent()
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    
    def normalize_data(self, X=None):
        if X is None:
            X = self.X
        mu = np.mean(X,0)
        sigma = np.std(X,0)
        X = (X-mu) / sigma
        #Add row of 1's to X
        X_zero=np.ones([X.shape[0],1], dtype='int')
        X = np.hstack((X_zero, X))
        theta = np.zeros((X.shape[1],1))
        return X, theta
    
    #minimize cost
    def gradient_descent(self):
        J_history = np.zeros((self.iterations,1))
        for i in range(self.iterations):
            predictions = np.dot(self.X, self.theta)
            error = predictions - self.y
            #take error transpose to multiply error by each value in column
            mul_error_x = np.dot(error.T,self.X)
            computed_theta = (self.alpha/self.m)*mul_error_x
            self.theta = self.theta - computed_theta.T
            J_history[i] = self.compute_cost()
        print('Least Cost', J_history[-1])
        return (J_history, self.theta)
    
    def compute_cost(self, X = None, y = None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        m = len(y)
        #prediction = h(x)* theta or X* Theta transpose
		#(1/2) x Mean Squared Error (MSE)
		#error = h(x) - y
        error = self.prediction_method(X) - y
        cost = (1/(2*m))*np.sum(error**2)
        return cost
    
    def prediction_method(self, X=None):
        if X is None:
            X = self.X
        return np.dot(X, self.theta)

    def predict(self, X):
        (X, theta) = self.normalize_data(X)
        return self.prediction_method(X)
    
    def get_theta(self):
        return self.theta
    


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.metrics import mean_squared_error as mse

dataset = load_boston()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)    

my_lr = LinearRegression(X_train, y_train)
my_lr.fit()
my_pred = my_lr.predict(X_test)

sklearn_regressor = SKLinearRegression().fit(X_train, y_train)

sklearn_pred = sklearn_regressor.predict(X_test)
sklearn_train_accuracy = sklearn_regressor.score(X_train, y_train)

sklearn_test_accuracy = sklearn_regressor.score(X_test, y_test)


print('testing cost:')
print('My LR',mse(y_test, my_pred, squared=False))
print('SK Learn', mse(y_test, sklearn_pred, squared=False))