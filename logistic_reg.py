# import required modules
import numpy as np
from sklearn.datasets import load_breast_cancer
import pars


def GetTestData(data):
    test_data = []
    count_f = 0
    for i in range(3001, 4000):
        if count_f < 400:
            test_data.append(data[i][1:9])
            if data[i][8] == 'False':
                count_f += 1
        else:
            if data[i][8] == 'True':
                test_data.append(data[i][1:9])

    for i in range(len(test_data)):
        test_data[i][0:7] = list(map(lambda x: float(x), test_data[i][0:7]))

    return test_data

class LogisticRegression:
    def __init__(self, x, y):
        self.intercept = np.ones((x.shape[0], 1))
        self.x = np.concatenate((self.intercept, x), axis=1)
        self.weight = np.zeros(self.x.shape[1])
        self.y = y

    # Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))

    # method to calculate the Loss
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, lr, iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)

            loss = self.loss(sigma, self.y)

            dW = self.gradient_descent(self.x, sigma, self.y)

            # Updating the weights
            self.weight -= lr * dW

        return print('fitted successfully to data')

    # Method to predict the class label.
    def predict(self, x_new, treshold):
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.sigmoid(x_new, self.weight)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True:
                y_pred[i] = 1
            else:
                continue

        return y_pred

#Loading the data
data = pars.csv_reader('nasa.csv')
#data = load_breast_cancer()

#Preparing the data
import pandas as ps
import numpy as np

df = ps.read_csv('nasa.csv')

target = df['Hazardous']
data = df.drop('Hazardous', 1)

dfT = df[df.Hazardous == True]
dfT = dfT.head(500)
dfF = df[df.Hazardous == False]
dfF = dfF.head(500)

df = ps.concat([dfT, dfF])

x = data.to_numpy()
y = np.array(list(map(lambda x: 1 if np.bool(x) else 0, target)))


#x = np.array(pars.pars_data(data))
#y = np.array(GetTestData(data))

#x = data.data
#y = data.target

#creating the class Object
regressor = LogisticRegression(x,y)

#
regressor.fit(0.1 , 5000)


y_pred = regressor.predict(x,0.5)

print('accuracy -> {}'.format(sum(y_pred == y) / y.shape[0]))
