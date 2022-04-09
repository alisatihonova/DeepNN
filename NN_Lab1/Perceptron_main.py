# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import Perceptron


df = pd.read_csv('data_err.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи


# нейронная сеть, которую мы создали перцептроном
# веса сгенерированы случайно
NN = Perceptron(inputSize, hiddenSizes, outputSize)

# для обучения передаем обучающую выборку, количество итераций и eta какую-нибудь
NN.train(X, y, eta = 0.01)

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
out, hidden_predict = NN.predict(X)

sum(out-y.reshape(-1, 1))
