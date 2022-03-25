# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# загружаем и подготавливаем данные
df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

# количество входных сигналов равно количеству признаков
# задаем число нейронов скрытого (А) слоя 
hiddenSizes = 10 
# количество выходных сигнало в равно количеству классов задачи 
outputSize = 1 if len(y.shape) else y.shape[1]  
inputSize = X.shape[1]  


# создаем матрицу весов скрытого слоя
Win = np.zeros((1+inputSize,hiddenSizes)) 
# пороги w0 задаем случайными числами
Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes))) 
# остальные веса  задаем случайно -1, 0 или 1 
Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes))) 

#Wout = np.zeros((1+hiddenSizes,outputSize))

# случайно инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
    
# функция прямого прохода (предсказания) 
# тут описывается перцептрон Розенблата
def predict(Xp):
    # выходы первого слоя = входные сигналы * веса первого слоя
    hidden_predict = np.where((np.dot(Xp, Win[1:,:]) + Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
    # выходы второго слоя = выходы первого слоя * веса второго слоя
    out = np.where((np.dot(hidden_predict, Wout[1:,:]) + Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict
# hidden_predict - выходы нейронов первого слоя они же признаки нейрона 2ого слоя, веса которого мы корректируем


# not original content
eta = 0.01
# сделал WoutOld - вектор старых весов выходного слоя - и рандомно его инициализировал
WoutOld =  np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
while True:
    # обучение
    # у перцептрона Розенблатта обучаются только веса выходного слоя
    # то есть меняется толкьо матрица Wout по тому же самому правилу Хебба
    # если ответы не совпали, кореектируем веса, если совпали - не трогаем 
    # как и раньше обучаем подавая по одному примеру и корректируем веса в случае ошибки
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr_out, hidden_layer = predict(xi) 
        # новый вес = старый вес + eta * признак * разница между ответом перцептрона и правильным
        Wout[1:] = Wout[1:] + ((eta * (target - pr_out)) * hidden_layer).reshape(-1, 1)
        Wout[0] = Wout[0] + eta * (target - pr_out)


    pr, hidden = predict(X)
    
    convergence = sum(pr-y.reshape(-1, 1))
    looping = sum(Wout-WoutOld.reshape(-1,1))
    if(convergence == 0 or looping == 0):
        if (convergence == 0):
            print('я обучился!')
        else :
            print('я необучаем :(')
        break
    WoutOld = Wout

# далее оформляем все это в виде отдельного класса neural.py
