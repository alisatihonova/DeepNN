import torch 
import numpy as np
import pandas as pd

# Основная структура данных pytorch - тензоры
# Основные отличия pytorch - тензоров от numpy массивов:
# 1 Место их хранения можно задать ( память CPU или GPU )
# 2 В отношении тензоров можно задать вычисление и отслеживание градиентов

# Создавать тензоры можно разными способами:
# Пустой тензор
a = torch.empty(5, 3)
print(a) # тензор содержит "мусор"

b = torch.Tensor(5, 3)
print(b) # тензор содержит "мусор"

# тензор с нулями
a = torch.zeros(5, 3)
print(a) # тензор содержит "мусор"

b = torch.ones(5, 3)
print(b) # тензор содержит "мусор"

# тензор со случайными числами
a = torch.rand(5, 3)
print(a) # распределеными по равномерному закону распределения

b = torch.randn(5, 3)
print(b) # # распределеными по нормальному закону распределения

# или можем явно указать нужные значения
a = torch.Tensor([[1,2],[3,4]])
print(a) 

# Наиболее часто используемые методы создания тензоров

#    torch.rand: случайные значения из равномерного распределения
#    torch.randn: случайные значения из нормального распределения
#    torch.eye(n): единичная матрица
#    torch.from_numpy(ndarray): тензор на основе ndarray NumPy-массива
#    torch.ones : тензор с единицами
#    torch.zeros_like(other): тензор с нулями такой же формы, что и other
#    torch.range(start, end, step): 1-D тензор со значениями между start и end с шагом steps


# тензоры можно преобразовать к Numpy массивам
# !!! но нужно не забывать про копирование данных !!!
c = a.numpy().copy()

# тензоры можно "слайсить" как и Numpy массивы
print(a[1,:].numpy())

# Понять можем ли мы использовать графический ускоритель для вычислений поможет функция
print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

# создавать тензоры можно в памяти видеокарты
a = torch.Tensor([[1,2],[3,4]], device='cuda')

# Для вычислений на основе обратного распространения ошибки некоторые тензоры 
# должны отслеживать градиенты
a = torch.randn(2, 2, requires_grad=False)
a.requires_grad

a.requires_grad=True
print(a)

# Теперь все операци над тензором a будут отслеживаться
# Выполним какую-нибудь операцию с тензором:

a = a + 2
print(a)

# В выводе появляется атрибут grad_fn, который хранит информацию об операции с тензором

print(a.grad_fn)

a = a * 2
print(a)

print(a.grad_fn)

# Все это нужно для вычисления градиентов
# посмотрим детально как это происходит
# создадим простую последовательность вычислений

x = torch.zeros(2, 2, requires_grad=True)
y = x + 3
z = y**2
out = z.mean()
print(z)
print(out)

# теперь воспользовавшись chain-rule продифференцируем полученную последовательность
# для этого вызовем метод .backward(), который проведет дифференцирование в обратном порядке 
# до изначально заданного тензора x
out.backward()
print(x.grad) # градиенты d(out)/dx

# метод .backward() без аргументов работает только для скаляра (например ошибки нейросети)
# чтобы вызвать его для многомерного тензра, внего в качестве параметра необходимо 
# передать значения градиентов с "предыдущего" блока  
x = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
print(z)
z = x**2
print(z) 
z.backward(torch.ones(2,2))
print(x.grad) # градиенты d(z)/dx = 2*x

############################################################################### 

# Для работы с нейрнными сетями предоставляется широкий набор инструментов:
# слои, функции активации, функционалы потерь, оптимизаторы
import torch.nn as nn

# Создадим 2 тензора - вход размером (10, 3) и выход, размером (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# создадим слой сумматоров или полносвязный слой (fully connected layer)
linear = nn.Linear(3, 2)

# при создании веса и смещения инициализируются автоматически
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# выберем вид функции ошибки и оптимизатор
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# прямой проход (пресказание) выглядит так:
pred = linear(x)

# имея предсказание можно вычислить ошибку
loss = lossFn(pred, y)
print('Ошибка: ', loss.item())

# и сделать обратный проход
loss.backward()

# обратный проход вычислит градиенты по параметрам
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# далее можем сделать шаг оптимизации
optimizer.step()

# далее итерационно повторяем шаги
pred = linear(x)
loss = lossFn(pred, y)
print('Ошибка: ', loss.item())
loss.backward()
optimizer.step()

# тоже самое в цикле (фактически это и есть функция обучения):
for i in range(0,10):
    pred = linear(x)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()
    

###############################################################################

# создадим свой класс перцептрона, но уже на основе модуля pytorch

# наш класс будем наследовать от nn.Module, который включает большую часть необходимого нам функционала

class Perceptron(nn.Module):
    # как и раньше для инициализации на вход нужно подать размеры входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Sigmoid(),                    # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid(),
        )
    # прямой проход    
    def forward(self,x):
        return self.layers(x)

# функция обучения
def train(x, y, num_iter):
    for i in range(0,num_iter):
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i%100==0:
           print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    return loss.item()


# теперь можно использовать созданный класс на практике

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)
X = df.iloc[0:100, 0:3].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи

net = Perceptron(inputSize,hiddenSizes,outputSize)
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

loss_ = train(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)), 1000)

pred = net.forward(x)
for name, param in net.named_parameters():
    print(name, param)

