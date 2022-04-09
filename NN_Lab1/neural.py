import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
    
        #тут храним все векторы весов (вообще все, что были)
        self.WoutOld = [self.Wout.tolist()]
        
        
    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    #функция для нахождения цикла
    def looping (self):
        WoutList = self.Wout.tolist()
        for i in range (len(self.WoutOld) - 1, len(self.WoutOld) // 2, -1):
            if str(self.WoutOld[i:] + [WoutList])[1:-1] in str(self.WoutOld[:i])[1:-1]:
                return False
        return True

    def train(self, X, y, n_iter=10, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
                
            #проверяем, есть ли ошибки
            pr, hidden = self.predict(X)
            convergence = sum(pr-y.reshape(-1, 1))
            
            #проверяем есть ли зацикливание
            loop = self.looping()
            if(convergence == 0 or loop == 0):
                if (convergence == 0):
                    print('я обучился!')    #произошла сходимость
                else :
                    print('я необучаем :(') #произошло зацикливание
                return self
            self.WoutOld.append(self.Wout.tolist())
        
