import numpy as np

'''
PERCEPTRON STEPS:
    1 ) inizializzare pesi a 0 o a numeri piccoli casuali
    2) per ogni esempio di addestramento x(i) : a) calcolare il valore di output y
                                                b) aggiornare i pesi
'''

class Perceptron(object):
    '''
    i parametri della classe saranno: eta, ovvero il learning rate, n_iter ovvero il numero di epoche
    ed infine random state

    gli attributi invece saranno w che è un array 1-dimensionale contenente i pesi dopo il fitting
    ed errors che contiene il numero di classificazioni in ogni epoca
    '''

    def __init__(self, eta =  0.01, n_iter = 50, random_state = 1):
        self.eta = eta #learning rate, compreso tra 0.0 e 1.0
        self.n_iter = n_iter #numero di epoche
        self.random_state = random_state #seme per numeri casuali, garantisce la riproduzione degli stessi risultati


    def fit(self, X, y):
        '''
        i parametri saranno X che è un array dato da [n_examples, n_features]
        y che è un array dato da [n_examples]
        '''
        random_generator = np.random.RandomState(self.random_state) #creazione generatore di numeri casuali con seme fisso
        self.w_ = random_generator.normal(loc=0.0, scale=0.01,size=1+X.shape[1])
        '''
        il metodo .normal genera numeri casuali da una funzione gaussiana o campanulare, con loc=0.0 indichiamo che la media della distribuzione
        è 0 e che i numeri saranno centrati intorno 0, scale è la deviazione standard ovvero quanto si allontanano i numeri dallo 0, con size
        invece indichiamo quanti numeri generare ed aggiungiuamo +1 perchè uno dei pesi deve essere il bias
        '''

        self.errors_ = []

        for _ in range(self.n_iter): #scorriamo nel numero di epoche
            errors = 0
            for xi,target in zip(X,y): #scorriamo ogni esempio xi con l 'etichetta target
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update #valore di bias
                errors += int(update != 0)
            self.errors_.append(errors)
        return self


    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0] #uso _ per indicare che questo attributo esiste solo dopo aver chiamato fit

    def predict(self,X):
        '''np.where è una funzione numpy con la sciamente sintassi : (condizione,valore se true, valore se false)
        essenzialmente stiamo traducento la funzione a gradino in python
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)