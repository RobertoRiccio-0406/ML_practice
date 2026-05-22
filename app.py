from Perceptron import Perceptron
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt



nomi_colonne = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
path = os.path.join(r"C:\Users\Rober\OneDrive\Desktop\machine learning", 'iris.data')

df = pd.read_csv(path,names = nomi_colonne, encoding = 'utf-8')

y = df.iloc[0:100,4].values #iloc seleziona un range di righe e value lo restituisce come array numpy, 4 indica la colonna selezionata

'''convertiamo ora y in 1 e -1'''

y = np.where(y=="Iris-setosa",-1,1)

X = df.iloc[0:100, [0,2]].values #estraiamo le caratteristiche della lunghezza del sepalo e del petalo

# #plot data
# plt.scatter(X[:50,0], X[:50,1], color='red', marker = 'o', label = 'setosa') #creo un grafico a dispersione prendendo i primi 50 fiori
# plt.scatter(X[50:100,0], X[50:100,1], color='yellow', marker = 'x', label = 'verscicolor')
# plt.xlabel('sepal lenght [cm]')
# plt.ylabel('petal lenght [cm]')
# plt.legend(loc='upper left')
# plt.savefig('foo.png', bbox_inches='tight')

ppn = Perceptron(eta=0.1 , n_iter=50) #creo l'oggetto
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.savefig('foo2.png', bbox_inches='tight')

