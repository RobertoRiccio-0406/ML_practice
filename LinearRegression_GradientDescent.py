import numpy as  np
from warnings import warn
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt


def LR_Gradient_Descent(X:np.ndarray,Y:np.ndarray,Epochs = 1000, LearningRate = 0.0001,tol = 0.000001) :

    a = 0
    b = 0
    n = len(X) #numero di campioni dati

    '''
    usiamo una Rl del tipo y = mX + c
    '''

    for rep in range(Epochs):
        y_pred = a*(X)+b
        partial_derivative_a = (-2/n) * np.sum(X * (Y - y_pred))
        partial_derivative_b = (-2/n) * np.sum(Y - y_pred)
        a = a-(LearningRate*partial_derivative_a)
        b = b-(LearningRate*partial_derivative_b)
        Mean_Square_Error = np.sum(np.square(y_pred-Y))/n

        if Mean_Square_Error < tol:
            print(f"early stop, epoch {rep}, error is {Mean_Square_Error}")
            break
        if Mean_Square_Error == float("inf"):
            warn('the model diverged')
            return 0,0

    return a,b

'''test'''
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
Y = np.array([2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.2, 16.0, 17.8, 20.1], dtype=float)


a,b = LR_Gradient_Descent(X,Y, Epochs = 5000)
print(f"{a},{b}")


plt.scatter(X, Y, marker='o', s=100, color='blue', label='Dati')
plt.plot(X, a*X + b, color='red', label=f'y = {a:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regressione Lineare - Gradient Descent')
plt.show()
