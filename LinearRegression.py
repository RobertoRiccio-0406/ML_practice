import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')
import numpy as np


x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

x_mean = np.mean(x)
y_mean = np.mean(y)

m_num = np.sum((x-x_mean)*(y-y_mean))
m_den = np.sum((x-x_mean)**2)
m = m_num/m_den

c = y_mean-(m*x_mean)


plt.scatter(x,y, color='blue',marker='x',s=30) #s is marker size
y_pred = c + m*x
plt.plot(x,y_pred,color='red')
plt.title('linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()