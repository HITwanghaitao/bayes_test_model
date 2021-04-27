import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az
import pandas as pd

params = ['Lgrad','vc','h','m','rou']
k = 4
theta = io.loadmat(params[k] +'_posterior.mat')
data = open(params[k] + '_map.txt','w+')
for j in range(4):
    x = theta[params[k]+'x'][j, :]
    y = theta[params[k]+'y'][j,:]
    xflag = np.where(y == max(y))
    print(str(j+1)+"_times",file=data)
    print(x[xflag], file=data)
    plt.plot(x, y, label=str(j+1))
data.close()
plt.yticks([])
plt.legend()
plt.title('posteriori distribution of ' + params[k] + ' for 4 times')

plt.show()
