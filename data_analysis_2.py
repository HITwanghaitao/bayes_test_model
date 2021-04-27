import numpy as np
import scipy.io as io

params = ['Lgrad','vc','h','m','rou']
for k in range(5):
    theta_param = io.loadmat(params[k] +'_data.mat')
    theta = theta_param[params[k]]
    print(theta)
    matpath1 = params[k] + '_data_c.mat'
    for j in range(4):
        theta[j+1,1] = np.abs(theta[j+1,0] - theta[0,0])/theta[0, 0]*100

        theta[j + 1, 4] = (theta[j+1,3] - theta[j+1,2])/(theta[0,3] - theta[0,2])*100
    io.savemat(matpath1,{params[k]+'_c':theta})
    print(theta)
