## 重要 贝叶斯推断
# 采用函数式模型（简易模型）进行贝叶斯推断
# Lgrad*x1*10 + vc*x2 + h*x3*100 + m*x4*100 + 4*rou*x5 + 3
import pymc3 as pm 
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from pymc3.distributions import Interpolated
from scipy import stats
import matplotlib as mpl
import scipy.io as io
import pandas as pd
import time
import os

#设置print输出全部显示
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 300) # 设置字符显示宽度
pd.set_option('display.max_rows', None) # 设置显示最大行
pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列

#利用采样点生成分布的函数，from_posterior
def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)
    
    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)

#获取当前特定格式的时间字符串,如2021_04_28_06_31_00
def getTimeStr():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

#观测数据个数
x_size = 100

#先验分布（均匀分布）区间
bounds = np.array([[0.2, 0.6], [1.5, 2.5], [0.03,0.05], [0.02,0.05], [1.0,1.3]])

########### 第一次贝叶斯推断 ###########
#生成第一次观测数据时所用X输入
global x1,x2,x3,x4,x5
x1 = np.random.randn(x_size)
x2 = np.random.randn(x_size)
x3 = np.random.randn(x_size)
x4 = np.random.randn(x_size)
x5 = np.random.randn(x_size)

# 简易模型（用于第一次）
def sim_model(Lgrad,vc,h,m,rou):
    return Lgrad*x1*10 + vc*x2 + h*x3*100 + m*x4*100 + 4*rou*x5 + 3

# 参数真实值，共5个
Lgradt = 0.403
vct = 2
ht = 0.04
mt = 0.04
rout = 1.2
theat_ts = [Lgradt, vct, ht, mt, rout ]
# 观测数据（第一次）
y_obs = sim_model(Lgradt,vct,ht,mt,rout)

# 构造pymc3.Model模型
with pm.Model() as my_model:
    Lgrad = pm.Uniform("Lgrad",lower=bounds[0,0],upper=bounds[0,1])
    vc = pm.Uniform("vc",lower=bounds[1,0],upper=bounds[1,1])
    h = pm.Uniform("h", lower=bounds[2,0], upper=bounds[2,1])
    m = pm.Uniform("m", lower=bounds[3,0], upper=bounds[3,1])
    rou = pm.Uniform("rou", lower=bounds[4,0], upper=bounds[4,1])
    
    s = pm.Simulator("s",sim_model,params=[Lgrad,vc,h,m,rou],sum_stat="sort", epsilon=1, observed=y_obs)

    trace, sim_data = pm.sample_smc(kernel="ABC", parallel=False, save_sim_data=True)
    # 暂时不用 idata = az.from_pymc3(trace, posterior_predictive=sim_data)

traces = [trace] #将第一次的trace数据添加到traces中
########### 第二，三，四次贝叶斯推断 ###########

#声明X输入为全局变量
global x11,x21,x31,x41,x51
#简易模型（用于第2，3，4次）
def sim_model_1(Lgrad, vc, h, m, rou):
    return Lgrad * x11 * 10 + vc * x21 + h * x31 * 100 + m * x41 * 100 + 4 * rou * x51 + 3

for _ in range(3):
    #生成X输入
    x11 = np.random.randn(x_size)
    x21 = np.random.randn(x_size)
    x31 = np.random.randn(x_size)
    x41 = np.random.randn(x_size)
    x51 = np.random.randn(x_size)
    
    #生成观测数据
    y_obs1 = sim_model_1(Lgradt,vct,ht,mt,rout)
    
    #构造pymc3.Model模型
    model = pm.Model()
    with model:
        # Priors are posteriors from previous iteration
        Lgrad = from_posterior("Lgrad",trace["Lgrad"])
        vc = from_posterior("vc", trace["vc"])
        h = from_posterior("h", trace["h"])
        m = from_posterior("m", trace["m"])
        rou = from_posterior("rou", trace["rou"])

        s = pm.Simulator("s", sim_model_1, params=[Lgrad,vc,h,m,rou], sum_stat="sort", epsilon=1, observed=y_obs1)
        
        trace, sim_data = pm.sample_smc(draws=2000*(_+1),kernel="ABC", parallel=False, save_sim_data=True)
        # 暂时不用 idata = az.from_pymc3(trace, posterior_predictive=sim_data)

        traces.append(trace) #添加trace到traces
        
#print贝叶斯推断次数
print("Posterior distributions after " + str(len(traces)) + " iterations.")

#将贝叶斯推断结果输出为txt与mat文件
flag = 0
main_path = os.getcwd() + '\\' + 'Time' + getTimeStr() + '_function_model_bayes_inference' #生成文件夹路径
os.makedirs(main_path) # 生成文件夹
for param in ["Lgrad", "vc", "h", "m", "rou"]:
    # 以下生成文件，存放在文件夹'Time' + getTimeStr()内，如 Time2021_04_28_06_31_00_function_model_bayes_inference
    mat_path0 = main_path + '\\' + param + '_bayes_analysis.mat'  # 参数param的后验分布分析数据mat文件路径，文件名如 'Lgrad_bayes_analysis.mat’
    mat_path1 = main_path + '\\' + param + '_posteriori.mat'  # 参数param的后验分布mat文件路径，文件名如 'Lgrad_posteriori.mat'

    az_summary = open(
        main_path + '\\' + param + '_posteriori_az_summary.txt',
        'w+')  # 写入贝叶斯推断产生数据的分析结果文本文件
    
    x4_100 = np.zeros((4,100)) #用于存储"Lgrad", "vc", "h", "m", "rou"的后验分布的横坐标数据 x
    y4_100 = np.zeros((4,100)) #用于存储"Lgrad", "vc", "h", "m", "rou"的后验分布的纵坐标数据 y
    
    data_table = np.zeros((5,5)) #用于贝叶斯推断数据表格的填写
    data_table[0,0] = theat_ts[flag] #写入数据
    data_table[0,2] = bounds[flag, 0]
    data_table[0, 3] = bounds[flag, 1]
    flag = flag + 1
    
    for update_i, trace in enumerate(traces): #对5个参数进行遍历
        samples = trace[param]
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100) #横坐标
        y = stats.gaussian_kde(samples)(x) #纵坐标
        
        x4_100[update_i,:] = x
        y4_100[update_i,:] = y
        
        az_summary_data = az.summary(samples,hdi_prob=0.95) #获取az.summary数据
        
        data_table[update_i + 1, 2] = az_summary_data['hdi_2.5%']['x'] #将置信数据写入表格数据内
        data_table[update_i + 1, 3] = az_summary_data['hdi_97.5%']['x']
        
        print(param + '_az_summary_'+str(update_i+1), file=az_summary) #写入
        print(az_summary_data, file=az_summary) #将az.summary全部数据写入文本文件
        
        xflag = np.where(y == max(y))
        data_table[update_i+1,0] = x[xflag] #将map写入表格数据内
        
        

    name1 = param + 'x' #mat文件变量名，如： Lgradx
    name2 = param + 'y' #mat文件变量名，如： Lgrady
    io.savemat(mat_path0, {param:data_table})
    io.savemat(mat_path1, {name1: x4_100, name2: y4_100}) # 保存为mat文件



