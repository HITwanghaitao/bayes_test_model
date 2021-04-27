## 重要 贝叶斯推断
import pymc3 as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from pymc3.distributions import Interpolated
from scipy import stats
import matplotlib as mpl
import scipy.io as io
import pandas as pd
np.set_printoptions(threshold=np.inf)

pd.set_option('display.width', 300) # 设置字符显示宽度
pd.set_option('display.max_rows', None) # 设置显示最大行
pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列
x_size = 100
global x1,x2,x3,x4,x5
x1 = np.random.randn(x_size)
x2 = np.random.randn(x_size)
x3 = np.random.randn(x_size)
x4 = np.random.randn(x_size)
x5 = np.random.randn(x_size)
def sim_model(Lgrad,vc,h,m,rou):
    return Lgrad*x1*10 + vc*x2 + h*x3*100 + m*x4*100 + 4*rou*x5 + 3
Lgradt = 0.403
vct = 2
ht = 0.04
mt = 0.04
rout = 1.2
theat_ts = [Lgradt, vct, ht, mt, rout ]
y_obs = sim_model(Lgradt,vct,ht,mt,rout)
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
bounds = np.array([[0.2, 0.6], [1.5, 2.5], [0.03,0.05], [0.02,0.05], [1.0,1.3]])

with pm.Model() as my_model:
    Lgrad = pm.Uniform("Lgrad",lower=bounds[0,0],upper=bounds[0,1])
    vc = pm.Uniform("vc",lower=bounds[1,0],upper=bounds[1,1])
    h = pm.Uniform("h", lower=bounds[2,0], upper=bounds[2,1])
    m = pm.Uniform("m", lower=bounds[3,0], upper=bounds[3,1])
    rou = pm.Uniform("rou", lower=bounds[4,0], upper=bounds[4,1])

    s = pm.Simulator("s",sim_model,params=[Lgrad,vc,h,m,rou],sum_stat="sort", epsilon=1, observed=y_obs)

    trace, sim_data = pm.sample_smc(kernel="ABC", parallel=False, save_sim_data=True)
    idata = az.from_pymc3(trace, posterior_predictive=sim_data)

#az.plot_trace(idata, kind="rank_vlines");
#pm.traceplot(trace)
#plt.show()
traces = [trace]

global x11,x21,x31,x41,x51
def sim_model_1(Lgrad, vc, h, m, rou):
    return Lgrad * x11 * 10 + vc * x21 + h * x31 * 100 + m * x41 * 100 + 4 * rou * x51 + 3
for _ in range(3):

    x11 = np.random.randn(x_size)
    x21 = np.random.randn(x_size)
    x31 = np.random.randn(x_size)
    x41 = np.random.randn(x_size)
    x51 = np.random.randn(x_size)


    y_obs1 = sim_model_1(Lgradt,vct,ht,mt,rout)
    # generate more data

    model = pm.Model()
    with model:
        # Priors are posteriors from previous iteration
        Lgrad = from_posterior("Lgrad",trace["Lgrad"])
        vc = from_posterior("vc", trace["vc"])
        h = from_posterior("h", trace["h"])
        m = from_posterior("m", trace["m"])
        rou = from_posterior("rou", trace["rou"])

        # Expected value of outcome
        s = pm.Simulator("s", sim_model_1, params=[Lgrad,vc,h,m,rou], sum_stat="sort", epsilon=1, observed=y_obs1)

        trace, sim_data = pm.sample_smc(draws=2000*(_+1),kernel="ABC", parallel=False, save_sim_data=True)
        idata = az.from_pymc3(trace, posterior_predictive=sim_data)

        traces.append(trace)

print("Posterior distributions after " + str(len(traces)) + " iterations.")
cmap = mpl.cm.autumn
flag = 0
for param in ["Lgrad", "vc", "h", "m", "rou"]:
    #plt.figure(figsize=(8, 2))
    mat_path1 = param + '_posterior.mat' #分布数据
    mat_path0 = param + '_data.mat' #生成表数据
    data = open(param + '_posterior_stat.txt', 'w+')  # 写入分析数据的文本文件

    x4_100 = np.zeros((4,100))
    y4_100 = np.zeros((4,100))

    data_table = np.zeros((5,5))
    data_table[0,0] = theat_ts[flag]
    data_table[0,2] = bounds[flag, 0]
    data_table[0, 3] = bounds[flag, 1]
    flag = flag + 1

    for update_i, trace in enumerate(traces):
        samples = trace[param]
        stat_y = az.summary(samples,hdi_prob=0.95)
        data_table[update_i+1,2] = stat_y['hdi_2.5%']['x']
        data_table[update_i + 1, 3] = stat_y['hdi_97.5%']['x']

        print(param + '_posterior_stat_'+str(update_i+1), file=data)
        print(stat_y, file=data)
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)

        xflag = np.where(y == max(y))
        data_table[update_i+1,0] = x[xflag]

        x4_100[update_i,:] = x
        y4_100[update_i,:] = y
        s_choice = str(update_i+1)

    data.close()

    name1 = param + 'x'
    name2 = param + 'y'

    io.savemat(mat_path1, {name1: x4_100, name2: y4_100})

    io.savemat(mat_path0, {param:data_table})




