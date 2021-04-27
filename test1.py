import pymc3 as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
x_size = 100
global x1,x2
x1 = np.random.randn(x_size)
x2 = np.random.randn(x_size)
def sim_model(a,b):
    #x_size = 10000
    #x1 = np.random.randn(x_size)
    #x2 = np.random.randn(x_size)
    return x1*a + x2*b +3
at = 3
bt = 5
y_obs = sim_model(at, bt)

with pm.Model() as my_model:
    a = pm.Uniform("a",lower=1,upper=4)
    b = pm.Uniform("b",lower=4,upper=7)
    s = pm.Simulator("s",sim_model,params=[a,b],sum_stat="sort", epsilon=1, observed=y_obs)

    trace, sim_data = pm.sample_smc(kernel="ABC", parallel=False, save_sim_data=True)
    idata = az.from_pymc3(trace, posterior_predictive=sim_data)

az.plot_trace(idata, kind="rank_vlines");
plt.show()

