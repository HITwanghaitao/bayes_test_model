import pymc3 as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from pymc3.distributions import Interpolated
from scipy import stats
import matplotlib as mpl
x_size = 100
global x1,x2
x1 = np.random.randn(x_size)
x2 = np.random.randn(x_size)
def sim_model(a,b):
    return x1*a + x2*b +3
at = 3
bt = 5
y_obs = sim_model(at, bt)
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

with pm.Model() as my_model:
    a = pm.Uniform("a",lower=1,upper=4)
    b = pm.Uniform("b",lower=4,upper=7)
    s = pm.Simulator("s",sim_model,params=[a,b],sum_stat="sort", epsilon=1, observed=y_obs)

    trace, sim_data = pm.sample_smc(kernel="ABC", parallel=False, save_sim_data=True)
    idata = az.from_pymc3(trace, posterior_predictive=sim_data)

#az.plot_trace(idata, kind="rank_vlines");
traces = [trace]
for _ in range(2):

    # generate more data

    model = pm.Model()
    with model:
        # Priors are posteriors from previous iteration
        a = from_posterior("a",trace["a"])
        b = from_posterior("b", trace["b"])

        # Expected value of outcome
        s = pm.Simulator("s", sim_model, params=[a, b], sum_stat="sort", epsilon=1, observed=y_obs)

        trace, sim_data = pm.sample_smc(kernel="ABC", parallel=False, save_sim_data=True)
        idata = az.from_pymc3(trace, posterior_predictive=sim_data)

        traces.append(trace)

print("Posterior distributions after " + str(len(traces)) + " iterations.")
cmap = mpl.cm.autumn
for param in ["a", "b"]:
    plt.figure(figsize=(8, 2))
    for update_i, trace in enumerate(traces):
        samples = trace[param]
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)
        plt.plot(x, y, label = 'line'+str(update_i),color=cmap(1 - update_i / len(traces)))

    plt.ylabel("Frequency")
    plt.title(param)
    plt.legend()

plt.tight_layout();
plt.show()
