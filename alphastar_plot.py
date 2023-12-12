import pickle as pkl
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor']='snow'

def plot_error(data, label, color):
    data_mean = np.mean(np.array(data), axis=0)
    error_bars = stats.sem(np.array(data))
    plt.plot(data_mean, label=label, color=color)
    plt.fill_between([i for i in range(data_mean.size)],
                     np.squeeze(data_mean - error_bars),
                     np.squeeze(data_mean + error_bars), 
                     color=color,
                     alpha=alpha)


data = pkl.load(open("results/AlphaStar/data.p", "rb"))
alpha = .4
j=0

fig_handle = plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#252d2e', 'tab:red']
label_dict = {
   "psro_exps": 'PSRO',
   "rectified_exps": 'PSRO-rN',
   "pipeline_psro_exps": 'P-PSRO',
   "dpp_psro_exps": 'DPP-PSRO',
   "bd_rd_psro_exps": 'BD&RD-PSRO',
   "psd_psro_exps": 'PSD-PSRO (Ours)',
}

all_methods = ['psro_exps', 'pipeline_psro_exps', 'rectified_exps', 'dpp_psro_exps', 'bd_rd_psro_exps', 'psd_psro_exps']
for ii, method in enumerate(all_methods):
    me_data = data[method]
    if method == "rectified_exps":
        length = min([len(l) for l in me_data])
        for i, l in enumerate(me_data):
            me_data[i] = me_data[i][:length]
    plot_error(me_data, label=label_dict[method], color=colors[ii])


plt.grid()
plt.legend(loc="upper right", prop={'size': 15})
plt.title("AlphaStar888", size=20)
plt.xlabel("Iterations", size=15)
plt.ylabel("Exploitability", size=15)
plt.yscale('log')
string = 'Exploitability Log'
plt.savefig('png/alphastar888_exp.png',dpi=300, bbox_inches='tight')

alpha = .4
j=0
fig_handle = plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#252d2e', 'tab:red']
label_dict = {
   "psro_cardinality": 'PSRO',
   "rectified_cardinality": 'PSRO-rN',
   "pipeline_psro_cardinality": 'P-PSRO',
   "dpp_psro_cardinality": 'DPP-PSRO',
   "psd_psro_cardinality": 'PSD-PSRO (Ours)',
   "bd_rd_psro_cardinality": 'BD&RD-PSRO'
}

all_methods = ['psro_cardinality', 'pipeline_psro_cardinality', 'rectified_cardinality', 
               'dpp_psro_cardinality', 'bd_rd_psro_cardinality', 'psd_psro_cardinality']
for ii, method in enumerate(all_methods):
    me_data = data[method]
    if method == "rectified_cardinality":
        length = min([len(l) for l in me_data])
        for i, l in enumerate(me_data):
            me_data[i] = me_data[i][:length]
    plot_error(me_data, label=label_dict[method], color=colors[ii])


plt.grid()
plt.legend(loc="upper right", prop={'size': 15})
plt.title("AlphaStar888", size=20)
plt.xlabel("Iterations", size=15)
plt.ylabel("Population Exploitability", size=15)
plt.yscale('log')
string = 'Exploitability Log'
plt.savefig('png/alphastar888_PE.png',dpi=300,bbox_inches='tight')