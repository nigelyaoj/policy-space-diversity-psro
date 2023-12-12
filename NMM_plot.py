import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
from scipy import stats
import pickle
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(123)


class MyGaussianPDF(nn.Module):
    def __init__(self, mu):
        super(MyGaussianPDF, self).__init__()
        self.mu = mu
        self.cov = 0.54*torch.eye(2)
        # self.c = (1./(2*np.pi))
        self.c = 1.

    def forward(self, x):
        return self.c*torch.exp(-0.5*torch.diagonal( (x-self.mu)@self.cov@(x-self.mu).t() ))

class GMMAgent(nn.Module):
    def __init__(self, mu):
        super(GMMAgent, self).__init__()
        self.gauss = MyGaussianPDF(mu).to(device)
        self.x = nn.Parameter(0.01*torch.randn(2, dtype=torch.float), requires_grad=False)

    def forward(self):
        return self.gauss(self.x)

class TorchPop:
    def __init__(self, num_learners, seed=0):
        torch.manual_seed(seed)
        self.pop_size = num_learners + 1

        mus = np.array([[2.8722, -0.025255],
                        [1.8105, 2.2298],
                        [1.8105, -2.2298],
                        [-0.61450, 2.8058],
                        [-0.61450, -2.8058],
                        [-2.5768, 1.2690],
                        [-2.5768, -1.2690]]
                       )
        mus = torch.from_numpy(mus).float().to(device)
        self.mus = mus

        self.game = torch.from_numpy(np.array([
                                               [0., 1., 1., 1, -1, -1, -1],
                                               [-1., 0., 1., 1., 1., -1., -1.],
                                               [-1., -1., 0., 1., 1., 1., -1],
                                               [-1., -1., -1., 0, 1., 1., 1.],
                                               [1., -1., -1., -1., 0., 1., 1.],
                                               [1., 1., -1., -1, -1, 0., 1.],
                                               [1., 1., 1., -1., -1., -1., 0.]
                                               ])).float().to(device)

        self.pop = [GMMAgent(mus) for _ in range(self.pop_size)]
        self.pop_hist = [[self.pop[i].x.detach().cpu().clone().numpy()] for i in range(self.pop_size)]

    def get_js_divergence(self, agent1, metanash, K):
        def entropy(p_k):
            p_k = p_k + 1e-8
            p_k = p_k / torch.sum(p_k)
            return -torch.sum(p_k * torch.log(p_k))

        agg_agent = metanash[0] * self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k] * self.pop[k]()
        agent1_values = agent1()
        agent1_values = agent1_values / torch.sum(agent1_values)
        agg_agent = agg_agent / torch.sum(agg_agent)
        return 2 * entropy((agent1_values + agg_agent) / 2) - entropy(agent1_values) - entropy(agg_agent)

    def visualise_pop(self, br=None, ax=None, color=None):

        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2 * np.pi) ** n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
            return np.exp(-fac / 2) / N

        agents = [agent.x.detach().cpu().numpy() for agent in self.pop]
        agents = list(zip(*agents))

        # Colors
        if color is None:
            colors = cm.rainbow(np.linspace(0, 1, len(agents[0])))
        else:
            colors = [color]*len(agents[0])

        # fig = plt.figure(figsize=(6, 6))
        ax.scatter(agents[0], agents[1], alpha=1., marker='.', color=colors, s=8*plt.rcParams['lines.markersize'] ** 2)
        if br is not None:
            ax.scatter(br[0], br[1], marker='.', c='k')
        for i, hist in enumerate(self.pop_hist):
            if hist:
                hist = list(zip(*hist))
                ax.plot(hist[0], hist[1], alpha=0.8, color=colors[i], linewidth=4)

        # ax = plt.gca()
        for i in range(7):
            ax.scatter(self.mus[i, 0].item(), self.mus[i, 1].item(), marker='x', c='k')
            for j in range(4):
                delta = 0.025
                x = np.arange(-4.5, 4.5, delta)
                y = np.arange(-4.5, 4.5, delta)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, self.mus[i,:].numpy(), 0.54 * np.eye(2))
                levels = 10
                # levels = np.logspace(0.01, 1, 10, endpoint=True)
                CS = ax.contour(X, Y, Z, levels, colors='k', linewidths=0.5, alpha=0.2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                # ax.clabel(CS, fontsize=9, inline=1)
                # circle = plt.Circle((0, 0), 0.2, color='r')
                # ax.add_artist(circle)
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])


    def get_payoff(self, agent1, agent2):
        p = agent1()
        q = agent2()
        return p @ self.game @ q + 0.5*(p-q).sum()

    def get_payoff_aggregate(self, agent1, metanash, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = metanash[0]*self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_payoff_aggregate_weights(self, agent1, weights, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = weights[0]*self.pop[0]()
        for k in range(1, len(weights)):
            agg_agent += weights[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_br_to_strat(self, metanash, lr, nb_iters=20, BR=None):
        if BR is None:
            br = GMMAgent(self.mus)
            br.x = nn.Parameter(0.1*torch.randn(2, dtype=torch.float), requires_grad=False)
            br.x.requires_grad = True
        else:
            br =  BR
        optimiser = optim.Adam(br.parameters(), lr=lr)
        for _ in range(nb_iters*20):
            loss = -self.get_payoff_aggregate(br, metanash, self.pop_size,)
            # Optimise !
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return br

    def get_metagame(self, k=None, numpy=False):
        if k==None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
                return metagame.detach().cpu().clone().numpy()
        else:
            metagame = torch.zeros(k, k)
            for i in range(k):
                for j in range(k):
                    metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
            return metagame

    def add_new(self):
        with torch.no_grad():
            self.pop.append(GMMAgent(self.mus))
            self.pop_hist.append([self.pop[-1].x.detach().cpu().clone().numpy()])
            self.pop_size += 1

    def get_exploitability(self, metanash, lr, nb_iters=20):
        br = self.get_br_to_strat(metanash, lr, nb_iters=nb_iters)
        with torch.no_grad():
            exp = self.get_payoff_aggregate(br, metanash, self.pop_size).item()

        return exp

PATH_RESULTS="results/NMM"
FILE_TRAJ = {
'rectified': 'rectified.p',
'psro': 'psro.p',
'p-psro': 'p_psro.p',
'dpp': 'dpp.p',
'psd': 'psd.p', 
'bd_rd': 'bd_rd.p'
}


method_nums = len(FILE_TRAJ)
titles = {
    'rectified': 'PSRO-rN',
    'dpp': 'DPP-PSRO',
    'p-psro': 'P-PSRO',
    'psro': 'PSRO',
    'psd': 'PSD-PSRO',
    'bd_rd': "BD&RD-PSRO"
}
methods = ['psro', 'p-psro', 'rectified', 'dpp', 'bd_rd', 'psd']
pops = {}
fig1, axs1 = plt.subplots(1, method_nums, figsize=(5 * method_nums, 5 * 1), dpi=200)
axs1 = axs1.flatten()
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#252d2e', 'tab:red'] #, '#7f7f7f', '#bcbd22', '#17becf']
# for i, key in enumerate(FILE_TRAJ.keys()):
for i, key in enumerate(methods):
    ax = axs1[i]
    d = pickle.load(open(os.path.join(PATH_RESULTS, FILE_TRAJ[key])+'4.p', 'rb'))
    pops[FILE_TRAJ[key]] = d['pop']
    pops[FILE_TRAJ[key]].visualise_pop(ax=ax, color=colors[i])
    ax.set_title(titles[key],size=20)
    # ax.set_xlabel(xlabels[i], size=20)

fig1.tight_layout()
fig1.savefig(os.path.join("png", 'NMM_traj.png'),dpi=300,bbox_inches='tight')