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
from numpy.random import RandomState
import copy
from tqdm import tqdm
from open_spiel.python.algorithms import lp_solver
import pyspiel
import random
from NMM_train import MyGaussianPDF, GMMAgent, TorchPop

seed = 1234

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def renormalize(probabilities):
    """Replaces all negative entries with zeroes and normalizes the result.
    Args:
        probabilities: probability vector to renormalize. Has to be one-dimensional.

    Returns:
        Renormalized probabilities.
    """
    probabilities[probabilities < 0] = 0
    probabilities = probabilities / np.sum(probabilities)
    return probabilities

def lp_solver_func(meta_game):
    if not isinstance(meta_game, list):
        meta_game_l = [meta_game, -meta_game]
    else:
        meta_game_l = meta_game
    meta_game_l = [x.tolist() for x in meta_game_l]
    nash_prob_1, nash_prob_2, _, _ = (
        lp_solver.solve_zero_sum_matrix_game(
            pyspiel.create_matrix_game(*meta_game_l)))
    result = [
        renormalize(np.array(nash_prob_1).reshape(-1)),
        renormalize(np.array(nash_prob_2).reshape(-1))
    ]

    return result[0], result[1], result[0].dot(meta_game).dot(result[1])[0]

device="cpu"
LR=0.1

def expl_from_pop(pop):
    global seed
    seed += 1
    set_seed(seed)
    meta_game = pop.get_metagame(numpy=True)
    # metanash = fictitious_play(iters=1000, payoffs=meta_game)[0][-1]
    metanash, _, nash_value = lp_solver_func([np.array(meta_game), -np.array(meta_game)])
    br = pop.get_br_to_strat(metanash, 0.01, nb_iters=1000, early_stop=100)
    expl = pop.get_payoff_aggregate(br, metanash, pop.pop_size)
    return expl.item()

def get_br_to_strat(strat, payoffs, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

#Fictituous play as a nash equilibrium solver
def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0,1,(1,dim))
    pop = pop/pop.sum(axis=1)[:,None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average@payoffs@br.T
        exp2 = br@payoffs@average.T
        exps.append(exp2-exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


if True:
    PATH_RESULTS = "results/NMM"
    methods = ['psro', 'p-psro', 'rectified', 'dpp', 'psd', 'bd_rd']
    # methods = ['p-psro', 'dpp', 'psd', 'bd_rd']
    FILE_TRAJ = {
    'rectified': 'rectified.p',
    'psro': 'psro.p',
    'p-psro': 'p_psro.p',
    'dpp': 'dpp.p',
    'psd': 'psd.p', 
    'bd_rd': 'bd_rd.p'
            }

    exp_res = {}
    pe_res = {}

    for key in methods:
        print(key)
        for exp_id in range(5):
            data = pickle.load(open(os.path.join(PATH_RESULTS, FILE_TRAJ[key])+f'{exp_id}.p', 'rb'))
            EXP_ = expl_from_pop(data['pop'])
            exp_res.setdefault(key, []).append(EXP_)
            # print(exp_id, key, EXP_)

    pickle.dump(exp_res, open(os.path.join(PATH_RESULTS, 'exp_res.pkl'), 'wb'))

    for k in exp_res:
        print(k, end="\t")
        print(np.round(np.mean(exp_res[k]),4), end="\t")
        print(np.round(np.std(exp_res[k])/np.sqrt(5), 4))