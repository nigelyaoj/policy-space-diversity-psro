from glob import escape
from sre_parse import FLAGS
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import stats
import pickle
import json
from numpy.random import RandomState
import argparse
import multiprocessing as mp
from tqdm import tqdm


np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

np.random.seed(0)

parser = argparse.ArgumentParser(description='Random Game Skill')
parser.add_argument('--nb_iters', type=int, default=200) # 200
parser.add_argument('--nb_exps', type=int, default=5) 
parser.add_argument('--mp', default=False, action='store_false', help='Set --mp for False, otherwise leave it for True')
parser.add_argument('--game_name', type=str, default='AlphaStar')

args = parser.parse_args()

LR = 0.5
TH = 0.03

expected_card = []
sizes = []

time_string = time.strftime("%Y%m%d-%H%M%S")
PATH_RESULTS = os.path.join('results', 'AlphaStar')
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)


# Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br


# Fictituous play as a nash equilibrium solver
def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0, 1, (1, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average @ payoffs @ br.T
        exp2 = br @ payoffs @ average.T
        exps.append(exp2 - exp1)
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


# Solve exploitability of a nash equilibrium over a fixed population
def get_exploitability(pop, payoffs, iters=1000):
    emp_game_matrix = pop @ payoffs @ pop.T
    averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)
    strat = averages[-1] @ pop  # Aggregate
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp1 = strat @ payoffs @ test_br.T
    exp2 = test_br @ payoffs @ strat
    return exp2 - exp1


def distance_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    cards = []

    if np.random.randn() < lambda_weight:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T
        br[np.argmax(values)] = 1
    else:
        for i in range(dim):
            br_tmp = np.zeros((dim,))
            br_tmp[i] = 1.

            pop_k = lr * br_tmp + (1 - lr) * pop[k]
            pop_tmp = np.vstack((pop[:k], pop_k))
            M = pop_tmp @ payoffs @ pop[:k].T
            old_payoff = M[0:-1].T
            new_vec = M[-1].reshape(-1, 1)
            distance = distance_solver(old_payoff, new_vec)
            cards.append(distance)
        br[np.argmax(cards)] = 1

    return br

def distance_solver(A, b):
    One = np.ones(shape=(A.shape[1], 1))
    I = np.identity(A.shape[0])
    A_pinv = np.linalg.pinv(A)
    I_minus_AA_pinv = I - A @ A_pinv
    Sigma_min = min(np.linalg.svd(A.T, full_matrices=True)[1])
    distance = ((Sigma_min ** 2) / A.shape[1]) * ((1 - (One.T @ A_pinv @ b)[0, 0]) ** 2) + np.square(
        I_minus_AA_pinv @ b).sum()
    return distance


def joint_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    values = []
    cards = []

    aggregated_enemy = meta_nash @ pop[:k]
    values = payoffs @ aggregated_enemy.T

    if np.random.randn() < lambda_weight:
        br[np.argmax(values)] = 1
    
    else:
        for i in range(dim):
            br_tmp = np.zeros((dim, ))
            br_tmp[i] = 1.

            aggregated_enemy = meta_nash @ pop[:k]
            pop_k = lr * br_tmp + (1 - lr) * pop[k]
            pop_tmp = np.vstack((pop[:k], pop_k))
            M = pop_tmp @ payoffs @ pop_tmp.T
            # metanash_tmp, _ = fictitious_play(payoffs=M, iters=1000)
            #L = np.diag(metanash_tmp[-1]) @ M @ M.T @ np.diag(metanash_tmp[-1])
            L = M @ M.T
            l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
            cards.append(l_card)
        br[np.argmax(cards)] = 1

    return br

def js_divergence(n, target_dist):
    def entropy(p_k):
        p_k = p_k + 1e-8
        p_k = p_k / sum(p_k)
        return -(p_k * np.log(p_k)).sum()

    original_dist = np.zeros(shape=target_dist.shape)
    original_dist[n] = 1
    return 2 * entropy(original_dist + target_dist) - entropy(original_dist) - entropy(target_dist)

def kl_divergence(prob_a, prob_b):
    prob_a += 1e-3
    prob_a /= prob_a.sum()
    prob_b += 1e-3
    prob_b = (prob_b.T / prob_b.sum(1)).T
    res = prob_a * np.log(prob_a/prob_b)
    return res.sum(1)

def divergence_loss(pop, payoffs, meta_nash, k, lambda_weight, lr, i):
    dim = payoffs.shape[0]
    br = np.zeros((dim,))
    if i <= 75:
        alpha = 500
    elif i <= 150:
        alpha = 100
    else:
        alpha = 50
    if np.random.randn() < lambda_weight:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T
        br[np.argmax(values)] = 1
        # print(f'Best Response {np.argmax(values)}')
    else:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T

        aggregated_enemy = aggregated_enemy.reshape(-1)
        # min_index = [i for i in range(len(aggregated_enemy)) if aggregated_enemy[i] == np.min(aggregated_enemy)]
        diverse_response = [values[i] + alpha * js_divergence(i, aggregated_enemy) for i in
                            range(len(aggregated_enemy))]
        selected_index = np.argmax(diverse_response)
        br[selected_index] = 1
        # print(f'Diverse: value[{np.argmax(values)}]={np.max(values)} diverse[{selected_index}]={np.max(diverse_response)}')

    return br


def fsp_non_symmetric_game(emp_game_matrix, iters=1000):
    row_player_dim = emp_game_matrix.shape[0]
    column_player_dim = emp_game_matrix.shape[1]
    row_avg = np.random.uniform(0, 1, row_player_dim)
    row_avg = row_avg / row_avg.sum()
    column_avg = np.random.uniform(0, 1, column_player_dim)
    column_avg = column_avg / column_avg.sum()
    for i in range(iters):
        # row_avg = np.average(row_pop, axis=0)
        # column_avg = np.average(column_pop, axis=0)
        br_column = get_br_to_strat(row_avg, emp_game_matrix)
        br_row = get_br_to_strat(column_avg, -emp_game_matrix.T)
        row_avg = (row_avg * (i+1) + br_row) / (i+2)
        column_avg = (column_avg * (i+1) + br_column) / (i+2)
    # row_avg = np.average(row_pop, axis=0)
    # column_avg = np.average(column_pop, axis=0)
    # print(f"Nash is {row_avg}")
    return - row_avg @ emp_game_matrix @ column_avg.T



def psd_update(pop, payoffs, meta_nash, k, lr, it):
    dim = payoffs.shape[0]
    cards = []
    lambda_weight = 0.85
    update_psd_term = 0.6

    aggregated_enemy = meta_nash @ pop[:k]
    values = payoffs @ aggregated_enemy.T
    if np.random.randn() < lambda_weight:
        br = np.zeros((dim,))
        br[np.argmax(values)] = 1
        pop[k] = lr * br + (1 - lr) * pop[k]
    else:
        br = np.zeros((dim,))
        for i in range(dim):
            br_tmp = np.zeros((dim, ))
            br_tmp[i] = 1.
            pop_k = br_tmp
            cards.append(kl_divergence(pop_k.copy(), pop[:k].copy()).min() * 0.01  + values[i])
        br[np.argmax(cards)] = 1
        pop[k] = update_psd_term * br + (1 - update_psd_term) * pop[k]


def psd_psro_steps(iters=5, payoffs=None, verbose=False, seed=0,
                        num_learners=4, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False):
    dim = payoffs.shape[0]

    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    l_cards = [exp]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    for i in tqdm(range(iters)):
        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash

            psd_update(pop, payoffs, meta_nash[-1], k, lr, i)
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            learner_performances[k].append(performance)

            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])

        # calculate exploitability for meta Nash of whole population
        exp = get_exploitability(pop, payoffs, iters=1000)
        print(f"Iteration: {i}, Exp: {exp}")
        exps.append(exp)

        emp_game_matrix = pop[:k] @ payoffs
        l_cards.append(fsp_non_symmetric_game(emp_game_matrix))

    return pop, exps, l_cards

def psro_steps(iters=5, payoffs=None, verbose=False, seed=0,
                        num_learners=4, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False):
    dim = payoffs.shape[0]

    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    l_cards = [exp]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    for i in tqdm(range(iters)):
        # Define the weighting towards diversity as a function of the fixed population size, this is currently a hyperparameter
        lambda_weight = 0.85

        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash

            if loss_func == 'br':
                # standard PSRO
                br = get_br_to_strat(population_strategy, payoffs=payoffs)
            elif loss_func == 'dpp':
                # Diverse PSRO
                br = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                br_orig = get_br_to_strat(population_strategy, payoffs=payoffs)
            elif loss_func == "bd_rd":
                if np.random.uniform() < 0.5:
                    br = divergence_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr, i)
                else:
                    br = distance_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
            else:
                raise

            # Update the mixed strategy towards the pure strategy which is returned as the best response to the
            # nash equilibrium that is being trained against.
            pop[k] = lr * br + (1 - lr) * pop[k]
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            learner_performances[k].append(performance)

            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])

        # calculate exploitability for meta Nash of whole population
        exp = get_exploitability(pop, payoffs, iters=1000)
        print(f"Iteration: {i}, Exp: {exp}")
        exps.append(exp)

        emp_game_matrix = pop[:k] @ payoffs
        l_cards.append(fsp_non_symmetric_game(emp_game_matrix))

    return pop, exps, l_cards


# Define the self-play algorithm
def self_play_steps(iters=10, payoffs=None, verbose=False, improvement_pct_threshold=.03, lr=.2, seed=0):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (2, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    performances = [.01]

    M = pop @ payoffs @ pop.T
    L = M@M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    for i in range(iters):
        br = get_br_to_strat(pop[-2], payoffs=payoffs)
        pop[-1] = lr * br + (1 - lr) * pop[-1]
        performance = pop[-1] @ payoffs @ pop[-2].T + 1
        performances.append(performance)
        if performance / performances[-2] - 1 < improvement_pct_threshold:
            learner = np.random.uniform(0, 1, (1, dim))
            learner = learner / learner.sum(axis=1)[:, None]
            pop = np.vstack((pop, learner))
        exp = get_exploitability(pop, payoffs, iters=1000)
        exps.append(exp)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        l_cards.append(l_card)

    return pop, exps, l_cards


# Define the PSRO rectified nash algorithm
def psro_rectified_steps(iters=10, payoffs=None, verbose=False, eps=1e-2, seed=0,
                         num_start_strats=1, num_pseudo_learners=4, lr=0.3, threshold=0.001):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (num_start_strats, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    counter = 0

    l_cards = [exp]

    while counter < iters * num_pseudo_learners:

        new_pop = np.copy(pop)
        emp_game_matrix = pop @ payoffs @ pop.T
        averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)

        # go through all policies. If the policy has positive meta Nash mass,
        # find policies it wins against, and play against meta Nash weighted mixture of those policies
        for j in range(pop.shape[0]):
            if counter > iters * num_pseudo_learners:
                return pop, exps, l_cards
            # if positive mass, add a new learner to pop and update it with steps, submit if over thresh
            # keep track of counter
            if averages[-1][j] > eps:
                # create learner
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                new_pop = np.vstack((new_pop, learner))
                idx = new_pop.shape[0] - 1

                current_performance = 0.02
                last_performance = 0.01
                while current_performance / last_performance - 1 > threshold:
                    counter += 1
                    mask = emp_game_matrix[j, :]
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    weights = np.multiply(mask, averages[-1])
                    weights /= weights.sum()
                    strat = weights @ pop
                    br = get_br_to_strat(strat, payoffs=payoffs)
                    new_pop[idx] = lr * br + (1 - lr) * new_pop[idx]
                    last_performance = current_performance
                    current_performance = new_pop[idx] @ payoffs @ strat + 1

                    if counter % num_pseudo_learners == 0:
                        # count this as an 'iteration'

                        # exploitability
                        exp = get_exploitability(new_pop, payoffs, iters=1000)
                        exps.append(exp)

                        emp_game_matrix2 = new_pop @ payoffs
                        l_cards.append(fsp_non_symmetric_game(emp_game_matrix2))

        pop = np.copy(new_pop)

    return pop, exps, l_cards


def run_experiment(param_seed):
    params, seed = param_seed
    iters = params['iters']
    num_threads = params['num_threads']
    lr = params['lr']
    thresh = params['thresh']
    psro = params['psro']
    pipeline_psro = params['pipeline_psro']
    dpp_psro = params['dpp_psro']
    rectified = params['rectified']
    self_play = params['self_play']
    psd_psro = params['psd_psro']
    bd_rd_psro = params["bd_rd_psro"]

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []
    psd_psro_exps = []
    psd_psro_cardinality = []
    bd_rd_psro_exps = []
    bd_rd_psro_cardinality = []

    print('Experiment: ', seed + 1)
    np.random.seed(seed)
    with open("payoffs_data/" + str(args.game_name) + ".pkl", "rb") as fh:
        payoffs = pickle.load(fh)
        payoffs /= np.abs(payoffs).max() 
    
    if psd_psro:
        pop, exps, cards = psd_psro_steps(iters=iters, num_learners=num_threads, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='psd')
        psd_psro_exps = exps
        psd_psro_cardinality = cards

    if psro:
        #print('PSRO')
        pop, exps, cards = psro_steps(iters=iters, num_learners=1, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='br')
        psro_exps = exps
        psro_cardinality = cards
    if pipeline_psro:
        #print('Pipeline PSRO')
        pop, exps, cards = psro_steps(iters=iters, num_learners=num_threads, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='br')
        pipeline_psro_exps = exps
        pipeline_psro_cardinality = cards
    if dpp_psro:
        #print('Pipeline DPP')
        pop, exps, cards = psro_steps(iters=iters, num_learners=num_threads, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='dpp')
        dpp_psro_exps = exps
        dpp_psro_cardinality = cards
    if rectified:
        #print('Rectified')
        pop, exps, cards = psro_rectified_steps(iters=iters, num_pseudo_learners=num_threads, payoffs=payoffs, seed=seed+1,
                                         lr=lr, threshold=thresh)
        rectified_exps = exps
        rectified_cardinality = cards
    if self_play:
        #print('Self-play')
        pop, exps, cards = self_play_steps(iters=iters, payoffs=payoffs, improvement_pct_threshold=thresh, lr=lr, seed=seed+1)
        self_play_exps = exps
        self_play_cardinality = cards
    if bd_rd_psro:
        pop, exps, cards = psro_steps(iters=iters, num_learners=num_threads, seed=seed+1,
                                    improvement_pct_threshold=thresh, lr=lr,
                                    payoffs=payoffs, loss_func='bd_rd')
        bd_rd_psro_exps = exps
        bd_rd_psro_cardinality = cards

    return {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
        'pipeline_psro_exps': pipeline_psro_exps,
        'pipeline_psro_cardinality': pipeline_psro_cardinality,
        'dpp_psro_exps': dpp_psro_exps,
        'dpp_psro_cardinality': dpp_psro_cardinality,
        'rectified_exps': rectified_exps,
        'rectified_cardinality': rectified_cardinality,
        'self_play_exps': self_play_exps,
        'self_play_cardinality': self_play_cardinality,
        'psd_psro_exps': psd_psro_exps,
        'psd_psro_cardinality': psd_psro_cardinality,
        "bd_rd_psro_exps": bd_rd_psro_exps,
        "bd_rd_psro_cardinality": bd_rd_psro_cardinality
    }


def run_experiments(num_experiments=2, iters=40, num_threads=20, lr=0.5, thresh=0.001, logscale=True,
                    psro=False,
                    pipeline_psro=False,
                    rectified=False,
                    self_play=False,
                    dpp_psro=False,
                    psd_psro=False,
                    bd_rd_psro=False
                    ):

    params = {
        'num_experiments': num_experiments,
        'iters': iters,
        'num_threads': num_threads,
        'lr': lr,
        'thresh': thresh,
        'psro': psro,
        'pipeline_psro': pipeline_psro,
        'dpp_psro': dpp_psro,
        'rectified': rectified,
        'self_play': self_play,
        'psd_psro': psd_psro,
        'bd_rd_psro': bd_rd_psro
    }

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []
    psd_psro_exps = []
    psd_psro_cardinality = []
    bd_rd_psro_exps = []
    bd_rd_psro_cardinality = []

    with open(os.path.join(PATH_RESULTS, 'params.json'), 'w', encoding='utf-8') as json_file:
        json.dump(params, json_file, indent=4)

    result = []

    #print(args.mp)
    if args.mp == False:
        for i in range(num_experiments):
            result.append(run_experiment((params, i)))

    else:
        pool = mp.Pool()
        result = pool.map(run_experiment, [(params, i) for i in range(num_experiments)])

    for r in result:
        psro_exps.append(r['psro_exps'])
        psro_cardinality.append(r['psro_cardinality'])
        pipeline_psro_exps.append(r['pipeline_psro_exps'])
        pipeline_psro_cardinality.append(r['pipeline_psro_cardinality'])
        dpp_psro_exps.append(r['dpp_psro_exps'])
        dpp_psro_cardinality.append(r['dpp_psro_cardinality'])
        rectified_exps.append(r['rectified_exps'])
        rectified_cardinality.append(r['rectified_cardinality'])
        self_play_exps.append(r['self_play_exps'])
        self_play_cardinality.append(r['self_play_cardinality'])
        psd_psro_exps.append(r['psd_psro_exps'])
        psd_psro_cardinality.append(r['psd_psro_cardinality'])
        bd_rd_psro_exps.append(r["bd_rd_psro_exps"])
        bd_rd_psro_cardinality.append(r["bd_rd_psro_cardinality"])

    d = {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
        'pipeline_psro_exps': pipeline_psro_exps,
        'pipeline_psro_cardinality': pipeline_psro_cardinality,
        'dpp_psro_exps': dpp_psro_exps,
        'dpp_psro_cardinality': dpp_psro_cardinality,
        'rectified_exps': rectified_exps,
        'rectified_cardinality': rectified_cardinality,
        'self_play_exps': self_play_exps,
        'self_play_cardinality': self_play_cardinality,
        'psd_psro_exps': psd_psro_exps, 
        'psd_psro_cardinality': psd_psro_cardinality,
        'bd_rd_psro_exps': bd_rd_psro_exps, 
        'bd_rd_psro_cardinality':bd_rd_psro_cardinality
    }

    pickle.dump(d, open(os.path.join(PATH_RESULTS, 'data.p'), 'wb'))


if __name__ == "__main__":
    start_time = time.time()
    run_experiments(num_experiments=args.nb_exps, num_threads=2, iters=args.nb_iters,  lr=.5, thresh=TH,
                    psro=True,
                    pipeline_psro=True,
                    rectified=True,
                    self_play=True,
                    dpp_psro=True,
                    psd_psro=True,
                    bd_rd_psro=True
                    )
    end_time = time.time()