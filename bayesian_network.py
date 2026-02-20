""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
  
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    bayes_net = BayesNet()
    data = pickle.load(open("data/bn_data.p","rb"))

    def get_prob(target_attr, parent_conditions=None):
        """
        Calculates P(target_attr=True | parent_conditions)
        parent_conditions: dict {parent_name: value}
        """
        if parent_conditions is None:
            parent_conditions = {}
        matching_rows = []
        for d in data:
            match = True
            for attr, val in parent_conditions.items():
                if getattr(d, attr) != val:
                    match = False
                    break
            if match:
                matching_rows.append(d)
        
        if not matching_rows:
            return 0.0
        true_count = sum(1 for d in matching_rows if getattr(d, target_attr) is T)
        return true_count / len(matching_rows)
    p_mf = get_prob('muchfaster')
    p_early = get_prob('early')
    cpt_overtake = {}
    cpt_crash = {}
    
    for mf in [T, F]:
        for early in [T, F]:
            parents = {'muchfaster': mf, 'early': early}
            cpt_overtake[(mf, early)] = get_prob('overtake', parents)
            cpt_crash[(mf, early)] = get_prob('crash', parents)
    cpt_win = {}
    for overtake in [T, F]:
        for crash in [T, F]:
            parents = {'overtake': overtake, 'crash': crash}
            cpt_win[(overtake, crash)] = get_prob('win', parents)
    bayes_net = BayesNet([
        ('MuchFaster', '', p_mf),
        ('Early', '', p_early),
        ('Overtake', 'MuchFaster Early', cpt_overtake),
        ('Crash', 'MuchFaster Early', cpt_crash),
        ('Win', 'Overtake Crash', cpt_win)
    ])
    
    return bayes_net

def find_best_overtake_condition(bayes_net):
    best_prob = -1.0
    best_condition = (None, None)
    for mf in [T, F]:
        for early in [T, F]:
            evidence = {'MuchFaster': mf, 'Early': early}
            result_dist = enumeration_ask('Win', evidence, bayes_net)
            prob_win = result_dist[T]
            if prob_win > best_prob:
                best_prob = prob_win
                best_condition = (mf, early)

    return best_condition


def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()
