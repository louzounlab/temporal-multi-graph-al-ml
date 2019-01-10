import itertools
from parameters import REFAEL_PARAM


""" BUILT FOR MULTICOLOR ONLY !!! """


def param_grid(kind):
    # kind is one of 'gblinear', 'gbtree' or 'dart'
    default_categories = {
        'silent': True,
        'objective': 'multi:softprob',
        'num_class': 4 if REFAEL_PARAM['task'] == "Multiclass1" else 8
    }

    if kind == 'gblinear':
        categories = {
            'booster': ['gblinear'],
            'lambda': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 5],
            'eta': [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075],
            'early_stopping_rounds': [100]
        }
        combs = list(itertools.product(
            categories['booster'], categories['lambda'], categories['eta'], categories['early_stopping_rounds']))
        grid_list = [{}] * len(combs)
        for i in range(len(combs)):
            grid_list[i] = default_categories.copy()
            grid_list[i]['booster'] = combs[i][0]
            grid_list[i]['lambda'] = combs[i][1]
            grid_list[i]['eta'] = combs[i][2]
            grid_list[i]['early_stopping_round'] = combs[i][3]
        return grid_list

    elif kind == 'gbtree':
        categories = {
            'booster': ['gbtree'],
            'lambda': [0.05, 0.075, 0.1, 0.125, 0.15],
            'eta': [0.05, 0.075, 0.1, 0.125, 0.15],
            'early_stopping_rounds': [100],
            'min_child_weight': [100],
            'ntree_limit': [0, 10, 1000],
            'gamma': [0, 0.1, 1, 10]
        }
        combs = list(itertools.product(
            categories['booster'], categories['lambda'], categories['eta'], categories['early_stopping_rounds'],
            categories['min_child_weight'], categories['ntree_limit'], categories['gamma']))
        grid_list = [{}] * len(combs)
        for i in range(len(combs)):
            grid_list[i] = default_categories.copy()
            grid_list[i]['booster'] = combs[i][0]
            grid_list[i]['lambda'] = combs[i][1]
            grid_list[i]['eta'] = combs[i][2]
            grid_list[i]['early_stopping_round'] = combs[i][3]
            grid_list[i]['min_child_weight'] = combs[i][4]
            grid_list[i]['ntree_limit'] = combs[i][5]
            grid_list[i]['gamma'] = combs[i][6]
        return grid_list

    elif kind == 'dart':
        categories = {
            'booster': ['dart'],
            'lambda': [0.01, 0.1, 1],
            'eta': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'early_stopping_rounds': [100],
            'max_depth': [3, 6, 9],
            'min_child_weight': [100],
            'subsample': [0.25, 0.5, 0.75, 1],
            'ntree_limit': [1000],
            'rate_drop': [0, 0.25, 0.5, 0.75],
            'gamma': [0, 0.1, 1, 10]
        }
        combs = list(itertools.product(
            categories['booster'], categories['lambda'], categories['eta'],
            categories['early_stopping_rounds'], categories['max_depth'], categories['min_child_weight'],
            categories['subsample'], categories['ntree_limit'], categories['rate_drop'], categories['gamma']))
        grid_list = [{}] * len(combs)
        for i in range(len(combs)):
            grid_list[i] = default_categories.copy()
            grid_list[i]['booster'] = combs[i][0]
            grid_list[i]['lambda'] = combs[i][1]
            grid_list[i]['eta'] = combs[i][2]
            grid_list[i]['early_stopping_round'] = combs[i][3]
            grid_list[i]['max_depth'] = combs[i][4]
            grid_list[i]['min_child_weight'] = combs[i][5]
            grid_list[i]['subsample'] = combs[i][6]
            grid_list[i]['ntree_limit'] = combs[i][7]
            grid_list[i]['rate_drop'] = combs[i][8]
            grid_list[i]['gamma'] = combs[i][9]
        return grid_list

